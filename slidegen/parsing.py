"""Stage 1 - Document Parsing.

Turns a PDF into (a) clean body text and (b) a list of embedded images, each linked to its caption
and figure number so later stages can place figures next to the text that discusses them.
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pymupdf
from PIL import Image, ImageStat

from .config import DEFAULT_SETTINGS, Settings
from .models import ExtractedImage, ParsedDocument

_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?\.?)\s*(\d+)\s*[.:)\-]?\s*(.*)", re.IGNORECASE | re.DOTALL)
_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


# --------------------------------------------------------------------------- text
def clean_text(text: str) -> str:
    """Normalise PDF text: de-hyphenate, drop citations/URLs/page furniture, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                 # re-join hyphenated line breaks
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\[\d+(?:\s*[,–-]\s*\d+)*\]", "", text)   # [1], [2, 3], [4-6]
    text = re.sub(r"^\s*Page \d+ of \d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)  # bare page numbers
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def strip_references(text: str) -> str:
    """Cut the bibliography: everything from the last 'References' heading in the back half."""
    matches = [m for m in _REFERENCES_RE.finditer(text) if m.start() > len(text) * 0.4]
    return text[: matches[-1].start()] if matches else text


def _guess_title(doc: pymupdf.Document, fallback: str) -> str:
    """Title = the largest-font text block on page 1 (metadata is usually unreliable for papers)."""
    lines = []  # (font size, y, text) per visual line
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(s["text"].strip() for s in line["spans"] if s["text"].strip())
            if text:
                lines.append((round(max(s["size"] for s in line["spans"]), 1), line["bbox"][1], line["bbox"][3], text))
    if not lines:
        return fallback
    top = max(size for size, *_ in lines)
    big = sorted((y0, y1, t) for size, y0, y1, t in lines if size >= top - 0.6)
    # group vertically adjacent big lines; the real title is the longest group (a logo/journal name is usually short)
    groups: list[list[tuple[float, float, str]]] = [[big[0]]]
    for cur in big[1:]:
        if cur[0] - groups[-1][-1][1] < top * 0.9:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    title = " ".join(t for *_, t in max(groups, key=lambda g: sum(len(t) for *_, t in g)))
    title = re.sub(r"\s+", " ", title).strip()
    return title if 8 <= len(title) <= 200 else fallback


# --------------------------------------------------------------------------- images
def is_blank_image(img: Image.Image, threshold: float) -> bool:
    """True for (almost) uniformly black images - usually masks or decoding artefacts."""
    return sum(ImageStat.Stat(img.convert("RGB")).mean) / 3 < threshold


def _image_bytes(doc: pymupdf.Document, xref: int) -> tuple[bytes, str]:
    """Extract an image; re-attach its soft mask so transparent figures don't turn black."""
    info = doc.extract_image(xref)
    if info.get("smask"):
        pix = pymupdf.Pixmap(pymupdf.Pixmap(doc, xref), pymupdf.Pixmap(doc, info["smask"]))
        return pix.tobytes("png"), "png"
    return info["image"], info["ext"]


def _find_caption(page: pymupdf.Page, rect: pymupdf.Rect) -> tuple[int | None, str]:
    """Nearest 'Fig. N ...' text block that horizontally overlaps the image (below preferred)."""
    best: tuple[float, int, str] | None = None
    for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
        m = _CAPTION_RE.match(txt.strip())
        if not m or min(x1, rect.x1) - max(x0, rect.x0) <= 0:
            continue
        gap = y0 - rect.y1 if y0 >= rect.y1 - 5 else (rect.y0 - y1) + 40   # penalise captions above
        if gap < -5 or gap > 150:
            continue
        caption = re.sub(r"\s+", " ", txt).strip()
        if best is None or gap < best[0]:
            best = (gap, int(m.group(1)), caption[:300])
    return (best[1], best[2]) if best else (None, "")


def extract_images(doc: pymupdf.Document, out_dir: Path, settings: Settings = DEFAULT_SETTINGS) -> list[ExtractedImage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    images: list[ExtractedImage] = []
    for page in doc:
        idx = 0
        for xref, *_ in page.get_images(full=True):
            try:
                data, ext = _image_bytes(doc, xref)
                pil = Image.open(io.BytesIO(data))
                pil.load()
            except Exception:
                continue
            w, h = pil.size
            digest = hashlib.md5(data).hexdigest()
            if (min(w, h) < settings.min_image_side_px or w * h < settings.min_image_area_px
                    or digest in seen or is_blank_image(pil, settings.black_image_mean)):
                continue
            seen.add(digest)
            idx += 1
            ext = ext if ext in ("jpeg", "jpg", "png") else "png"
            path = out_dir / f"page{page.number + 1:02d}_img{idx}.{ext}"
            if pil.mode not in ("RGB", "RGBA", "L"):
                pil = pil.convert("RGB")
            if ext in ("jpeg", "jpg") and pil.mode == "RGBA":
                pil = pil.convert("RGB")
            pil.save(path)
            rects = page.get_image_rects(xref)
            fig_no, caption = _find_caption(page, rects[0]) if rects else (None, "")
            images.append(ExtractedImage(str(path), page.number + 1, idx, w, h, caption, fig_no))
    return images


# --------------------------------------------------------------------------- entry point
def parse_document(pdf_path: str | Path, work_dir: str | Path, settings: Settings = DEFAULT_SETTINGS) -> ParsedDocument:
    pdf_path, work_dir = Path(pdf_path), Path(work_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pymupdf.open(pdf_path) as doc:
        raw = "\n".join(page.get_text("text") for page in doc)
        text = clean_text(strip_references(raw))
        if len(text) < 200:
            raise ValueError(f"No extractable text in {pdf_path.name} (scanned PDF? OCR is not supported).")
        title = _guess_title(doc, pdf_path.stem.replace("_", " "))
        images = extract_images(doc, work_dir / "images" / "original", settings)
        return ParsedDocument(str(pdf_path), sha, title, len(doc), text, images)
