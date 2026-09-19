"""Stage 4 - Context Synthesis: merge the text branch and the image branch into one slide-ready context.

Figures are attached to the section they belong to using three deterministic signals
(explicit "Fig. N" mentions, image-type affinity, keyword overlap) so results are reproducible.
"""
from __future__ import annotations

import re

from .models import ExtractedImage, Figure, ImageAnalysis, SectionResult, SlideContext, SlideSection

MAX_FIGURES = {"methods": 2, "results": 4, "stats": 2}     # everything else: 1
DUPLICATE_JACCARD = 0.6
MIN_ASSIGNMENT_SCORE = 3.0

# which kinds of figure each section prefers (weight added to the score)
_AFFINITY: dict[str, dict[str, float]] = {
    "intro": {"diagram": 1.0, "flowchart": 1.0, "photo": 0.5},
    "topics": {"diagram": 1.0, "flowchart": 1.0},
    "methods": {"flowchart": 3.0, "diagram": 3.0, "table": 0.5},
    "results": {"graph": 3.0, "table": 1.0},
    "stats": {"graph": 3.0, "table": 2.0},
    "applications": {"diagram": 1.0, "photo": 1.0},
    "limits": {},
    "future": {},
    "conclusion": {"diagram": 0.5},
}

_STOP = set("""a an the and or of to in on for with by from as at is are was were be been this that these those it its
we our their they which using used use based can also such than then into over per via not no more most other new one two
figure fig image shows show shown chart graph diagram""".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if w not in _STOP}


def figure_data_line(a: ImageAnalysis) -> str:
    """One readable line from extracted graph data, e.g. 'RF 0.91 | XGB 0.93 | LGBM 0.92'."""
    if not a.data:
        return ""
    series = max(a.data, key=lambda s: len(s["points"]))
    pts = " | ".join(f"{p['label']} {p['value']:g}" for p in series["points"][:6])
    prefix = f"{series['series']}: " if series["series"] else ""
    return f"{prefix}{pts}"


def clean_caption(caption: str, analysis: ImageAnalysis, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", caption).strip()
    if not text:
        text = analysis.description or analysis.title
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return text


_CAPTION_KIND = (
    ("diagram", re.compile(r"\b(approach|framework|architecture|workflow|flow ?chart|pipeline|overview|schematic|"
                           r"model structure|proposed (?:model|method)|block diagram)\b", re.IGNORECASE)),
    ("graph", re.compile(r"\b(graph|plot|chart|curve|matrix|histogram|heat ?map|comparison|accuracy|loss|forecast\w*)\b",
                         re.IGNORECASE)),
)


def effective_kind(image: ExtractedImage, a: ImageAnalysis) -> str:
    """LLaVA's kind, or - when it could not classify the image ('other') - a kind inferred from the PDF caption."""
    if a.kind != "other":
        return a.kind
    for kind, pattern in _CAPTION_KIND:
        if pattern.search(image.caption):
            return kind
    return "other"


def trusted_details(image: ExtractedImage, a: ImageAnalysis) -> str:
    """Text describing the image that may be used for matching/captions: the PDF caption always, LLaVA's
    description/OCR only when it did not contradict that caption (see image_analysis.check_against_caption)."""
    parts = [image.caption]
    if a.confidence != "conflict":
        parts += [a.description, a.title, *a.objects, a.text_in_image]
    return " ".join(p for p in parts if p)


def _score(image: ExtractedImage, analysis: ImageAnalysis, section: SectionResult) -> float:
    affinity = _AFFINITY.get(section.key, {}).get(effective_kind(image, analysis), 0.0)
    if affinity <= 0:
        return 0.0            # a figure only goes where its type belongs (never a confusion matrix under "Limitations")
    score = 2.0 * affinity
    if image.figure_no is not None:
        ref = re.compile(rf"\bfig(?:ure|s)?\.?\s*{image.figure_no}\b", re.IGNORECASE)
        if ref.search(" ".join([*section.bullets, section.notes])):
            score += 3.0                                # discussed in the slide text itself
        elif ref.search(section.retrieved):
            score += 1.5                                # only mentioned in the retrieved source passages
    image_terms = _tokens(trusted_details(image, analysis))
    body_terms = _tokens(" ".join([*section.bullets, section.notes]))
    if image_terms and body_terms:
        score += 6.0 * len(image_terms & body_terms) / min(len(image_terms), 25)
    return score


def assign_figures(images: list[ExtractedImage], analyses: dict[str, ImageAnalysis],
                   sections: list[SectionResult], show_chart_values: bool = False) -> dict[str, list[Figure]]:
    pairs = []
    for image in images:
        analysis = analyses.get(image.name)
        if analysis is None or analysis.error:
            continue
        for section in sections:
            if section.bullets:
                pairs.append((_score(image, analysis, section), image, analysis, section))
    pairs.sort(key=lambda p: -p[0])
    assigned: dict[str, list[Figure]] = {s.key: [] for s in sections}
    used: set[str] = set()
    for score, image, analysis, section in pairs:
        if score < MIN_ASSIGNMENT_SCORE:
            break
        if image.name in used or len(assigned[section.key]) >= MAX_FIGURES.get(section.key, 1):
            continue
        used.add(image.name)
        trusted = analysis.confidence != "conflict"
        assigned[section.key].append(Figure(
            path=image.path, caption=clean_caption(image.caption, analysis if trusted else ImageAnalysis(image.name)),
            kind=effective_kind(image, analysis), summary=analysis.description if trusted else "",
            data_line=figure_data_line(analysis) if show_chart_values and analysis.confidence == "consistent" else "",
            width=image.width, height=image.height,
        ))
    return assigned


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def dedupe_sections(sections: list[SectionResult]) -> list[SectionResult]:
    """Drop bullets that repeat an earlier section's bullet; a section left with < 2 bullets is redundant and removed."""
    seen: list[str] = []
    out: list[SectionResult] = []
    for sec in sections:
        kept = [b for b in sec.bullets if all(_jaccard(b, prev) < DUPLICATE_JACCARD for prev in seen)]
        if len(kept) >= 2:
            seen.extend(kept)
            out.append(SectionResult(sec.key, sec.title, kept, sec.notes, sec.retrieved))
    return out


def synthesize(doc_title: str, tagline: str, source: str, sections: list[SectionResult],
               images: list[ExtractedImage], analyses: dict[str, ImageAnalysis],
               show_chart_values: bool = False) -> SlideContext:
    usable = dedupe_sections([s for s in sections if s.bullets])
    figures = assign_figures(images, analyses, usable, show_chart_values)
    return SlideContext(
        title=doc_title, subtitle=tagline, source=source,
        sections=[SlideSection(s.key, s.title, s.bullets, s.notes, figures.get(s.key, [])) for s in usable],
    )
