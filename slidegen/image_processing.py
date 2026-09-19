"""Stage 2b - "Image Extraction" box of the pipeline: Image Processing -> Image Denoising -> Image Enhancement.

The processed copy is what LLaVA sees. The untouched original is what goes on the slide.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT_SETTINGS, Settings
from .models import ExtractedImage

MIN_LONG_SIDE = 640


def process_image(img: Image.Image, target_long_side: int) -> Image.Image:
    """Normalise mode (alpha -> white background, CMYK/P/L -> RGB) and bring the size into a sane range."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    long_side = max(img.size)
    if long_side < MIN_LONG_SIDE:                       # small figures: upsample so text becomes legible
        scale = min(MIN_LONG_SIDE / long_side, 4.0)
    elif long_side > target_long_side:
        scale = target_long_side / long_side
    else:
        return img
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)


def denoise_image(arr: np.ndarray, strength: int) -> np.ndarray:
    """Non-local-means denoising; mild by default so thin plot lines and small text survive."""
    if strength <= 0:
        return arr
    return cv2.fastNlMeansDenoisingColored(arr, None, strength, strength, 7, 21)


def enhance_image(arr: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel (local contrast) followed by a light unsharp mask (crisper edges/text)."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.6, blurred, -0.6, 0)


def prepare_for_analysis(image: ExtractedImage, out_dir: Path, settings: Settings = DEFAULT_SETTINGS) -> ExtractedImage:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image.path) as src:
        processed = process_image(src, settings.analysis_long_side_px)
    arr = enhance_image(denoise_image(np.asarray(processed), settings.denoise_strength))
    out_path = out_dir / f"{Path(image.path).stem}.png"
    Image.fromarray(arr).save(out_path)
    image.processed_path = str(out_path)
    return image
