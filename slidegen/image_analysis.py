"""Stage 3 - "Data Extraction using LLaVA".

    graph?  --yes-->  Graph Detection -> Graph Type Identification -> Data Extraction from Graphs
            --no-->   Other Image Types -> General Image Analysis -> { Object Detection , Text-in-Image OCR }

Every step is a small, single-purpose prompt that returns JSON (7B vision models are far more reliable
with narrow questions than with one long multi-part prompt).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import DEFAULT_SETTINGS, Settings
from .llm import LLMClient, parse_json
from .models import ExtractedImage, ImageAnalysis

PROMPT_VERSION = 3
GRAPH_TYPES = ["bar", "line", "scatter", "pie", "histogram", "heatmap", "confusion_matrix", "roc_curve",
               "box_plot", "area", "other"]
NON_GRAPH_KINDS = ["flowchart", "diagram", "table", "photo", "screenshot", "other"]

_P_DETECT = (
    "Does this image plot quantitative data as a graph or chart (bar chart, line chart, scatter plot, pie chart, "
    "histogram, heatmap, confusion matrix, ROC curve, box plot)? Flowcharts, architecture diagrams, tables, "
    "photographs and screenshots are NOT graphs. "
    'Reply as JSON: {"is_graph": true or false}'
)
_P_TYPE = (
    f"This image is a graph. Identify it. graph_type must be one of {GRAPH_TYPES}. "
    'Reply as JSON: {"graph_type": "...", "title": "<title or empty>", "x_axis": "<x label or empty>", '
    '"y_axis": "<y label or empty>"}'
)
_P_DATA = (
    "Read the data values from this graph. Only include values you can actually read from the image; never guess. "
    "At most 10 points per series. "
    'Reply as JSON: {"series": [{"name": "<series name>", "points": [{"label": "<x category or value>", "value": <number>}]}]}'
)
_P_GENERAL = (
    f"What kind of image is this? kind must be one of {NON_GRAPH_KINDS}. "
    'Describe it in ONE factual sentence. Reply as JSON: {"kind": "...", "description": "..."}'
)
_P_OBJECTS = (
    "List the distinct objects, components or labelled elements visible in this image (at most 10 short noun phrases). "
    'Reply as JSON: {"objects": ["...", "..."]}'
)
_P_OCR = (
    "Transcribe all legible text in this image in reading order. Do not invent text; if there is none use an empty string. "
    'Reply as JSON: {"text": "..."}'
)


def _ask(llm: LLMClient, model: str, prompt: str, image_path: str, max_tokens: int = 150) -> dict[str, Any]:
    """One narrow question to LLaVA. Forced JSON mode is avoided: with LLaVA it tends to pad output up to the token cap."""
    try:
        data = parse_json(llm.chat(model, prompt, images=[image_path], max_tokens=max_tokens))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_series(raw: Any) -> list[dict[str, Any]]:
    """Keep only series whose points have a label and a finite numeric value."""
    out: list[dict[str, Any]] = []
    for s in raw if isinstance(raw, list) else []:
        points = []
        for p in (s.get("points", []) if isinstance(s, dict) else []):
            try:
                value = float(str(p["value"]).replace("%", "").replace(",", ""))
            except (KeyError, TypeError, ValueError):
                continue
            label = str(p.get("label", "")).strip()
            if label and value == value and abs(value) != float("inf"):
                points.append({"label": label, "value": value})
        if len(points) >= 2 and not (len(points) >= 3 and len({pt["value"] for pt in points}) == 1):   # constant = made up
            out.append({"series": str(s.get("name", "")).strip(), "points": points[:10]})
    return out


_STOP = {"the", "and", "of", "for", "in", "on", "with", "a", "an", "to", "is", "are", "by", "fig", "figure", "image", "chart",
         "graph", "diagram", "shows", "showing", "different", "using", "based"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def check_against_caption(image: ExtractedImage, a: ImageAnalysis) -> str:
    """LLaVA-7B regularly hallucinates titles/values for charts, so its output is cross-checked with the caption
    printed in the PDF: 'consistent' if they share vocabulary, 'conflict' if not, 'unverified' if there is no caption."""
    if not image.caption:
        return "unverified"
    described = _words(" ".join([a.title, a.description, a.text_in_image, *a.objects]))
    return "consistent" if _words(image.caption) & described else "conflict"


def analyze_image(image: ExtractedImage, llm: LLMClient, settings: Settings = DEFAULT_SETTINGS) -> ImageAnalysis:
    path = image.processed_path or image.path
    model = settings.vision_model
    result = ImageAnalysis(image=image.name)
    try:
        if _ask(llm, model, _P_DETECT, path, 30).get("is_graph") is True:              # Graph Detection
            result.is_graph, result.kind = True, "graph"
            t = _ask(llm, model, _P_TYPE, path, 120)                                    # Graph Type Identification
            gt = str(t.get("graph_type", "other")).lower().replace(" ", "_")
            result.graph_type = gt if gt in GRAPH_TYPES else "other"
            result.title, result.x_axis, result.y_axis = (str(t.get(k, "")).strip() for k in ("title", "x_axis", "y_axis"))
            result.data = _clean_series(_ask(llm, model, _P_DATA, path, 450).get("series"))   # Data Extraction from Graphs
            result.description = " ".join(filter(None, [
                f"{result.graph_type.replace('_', ' ').capitalize()} chart", f"of {result.title}" if result.title else ""]))
        else:                                                                      # Other Image Types
            g = _ask(llm, model, _P_GENERAL, path, 120)                                 # General Image Analysis
            kind = str(g.get("kind", "other")).lower()
            result.kind = kind if kind in NON_GRAPH_KINDS else "other"
            result.description = str(g.get("description", "")).strip()
        objs = _ask(llm, model, _P_OBJECTS, path, 120).get("objects", [])               # Object Detection
        result.objects = [str(o).strip() for o in objs if str(o).strip()][:10] if isinstance(objs, list) else []
        result.text_in_image = str(_ask(llm, model, _P_OCR, path, 250).get("text", "")).strip()   # Text-in-Image OCR
    except Exception as exc:  # one bad image must not sink the whole document
        result.error = str(exc)
    result.confidence = check_against_caption(image, result)
    return result


def _cache_key(image: ExtractedImage, settings: Settings) -> str:
    digest = hashlib.md5(Path(image.processed_path or image.path).read_bytes()).hexdigest()
    return f"{digest}:{settings.vision_model}:v{PROMPT_VERSION}"


def analyze_images(images: list[ExtractedImage], llm: LLMClient, cache_file: Path,
                   settings: Settings = DEFAULT_SETTINGS, progress=print) -> dict[str, ImageAnalysis]:
    """Analyse all images, re-using cached results for unchanged images."""
    cache: dict[str, Any] = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    results: dict[str, ImageAnalysis] = {}
    for i, image in enumerate(images, 1):
        key = _cache_key(image, settings)
        if key in cache:
            results[image.name] = ImageAnalysis(**cache[key])
            continue
        progress(f"  [{i}/{len(images)}] LLaVA analysing {image.name}")
        analysis = analyze_image(image, llm, settings)
        results[image.name] = analysis
        if not analysis.error:
            cache[key] = analysis.__dict__
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return results
