"""Orchestrates the whole pipeline, mirroring the architecture diagram (see README).

    Input -> Document Parsing -+-> [Text branch]  RAG (Llama 3.1): Text Extraction -> Text Analysis --------+
                               |                                                                            +-> Context Synthesis -> context.json -> slides.pptx
                               +-> [Image branch] Image Extraction (process/denoise/enhance) -> LLaVA ----+
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings
from .evaluation import evaluate_sections
from .image_analysis import analyze_images
from .image_processing import prepare_for_analysis
from .llm import LLMClient, OllamaClient
from .models import SlideContext
from .parsing import parse_document
from .slides import build_presentation
from .synthesis import synthesize
from .text_rag import SECTION_SPECS, DocumentIndex, analyze_section, generate_tagline


def run_pipeline(pdf: str | Path, out_dir: str | Path, settings: Settings = DEFAULT_SETTINGS,
                 llm: LLMClient | None = None, use_images: bool = True, show_chart_values: bool = False,
                 progress=print) -> dict[str, Path]:
    pdf, out_dir = Path(pdf), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = llm or OllamaClient(settings)

    progress("1/5 Document parsing")
    doc = parse_document(pdf, out_dir, settings)
    progress(f"    {doc.n_pages} pages, {len(doc.text):,} chars of text, {len(doc.images)} usable images")

    progress("2/5 Text branch: RAG with " + settings.text_model)
    index = DocumentIndex(out_dir / "chroma", llm, settings)
    n_chunks = index.build(doc)
    note = f" (embeddings unavailable: {index.fallback_reason}; `ollama pull {settings.embed_model}` enables Chroma)" if index.fallback_reason else ""
    progress(f"    indexed {n_chunks} chunks, retrieval backend: {index.backend}{note}")
    sections = []
    for spec in filter(lambda s: s.enabled, SECTION_SPECS):
        progress(f"    analysing: {spec.title}")
        sections.append(analyze_section(spec, index, llm, settings))
    tagline = generate_tagline(index, llm, settings)

    analyses = {}
    if use_images and doc.images:
        progress("3/5 Image branch: extraction (process/denoise/enhance) + " + settings.vision_model)
        for image in doc.images:
            prepare_for_analysis(image, out_dir / "images" / "processed", settings)
        analyses = analyze_images(doc.images, llm, out_dir / "image_analysis_cache.json", settings, progress)
        (out_dir / "image_analysis.json").write_text(
            json.dumps({k: asdict(v) for k, v in analyses.items()}, indent=2), encoding="utf-8")
    else:
        progress("3/5 Image branch skipped")

    progress("4/5 Context synthesis")
    context = synthesize(doc.title, tagline, pdf.name, sections, doc.images, analyses, show_chart_values)
    context_path = out_dir / "context.json"
    context_path.write_text(json.dumps(context.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    progress("5/5 Slide generation")
    pptx_path = build_presentation(context, out_dir / f"{pdf.stem}.pptx")
    report = evaluate_sections(sections, doc.text)
    report_path = out_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    avg = report["average"]
    progress(f"    ROUGE-1 P/R/F1 {avg['rouge1']['precision']:.2f}/{avg['rouge1']['recall']:.2f}/{avg['rouge1']['f1']:.2f}  "
             f"ROUGE-L F1 {avg['rougeL']['f1']:.2f}")
    return {"context": context_path, "pptx": pptx_path, "evaluation": report_path}


def rebuild_slides(context_path: str | Path, out_path: str | Path) -> Path:
    """Regenerate the deck from a saved (and possibly hand-edited) context.json without re-running any model."""
    ctx = SlideContext.from_dict(json.loads(Path(context_path).read_text(encoding="utf-8")))
    return build_presentation(ctx, out_path)
