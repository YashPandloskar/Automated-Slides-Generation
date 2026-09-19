"""Central configuration. Every tunable lives here; nothing else hard-codes paths or model names."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- models (Ollama) -------------------------------------------------
    text_model: str = field(default_factory=lambda: _env("SLIDEGEN_TEXT_MODEL", "llama3.1"))
    vision_model: str = field(default_factory=lambda: _env("SLIDEGEN_VISION_MODEL", "llava"))
    # Embedding model for the dense (Chroma) index. If it is not installed the pipeline falls back to BM25;
    # run `ollama pull nomic-embed-text` to enable dense retrieval.
    embed_model: str = field(default_factory=lambda: _env("SLIDEGEN_EMBED_MODEL", "nomic-embed-text"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    llm_timeout_s: float = 300.0
    llm_retries: int = 2

    # --- document parsing / RAG -----------------------------------------
    chunk_chars: int = 900
    chunk_overlap_chars: int = 150
    retrieval_top_k: int = 6
    context_char_budget: int = 4500

    # --- image extraction ------------------------------------------------
    min_image_side_px: int = 120         # drops logos / icons / rules
    min_image_area_px: int = 20_000
    black_image_mean: float = 12.0       # mean brightness (0-255) below which an image counts as blank
    analysis_long_side_px: int = 1024    # size handed to LLaVA
    denoise_strength: int = 3            # cv2 fastNlMeans "h"; 0 disables denoising

    # --- slides ----------------------------------------------------------
    max_bullets_per_slide: int = 5
    max_bullet_words: int = 28


DEFAULT_SETTINGS = Settings()
REPO_ROOT = Path(__file__).resolve().parent.parent
