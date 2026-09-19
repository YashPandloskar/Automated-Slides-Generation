"""Thin wrapper over Ollama so every stage shares retries, JSON parsing and one seam for test doubles."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from .config import DEFAULT_SETTINGS, Settings


class LLMClient(Protocol):
    def chat(self, model: str, prompt: str, *, system: str = "", images: list[str] | None = None,
             json_mode: bool = False, max_tokens: int = 700) -> str: ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


class OllamaClient:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        import ollama  # imported lazily so the rest of the package works without it

        self._settings = settings
        self._client = ollama.Client(host=settings.ollama_host, timeout=settings.llm_timeout_s)

    def chat(self, model: str, prompt: str, *, system: str = "", images: list[str] | None = None,
             json_mode: bool = False, max_tokens: int = 700) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        messages.append(user)

        last_err: Exception | None = None
        for attempt in range(self._settings.llm_retries + 1):
            try:
                resp = self._client.chat(
                    model=model, messages=messages, format="json" if json_mode else None,
                    options={"temperature": 0.2, "seed": 42, "num_predict": max_tokens},   # seed: reproducible reruns; cap: stops runaway generation
                )
                return resp["message"]["content"].strip()
            except Exception as exc:  # network hiccup, model still loading, ...
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed for model '{model}': {last_err}") from last_err

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(model=model, input=texts)
        return [list(v) for v in resp["embeddings"]]


def parse_json(raw: str) -> Any:
    """Parse model output as JSON, tolerating code fences and prose around the object."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Model did not return valid JSON: {raw[:200]!r}")
