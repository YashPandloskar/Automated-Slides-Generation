"""Quality metrics for the generated text.

ROUGE is computed against the RAG context each section was generated from (what the model actually saw), so
  * precision  = share of the summary's words/phrases that come from the source  (grounding)
  * recall     = share of the context that survives in the summary               (expected to be low for a summary)
The old notebook compared against a hand-picked 3000-character window, which made recall meaningless.
"""
from __future__ import annotations

import re
from collections import Counter

from .models import SectionResult

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _prf(overlap: int, n_pred: int, n_ref: int) -> dict[str, float]:
    p = overlap / n_pred if n_pred else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _ngram_overlap(ref: list[str], pred: list[str], n: int) -> dict[str, float]:
    grams = lambda t: Counter(zip(*(t[i:] for i in range(n))))
    r, p = grams(ref), grams(pred)
    return _prf(sum((r & p).values()), sum(p.values()), sum(r.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge(reference: str, prediction: str) -> dict[str, dict[str, float]]:
    ref, pred = tokenize(reference), tokenize(prediction)
    return {
        "rouge1": _ngram_overlap(ref, pred, 1),
        "rouge2": _ngram_overlap(ref, pred, 2),
        "rougeL": _prf(lcs_length(ref, pred), len(pred), len(ref)),
    }


def ungrounded_numbers(source: str, generated: str) -> list[str]:
    """Numbers that appear in the generated text but nowhere in the source - candidate hallucinations."""
    known = set(_NUMBER.findall(source))
    return sorted({n for n in _NUMBER.findall(generated) if n not in known})


def evaluate_sections(sections: list[SectionResult], full_text: str) -> dict:
    per_section = {}
    for s in sections:
        if not s.bullets:
            continue
        generated = " ".join(s.bullets)
        per_section[s.key] = {
            "title": s.title,
            "rouge": rouge(s.retrieved, generated),
            "ungrounded_numbers": ungrounded_numbers(full_text, generated + " " + s.notes),
            "n_bullets": len(s.bullets),
        }
    def avg(metric: str, field: str) -> float:
        vals = [v["rouge"][metric][field] for v in per_section.values()]
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "average": {m: {f: round(avg(m, f), 4) for f in ("precision", "recall", "f1")} for m in ("rouge1", "rouge2", "rougeL")},
        "sections": per_section,
    }
