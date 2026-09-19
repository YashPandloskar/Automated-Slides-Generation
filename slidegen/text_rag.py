"""Stage 2a - Text branch: "RAG using Llama 3.1" = Text Extraction (chunk + index) -> Text Analysis (retrieve + generate)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings
from .llm import LLMClient, parse_json
from .models import ParsedDocument, SectionResult

# --------------------------------------------------------------------------- chunking
_ABBREVIATIONS = re.compile(
    r"\b(Fig|Figs|Eq|Eqs|Tab|Ref|Refs|et al|e\.g|i\.e|vs|cf|approx|No|Dr|Prof|Sec|Inc|Ltd)\.", re.IGNORECASE
)
_DOT = "․"  # one-dot leader, used to hide abbreviation dots from the splitter


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that does not break on 'Fig. 3', 'et al.', 'e.g.' or decimals."""
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_DOT}\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    """Pack whole sentences into chunks of <= max_chars, carrying trailing sentences as overlap."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        if current and size + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):          # sentence-level overlap
                if carried + len(prev) > overlap_chars:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, size = carry, carried
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks



# --------------------------------------------------------------------------- lexical fallback
_STOP = frozenset("""a an the and or of to in on for with by from as at is are was were be been this that these those it its
we our their they which using used based can also such than then into over per not no more most other""".split())


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


class BM25:
    """Okapi BM25 over chunks - the no-download fallback when no embedding model is available."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(_terms(d)) for d in docs]
        self.len = [sum(t.values()) or 1 for t in self.tf]
        self.avg = sum(self.len) / max(1, len(docs))
        df: Counter[str] = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def top(self, query: str, k: int) -> list[int]:
        q = set(_terms(query))
        scores = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            scores.append(sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

# --------------------------------------------------------------------------- vector index
class DocumentIndex:
    """Retrieval index for one PDF. Dense (Chroma + Ollama embeddings, keyed by PDF content hash) when an
    embedding model is available, otherwise BM25. `backend` records which one is in use."""

    def __init__(self, persist_dir: Path, llm: LLMClient, settings: Settings = DEFAULT_SETTINGS):
        self._persist_dir = persist_dir
        self._llm = llm
        self.settings = settings
        self.chunks: list[str] = []
        self.backend = "bm25"
        self.fallback_reason = ""
        self._collection = None
        self._bm25: BM25 | None = None

    def build(self, doc: ParsedDocument) -> int:
        s = self.settings
        self.chunks = chunk_text(doc.text, s.chunk_chars, s.chunk_overlap_chars)
        self._bm25 = BM25(self.chunks)
        try:
            self._build_dense(doc)
            self.backend = "chroma"
        except Exception as exc:  # no embedding model / Ollama can't embed -> lexical retrieval still works
            self._collection, self.backend = None, "bm25"
            self.fallback_reason = str(exc).splitlines()[0][:120]
        return len(self.chunks)

    def _build_dense(self, doc: ParsedDocument) -> None:
        import chromadb

        s = self.settings
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._persist_dir))
        name = f"doc_{doc.sha256[:16]}_{re.sub(r'[^A-Za-z0-9]+', '', s.embed_model)[:20]}_{s.chunk_chars}"[:63]
        existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
        if name in existing:
            collection = client.get_collection(name)
            if collection.count() == len(self.chunks):
                self._collection = collection
                return
            client.delete_collection(name)
        collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(self.chunks), 16):
            part = self.chunks[start:start + 16]
            collection.add(
                ids=[f"chunk_{start + i}" for i in range(len(part))], documents=part,
                embeddings=self._llm.embed(s.embed_model, part),
                metadatas=[{"chunk_id": start + i} for i in range(len(part))],
            )
        self._collection = collection

    def _search(self, query: str, k: int) -> set[int]:
        k = min(k, len(self.chunks))
        if self._collection is not None:
            res = self._collection.query(query_embeddings=self._llm.embed(self.settings.embed_model, [query]), n_results=k)
            return {int(m["chunk_id"]) for m in res["metadatas"][0]}
        assert self._bm25 is not None, "call build() first"
        return set(self._bm25.top(query, k))

    def retrieve(self, query: str, k: int, anchor: str | None = None, char_budget: int | None = None) -> str:
        """Top-k chunks re-ordered into document order; optionally anchored to the start/end of the paper."""
        ids = self._search(query, k)
        if anchor == "start":
            ids |= {0, 1}
        elif anchor == "end":
            ids |= {len(self.chunks) - 2, len(self.chunks) - 1}
        ordered = [self.chunks[i] for i in sorted(ids) if 0 <= i < len(self.chunks)]
        text = "\n\n".join(ordered)
        budget = char_budget or self.settings.context_char_budget
        return text if len(text) <= budget else text[:budget].rsplit(" ", 1)[0]


# --------------------------------------------------------------------------- text analysis
@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    query: str
    focus: str
    anchor: str | None = None
    enabled: bool = True


SECTION_SPECS: list[SectionSpec] = [
    SectionSpec("intro", "Introduction",
                "title, research question, objective and background context of the paper",
                "the problem being addressed, why it matters, and the paper's objective", anchor="start"),
    SectionSpec("topics", "Key Topics",
                "main topics, themes and major sections covered in the paper",
                "the main themes and how the paper is organised"),
    SectionSpec("definitions", "Key Terms",
                "specialised terminology, concepts and models that are defined",
                "important technical terms, each with a short definition", enabled=False),
    SectionSpec("methods", "Methodology",
                "methods, techniques, algorithms, experimental design, data collection and analysis approach",
                "the datasets, models, and steps of the approach"),
    SectionSpec("results", "Findings & Results",
                "primary results, findings, outcomes, performance measurements and discoveries",
                "what was found, with the specific outcomes and numbers reported"),
    SectionSpec("stats", "Key Statistics",
                "quantitative data, metrics, accuracy, percentages, statistical analysis, tables of numbers",
                "the most important reported numbers and what each measures"),
    SectionSpec("applications", "Applications",
                "real-world applications, practical use cases and implementations",
                "where and how the work can be applied"),
    SectionSpec("limits", "Challenges & Limitations",
                "limitations, constraints, challenges, weaknesses and threats to validity",
                "the acknowledged limitations and open challenges"),
    SectionSpec("future", "Future Scope",
                "future work, suggestions for further research, recommendations and next steps",
                "what the authors propose to do next", anchor="end"),
    SectionSpec("conclusion", "Conclusion",
                "main conclusions, key takeaways, summary of contributions and significance",
                "the main conclusions and contributions", anchor="end"),
]

SYSTEM_PROMPT = (
    "You are an expert at turning scientific papers into clear presentation slides. "
    "You only use facts stated in the supplied excerpts. You never invent numbers, names or claims. "
    "You reply with a single JSON object and nothing else."
)

_META_PREFIX = re.compile(
    r"^(?:based on|according to|from|in)\s+(?:the\s+)?(?:provided|given|above|document|text|excerpts?|paper|study)[^,:]*[,:]\s*",
    re.IGNORECASE,
)
_META_SUBJECT = re.compile(
    r"^(?:the|this)\s+(?:document|text|excerpt|paper|study|authors?)\s+"
    r"(?:states|mentions|indicates|suggests|notes|describes|discusses|presents|shows|reports|explains|highlights)\s+(?:that\s+)?",
    re.IGNORECASE,
)


def _tidy_bullet(text: str, max_words: int) -> str:
    text = re.sub(r"\*+", "", str(text)).strip()
    text = re.sub(r"^(?:(?:[-•]+|\d{1,2}[.)])\s+)+", "", text)      # list markers only - never real leading numbers ("80/20 split")
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)
    text = _META_PREFIX.sub("", text)
    text = _META_SUBJECT.sub("", text).strip()
    sentences = split_sentences(text)
    text = sentences[0] if sentences else text          # one idea per bullet
    words = text.split()
    if len(words) > max_words:                          # cut at a clause boundary if possible
        cut = " ".join(words[:max_words])
        text = (cut.rsplit(",", 1)[0] if "," in cut[len(cut) // 2:] else cut).rstrip(",;:") + "…"
    return (text[:1].upper() + text[1:]) if text else text


def analyze_section(spec: SectionSpec, index: DocumentIndex, llm: LLMClient,
                    settings: Settings = DEFAULT_SETTINGS) -> SectionResult:
    context = index.retrieve(spec.query, settings.retrieval_top_k, spec.anchor)
    prompt = f"""Excerpts from a scientific paper:
\"\"\"
{context}
\"\"\"

Write the "{spec.title}" slide about: {spec.focus}.

Return JSON with exactly these keys:
  "bullets": 3 to {settings.max_bullets_per_slide} slide bullets. Each is ONE self-contained sentence of at most
             {settings.max_bullet_words - 8} words, starting with the key fact. Keep specific numbers, model names and
             technical terms exactly as written in the excerpts.
  "notes":   a 60-100 word narrative paragraph for the speaker notes that elaborates the bullets.
If the excerpts say nothing relevant, return {{"bullets": [], "notes": ""}}.
Do not mention "the text", "the excerpts" or "the document". Do not include citations or markdown."""
    raw = llm.chat(settings.text_model, prompt, system=SYSTEM_PROMPT, json_mode=True)
    data = parse_json(raw)
    seen: set[str] = set()
    bullets: list[str] = []
    for b in data.get("bullets", []) if isinstance(data, dict) else []:
        tidy = _tidy_bullet(b, settings.max_bullet_words)
        if len(tidy.split()) >= 4 and tidy.lower() not in seen:
            seen.add(tidy.lower())
            bullets.append(tidy)
    notes = re.sub(r"\s+", " ", str(data.get("notes", ""))).strip() if isinstance(data, dict) else ""
    return SectionResult(spec.key, spec.title, bullets[: settings.max_bullets_per_slide], notes, context)


def generate_tagline(index: DocumentIndex, llm: LLMClient, settings: Settings = DEFAULT_SETTINGS) -> str:
    context = index.retrieve("abstract summary objective contribution", 3, anchor="start", char_budget=2500)
    prompt = (f"Paper excerpts:\n\"\"\"\n{context}\n\"\"\"\n\n"
              'Return JSON {"tagline": "..."} where tagline is ONE sentence of at most 18 words saying what this paper does.')
    try:
        tagline = str(parse_json(llm.chat(settings.text_model, prompt, system=SYSTEM_PROMPT, json_mode=True)).get("tagline", ""))
    except (ValueError, AttributeError):
        return ""
    return re.sub(r"\s+", " ", tagline).strip().strip('"')
