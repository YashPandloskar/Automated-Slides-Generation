"""Plain data containers passed between pipeline stages (all JSON-serialisable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractedImage:
    path: str                      # original image as extracted from the PDF
    page: int                      # 1-based page number
    index: int                     # 1-based index on that page
    width: int
    height: int
    caption: str = ""              # "Fig. 3. ..." text found next to the image, if any
    figure_no: int | None = None   # parsed from the caption
    processed_path: str = ""       # denoised + enhanced copy used for analysis

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class ImageAnalysis:
    """Result of the LLaVA "Data Extraction" stage for one image."""
    image: str                          # file name
    is_graph: bool = False
    kind: str = "other"                 # graph | flowchart | diagram | table | photo | other
    graph_type: str = ""                # bar | line | scatter | pie | heatmap | confusion_matrix | ...
    title: str = ""
    x_axis: str = ""
    y_axis: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)   # [{"series": str, "points": [{"label","value"}]}]
    description: str = ""               # one-sentence description
    objects: list[str] = field(default_factory=list)           # object detection
    text_in_image: str = ""             # OCR
    confidence: str = "unverified"       # consistent | conflict | unverified  (vs. the PDF caption)
    error: str = ""


@dataclass
class ParsedDocument:
    source: str
    sha256: str
    title: str
    n_pages: int
    text: str                           # cleaned body text (references removed)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class SectionResult:
    key: str
    title: str
    bullets: list[str]
    notes: str                          # narrative paragraph -> speaker notes
    retrieved: str = ""                 # RAG context the LLM saw (used for evaluation)


@dataclass
class Figure:
    path: str
    caption: str
    kind: str
    summary: str = ""
    data_line: str = ""                 # short human-readable line built from extracted graph data
    width: int = 0
    height: int = 0


@dataclass
class SlideSection:
    key: str
    title: str
    bullets: list[str]
    notes: str = ""
    figures: list[Figure] = field(default_factory=list)


@dataclass
class SlideContext:
    """The "Extracted Context For Slide Generation" artefact (context.json)."""
    title: str
    subtitle: str
    source: str
    sections: list[SlideSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlideContext":
        sections = [
            SlideSection(
                key=s["key"], title=s["title"], bullets=list(s["bullets"]), notes=s.get("notes", ""),
                figures=[Figure(**f) for f in s.get("figures", [])],
            )
            for s in d["sections"]
        ]
        return cls(title=d["title"], subtitle=d.get("subtitle", ""), source=d.get("source", ""), sections=sections)
