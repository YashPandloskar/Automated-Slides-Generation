from slidegen.config import Settings
from slidegen.models import ParsedDocument
from slidegen.text_rag import (BM25, SECTION_SPECS, DocumentIndex, _tidy_bullet, analyze_section, chunk_text,
                               split_sentences)


def test_sentence_splitter_keeps_abbreviations_and_decimals_intact():
    text = "The workflow is shown in Fig. 1, as in Smith et al. It reached 0.95 accuracy. Next we test."
    assert split_sentences(text) == [
        "The workflow is shown in Fig. 1, as in Smith et al. It reached 0.95 accuracy.",
        "Next we test.",
    ]


def test_chunks_respect_sentence_boundaries_and_overlap():
    text = " ".join(f"Sentence number {i} is about models." for i in range(80))
    chunks = chunk_text(text, max_chars=300, overlap_chars=80)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.endswith(".") for c in chunks)                       # never cut mid-sentence
    assert chunks[0].split(". ")[-1] in chunks[1]                     # last sentence is carried over as overlap


def test_bm25_ranks_relevant_chunk_first():
    bm = BM25(["random forest accuracy", "limitations and challenges of the dataset", "future work directions"])
    assert bm.top("limitations challenges", 1) == [1]


def _doc(text):
    return ParsedDocument("x.pdf", "abc123" * 10, "T", 1, text)


def test_index_falls_back_to_bm25_when_embeddings_unavailable(tmp_path, fake_llm):
    doc = _doc(" ".join(f"Sentence {i} about topic {i % 5} and models." for i in range(120)))
    index = DocumentIndex(tmp_path / "chroma", fake_llm)
    n = index.build(doc)
    assert n > 3 and index.backend == "bm25"
    assert index.retrieve("topic 3 models", 3)


def test_index_dense_backend_and_document_order(tmp_path):
    from tests.conftest import FakeLLM
    doc = _doc(" ".join(f"Sentence {i} about topic {i % 5} and models." for i in range(120)))
    index = DocumentIndex(tmp_path / "chroma", FakeLLM(embed_ok=True))
    index.build(doc)
    assert index.backend == "chroma"
    ctx = index.retrieve("anything", 4, anchor="start")
    ids = [index.chunks.index(c) for c in ctx.split("\n\n")]
    assert ids == sorted(ids) and 0 in ids                            # document order, start anchor honoured


def test_tidy_bullet_strips_meta_phrases_and_markup():
    assert _tidy_bullet("**The document states that** XGBoost reached 92% accuracy [3].", 28).startswith("XGBoost reached 92%")
    assert _tidy_bullet("- Based on the provided text, the model is a stacking classifier.", 28).startswith("The model is")
    long = " ".join(["word"] * 60)
    assert len(_tidy_bullet(long, 20).split()) <= 21


def test_analyze_section_cleans_and_dedupes(tmp_path):
    from tests.conftest import FakeLLM
    llm = FakeLLM({"Methodology": {"bullets": ["The study shows that XGBoost was trained on 5000 samples.",
                                                "xgboost was trained on 5000 samples.", "ok", "Random Forest served as baseline."],
                                   "notes": "  A   narrative. "}})
    index = DocumentIndex(tmp_path / "c", llm)
    index.build(_doc(" ".join(f"Method sentence {i} with data." for i in range(60))))
    spec = next(s for s in SECTION_SPECS if s.key == "methods")
    res = analyze_section(spec, index, llm, Settings())
    assert res.bullets == ["XGBoost was trained on 5000 samples.", "Random Forest served as baseline."]
    assert res.notes == "A narrative."


def test_analyze_section_empty_when_model_finds_nothing(tmp_path):
    from tests.conftest import FakeLLM
    llm = FakeLLM({"Applications": {"bullets": [], "notes": ""}})
    index = DocumentIndex(tmp_path / "c", llm)
    index.build(_doc(" ".join(f"Some sentence {i}." for i in range(60))))
    res = analyze_section(next(s for s in SECTION_SPECS if s.key == "applications"), index, llm)
    assert res.bullets == []


def test_tidy_bullet_keeps_leading_numbers_but_strips_list_markers():
    assert _tidy_bullet("80/20 split was used for training and testing.", 28).startswith("80/20 split")
    assert _tidy_bullet("89 to 91% accuracy was reached by stacking.", 28).startswith("89 to 91%")
    assert _tidy_bullet("1. Data were split into folds.", 28) == "Data were split into folds."
    assert _tidy_bullet("- • Data were split into folds.", 28) == "Data were split into folds."
