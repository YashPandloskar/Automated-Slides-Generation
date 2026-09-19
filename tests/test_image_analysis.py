import json

from slidegen.image_analysis import _clean_series, analyze_image, analyze_images, check_against_caption
from slidegen.models import ExtractedImage, ImageAnalysis
from tests.conftest import FakeLLM


def _img(make_image, name="a.png"):
    return ExtractedImage(make_image(name), 1, 1, 400, 300)


def test_graph_branch_runs_detection_type_data_objects_ocr(make_image):
    llm = FakeLLM({
        "Does this image plot": {"is_graph": True},
        "This image is a graph": {"graph_type": "Bar", "title": "Accuracy by model", "x_axis": "Model", "y_axis": "Accuracy"},
        "Read the data values": {"series": [{"name": "acc", "points": [{"label": "RF", "value": "0.91"}, {"label": "XGB", "value": 0.93}]}]},
        "distinct objects": {"objects": ["bars", "axis"]},
        "Transcribe": {"text": "RF XGB"},
    })
    a = analyze_image(_img(make_image), llm)
    assert (a.is_graph, a.kind, a.graph_type, a.title) == (True, "graph", "bar", "Accuracy by model")
    assert a.data == [{"series": "acc", "points": [{"label": "RF", "value": 0.91}, {"label": "XGB", "value": 0.93}]}]
    assert a.objects == ["bars", "axis"] and a.text_in_image == "RF XGB"
    assert not any("What kind of image" in c for c in llm.calls)     # non-graph path not taken


def test_non_graph_branch_does_general_analysis_not_graph_steps(make_image):
    llm = FakeLLM({
        "Does this image plot": {"is_graph": False},
        "What kind of image": {"kind": "flowchart", "description": "A workflow of five steps."},
        "distinct objects": {"objects": ["box", "arrow"]},
        "Transcribe": {"text": ""},
    })
    a = analyze_image(_img(make_image), llm)
    assert (a.is_graph, a.kind, a.description) == (False, "flowchart", "A workflow of five steps.")
    assert not any("Read the data values" in c for c in llm.calls)


def test_garbage_model_output_degrades_gracefully(make_image):
    a = analyze_image(_img(make_image), FakeLLM({"Does this image plot": "not json at all"}))
    assert a.kind == "other" and not a.error


def test_clean_series_drops_non_numeric_and_short_series():
    raw = [{"name": "s", "points": [{"label": "a", "value": "12%"}, {"label": "b", "value": "n/a"}, {"label": "c", "value": 3}]},
           {"name": "one", "points": [{"label": "x", "value": 1}]}]
    assert _clean_series(raw) == [{"series": "s", "points": [{"label": "a", "value": 12.0}, {"label": "c", "value": 3.0}]}]


def test_analysis_cache_avoids_repeat_llm_calls(make_image, tmp_path):
    llm = FakeLLM({"Does this image plot": {"is_graph": False}, "What kind of image": {"kind": "photo", "description": "d"}})
    images = [_img(make_image)]
    cache = tmp_path / "cache.json"
    analyze_images(images, llm, cache, progress=lambda *_: None)
    n = len(llm.calls)
    again = analyze_images(images, llm, cache, progress=lambda *_: None)
    assert len(llm.calls) == n and again["a.png"].kind == "photo"
    assert json.loads(cache.read_text())


def test_constant_series_is_rejected_as_fabricated():
    raw = [{"name": "s", "points": [{"label": str(i), "value": 100} for i in range(5)]}]
    assert _clean_series(raw) == []


def test_caption_cross_check_flags_hallucinated_descriptions(make_image):
    img = ExtractedImage(make_image("c.png"), 8, 1, 400, 300, "Fig. 3: Comparison of Effectiveness of Different Models", 3)
    hallucinated = ImageAnalysis("c.png", title="Market Share of Competitors", description="Line chart of market share")
    faithful = ImageAnalysis("c.png", title="Model Comparison: Accuracy", description="Bar chart comparing models")
    assert check_against_caption(img, hallucinated) == "conflict"
    assert check_against_caption(img, faithful) == "consistent"
    assert check_against_caption(ExtractedImage(make_image("d.png"), 1, 1, 4, 3), faithful) == "unverified"
