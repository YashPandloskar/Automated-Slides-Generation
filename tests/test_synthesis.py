from slidegen.models import ExtractedImage, ImageAnalysis, SectionResult
from slidegen.synthesis import assign_figures, clean_caption, figure_data_line, synthesize


def _img(name, fig_no=None, caption="", w=800, h=500):
    return ExtractedImage(f"/x/{name}", 1, 1, w, h, caption, fig_no)


def _sec(key, bullets, notes="", retrieved=""):
    return SectionResult(key, key.title(), bullets, notes, retrieved)


def test_explicit_figure_mention_wins():
    images = [_img("a.png", 1, "Fig. 1: Flowchart of the system"), _img("b.png", 2, "Fig. 2: Something")]
    analyses = {"a.png": ImageAnalysis("a.png", kind="flowchart"), "b.png": ImageAnalysis("b.png", kind="flowchart")}
    secs = [_sec("methods", ["We train a model."], retrieved="The workflow is shown in Fig. 1."),
            _sec("results", ["Accuracy was high."])]
    out = assign_figures(images, analyses, secs)
    assert out["methods"][0].path == "/x/a.png"          # the explicitly mentioned figure comes first


def test_graph_goes_to_results_and_each_image_used_once():
    images = [_img("g.png", 3, "Fig. 3: Comparison of accuracy of models")]
    analyses = {"g.png": ImageAnalysis("g.png", is_graph=True, kind="graph", graph_type="bar", title="accuracy of models")}
    secs = [_sec("methods", ["Models were trained."]), _sec("results", ["Model accuracy compared."]),
            _sec("stats", ["Accuracy 93%."])]
    out = assign_figures(images, analyses, secs)
    assigned = [k for k, v in out.items() if v]
    assert len(assigned) == 1 and assigned[0] in ("results", "stats")


def test_irrelevant_image_is_not_placed():
    images = [_img("logo.png", None, "")]
    analyses = {"logo.png": ImageAnalysis("logo.png", kind="other", description="a company logo")}
    assert assign_figures(images, analyses, [_sec("results", ["Accuracy was high."])])["results"] == []


def test_failed_analyses_are_ignored_and_empty_sections_dropped():
    images = [_img("a.png", 1, "Fig. 1: x")]
    analyses = {"a.png": ImageAnalysis("a.png", kind="graph", error="boom")}
    ctx = synthesize("T", "tag", "s.pdf", [_sec("results", ["A finding here.", "Another finding."]), _sec("limits", [])], images, analyses)
    assert [s.key for s in ctx.sections] == ["results"] and ctx.sections[0].figures == []


def test_data_line_and_caption_helpers():
    a = ImageAnalysis("a", data=[{"series": "acc", "points": [{"label": "RF", "value": 0.91},
                                                              {"label": "XGB", "value": 0.93}]}],
                      description="desc")
    assert figure_data_line(a) == "acc: RF 0.91 | XGB 0.93"
    assert clean_caption("", a) == "desc"
    assert clean_caption("word " * 100, a).endswith("…")


def test_conflicting_llava_output_is_ignored_and_caption_wins():
    images = [_img("g.png", 3, "Fig. 3: Comparison of accuracy of models")]
    bad = ImageAnalysis("g.png", is_graph=True, kind="graph", title="Market Share", description="Bar chart of Market Share",
                        data=[{"series": "m", "points": [{"label": "1", "value": 5}, {"label": "2", "value": 9}]}],
                        confidence="conflict")
    fig = assign_figures(images, {"g.png": bad}, [_sec("results", ["Model accuracy compared."])], show_chart_values=True)["results"][0]
    assert fig.caption.startswith("Fig. 3: Comparison") and fig.summary == "" and fig.data_line == ""


def test_chart_values_are_opt_in_and_require_consistency():
    images = [_img("g.png", 3, "Fig. 3: Comparison of accuracy of models")]
    good = ImageAnalysis("g.png", is_graph=True, kind="graph", title="accuracy of models", confidence="consistent",
                         data=[{"series": "acc", "points": [{"label": "RF", "value": 0.91}, {"label": "XGB", "value": 0.93}]}])
    secs = [_sec("results", ["Model accuracy compared."])]
    assert assign_figures(images, {"g.png": good}, secs)["results"][0].data_line == ""
    assert assign_figures(images, {"g.png": good}, secs, show_chart_values=True)["results"][0].data_line == "acc: RF 0.91 | XGB 0.93"


def test_figures_are_not_placed_by_word_overlap_alone():
    images = [_img("cm.png", 6, "Fig. 6: Confusion matrix for random forest")]
    analyses = {"cm.png": ImageAnalysis("cm.png", is_graph=True, kind="graph")}
    limits = _sec("limits", ["Random forest confusion matrix shows limits."])       # shares words, but no affinity/mention
    assert assign_figures(images, analyses, [limits])["limits"] == []


def test_methods_and_results_can_hold_two_figures():
    images = [_img(f"f{i}.png", i, f"Fig. {i}: flowchart {i}") for i in (1, 2)]
    analyses = {im.name: ImageAnalysis(im.name, kind="flowchart") for im in images}
    sec = _sec("methods", ["We use a workflow."], retrieved="See Fig. 1 and Fig. 2.")
    assert len(assign_figures(images, analyses, [sec])["methods"]) == 2


def test_duplicate_bullets_across_sections_are_removed():
    from slidegen.synthesis import dedupe_sections
    a = _sec("intro", ["Sarcomas are rare tumours of connective tissue.", "Machine learning is proposed for classification."])
    b = _sec("topics", ["Sarcomas are rare tumours of connective tissue.", "The paper covers datasets and stacking."])
    c = _sec("results", ["XGBoost reached the highest accuracy.", "Stacking reached 91% accuracy."])
    out = dedupe_sections([a, b, c])
    assert [s.key for s in out] == ["intro", "results"]        # 'topics' kept only 1 novel bullet -> dropped


def test_caption_decides_kind_when_llava_cannot_classify_the_image():
    from slidegen.synthesis import effective_kind
    approach = _img("a.png", 1, "Fig. 1. A graphical representation of the approach")
    curve = _img("b.png", 2, "Fig. 2. Graph denoting forecasting before intervention")
    unknown = ImageAnalysis("x", kind="other")
    assert effective_kind(approach, unknown) == "diagram" and effective_kind(curve, unknown) == "graph"
    assert effective_kind(_img("c.png", None, ""), unknown) == "other"
    assert effective_kind(curve, ImageAnalysis("x", kind="flowchart")) == "flowchart"      # LLaVA's answer wins when it gave one
    sec = _sec("methods", ["We use an approach."], retrieved="See Fig. 1.")
    assert len(assign_figures([approach], {"a.png": unknown}, [sec])["methods"]) == 1
