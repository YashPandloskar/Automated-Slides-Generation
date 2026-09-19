import pytest

from slidegen.evaluation import lcs_length, rouge, tokenize, ungrounded_numbers


def test_identical_text_scores_one():
    r = rouge("the cat sat on the mat", "the cat sat on the mat")
    assert all(r[m]["f1"] == pytest.approx(1.0) for m in r)


def test_known_rouge1_values():
    r = rouge("the cat sat on the mat today", "the cat sat")     # 3 of 3 predicted words appear, 3 of 7 reference words
    assert r["rouge1"]["precision"] == pytest.approx(1.0)
    assert r["rouge1"]["recall"] == pytest.approx(3 / 7)
    assert r["rouge2"]["precision"] == pytest.approx(1.0)
    assert r["rougeL"]["recall"] == pytest.approx(3 / 7)


def test_disjoint_text_scores_zero_without_division_errors():
    assert rouge("alpha beta", "gamma delta")["rouge1"]["f1"] == 0.0
    assert rouge("", "")["rougeL"]["f1"] == 0.0


def test_lcs_and_tokenizer():
    assert lcs_length("a b c d".split(), "a c d".split()) == 3
    assert tokenize("XGBoost reached 92.5% accuracy.") == ["xgboost", "reached", "92.5", "accuracy"]


def test_ungrounded_numbers_flags_only_invented_figures():
    assert ungrounded_numbers("accuracy was 92.5 on 5000 samples", "reached 92.5 and 97 on 5000") == ["97"]
