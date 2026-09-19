import io

import pymupdf
import pytest
from PIL import Image

from slidegen.cli import main
from slidegen.parsing import clean_text, parse_document, strip_references


def _png(size, color):
    """'noise' gives a non-blank, unique picture; a colour gives a flat one."""
    img = Image.effect_noise(size, 60).convert("RGB") if color == "noise" else Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def sample_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Deep Learning for Widgets", fontsize=22)
    body = "This paper studies widgets with machine learning. " * 12
    page.insert_textbox(pymupdf.Rect(72, 90, 520, 330), body + "See [1] and https://example.com for more.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 340, 272, 490), stream=_png((300, 220), "noise"))
    page.insert_text((72, 505), "Fig. 1: Overview of the widget pipeline", fontsize=9)
    page.insert_image(pymupdf.Rect(300, 340, 320, 360), stream=_png((30, 30), "noise"))        # too small (logo)
    page.insert_image(pymupdf.Rect(340, 340, 440, 440), stream=_png((300, 300), (0, 0, 0)))    # blank black
    page2 = doc.new_page()
    page2.insert_textbox(pymupdf.Rect(72, 72, 520, 200), "Conclusion text " * 20, fontsize=10)
    page2.insert_text((72, 240), "References", fontsize=12)
    page2.insert_text((72, 260), "[1] Smith, J. A widget paper. 2020.", fontsize=9)
    path = tmp_path / "paper.pdf"
    doc.save(path)
    return path


def test_clean_text_rules():
    out = clean_text("A hyphen-\nated word [1, 2] see https://x.org/a and www.y.com end.\n12\nPage 3 of 9\n")
    assert "hyphenated" in out and "[1" not in out and "http" not in out and "www" not in out and "Page 3" not in out


def test_strip_references_only_cuts_the_back_half():
    text = "References are discussed early on.\n" + "body " * 200 + "\nReferences\n[1] cite"
    assert strip_references(text).rstrip().endswith("body")
    assert "References are discussed" in strip_references(text)


def test_parse_document_extracts_title_text_and_filters_images(sample_pdf, tmp_path):
    doc = parse_document(sample_pdf, tmp_path / "out")
    assert doc.title == "Deep Learning for Widgets" and doc.n_pages == 2
    assert "widgets with machine learning" in doc.text and "Smith" not in doc.text and "example.com" not in doc.text
    assert len(doc.images) == 1                                       # logo and black image dropped
    img = doc.images[0]
    assert (img.page, img.figure_no) == (1, 1) and img.caption.startswith("Fig. 1")


def test_identical_images_are_deduplicated(tmp_path):
    doc = pymupdf.open()
    png = _png((300, 220), "noise")
    for _ in range(2):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 200), "Some body text about things. " * 20, fontsize=10)
        page.insert_image(pymupdf.Rect(72, 300, 272, 450), stream=png)
    doc.save(tmp_path / "d.pdf")
    assert len(parse_document(tmp_path / "d.pdf", tmp_path / "o").images) == 1


def test_missing_and_textless_pdfs_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "nope.pdf", tmp_path)
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")
    with pytest.raises(ValueError, match="No extractable text"):
        parse_document(tmp_path / "blank.pdf", tmp_path)


def test_cli_reports_missing_input_with_exit_code_1(capsys, tmp_path):
    assert main(["run", str(tmp_path / "missing.pdf"), "-o", str(tmp_path / "o")]) == 1
    assert "not found" in capsys.readouterr().err
