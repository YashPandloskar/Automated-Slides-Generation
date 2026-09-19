"""Stage 5 - Slide generation: SlideContext -> .pptx (16:9, consistent theme, figures with captions, speaker notes)."""
from __future__ import annotations

import math
import re
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from .models import Figure, SlideContext, SlideSection

# ---- theme -----------------------------------------------------------------------------------
NAVY = RGBColor(0x14, 0x21, 0x3D)
TEAL = RGBColor(0x1B, 0x9A, 0xAA)
INK = RGBColor(0x22, 0x2B, 0x3A)
MUTED = RGBColor(0x6B, 0x75, 0x85)
CARD = RGBColor(0xF3, 0xF5, 0xF8)
CARD_LINE = RGBColor(0xD9, 0xDE, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

SLIDE_W, SLIDE_H = 13.333, 7.5          # inches
MARGIN = 0.7
TITLE_TOP, BODY_TOP, BODY_BOTTOM = 0.45, 1.75, 6.75
WIDE_ASPECT = 1.3                        # figures at least this wide get their own full-width slide
FONT_SIZES = (26, 24, 22, 20, 18, 16, 14)


# ---- helpers ---------------------------------------------------------------------------------
def _run_font(run, size: float, color: RGBColor, bold: bool = False, italic: bool = False) -> None:
    run.font.name, run.font.size, run.font.bold, run.font.italic = FONT, Pt(size), bold, italic
    run.font.color.rgb = color


def _textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return box, tf


def _rect(slide, x, y, w, h, fill: RGBColor, line: RGBColor | None = None, shape=MSO_SHAPE.RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def _background(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _estimate_lines(text: str, width_in: float, size_pt: float) -> int:
    chars_per_line = max(1, int(width_in / (size_pt * 0.50 / 72)))   # Calibri averages ~0.5 em per glyph
    return max(1, math.ceil(len(text) / chars_per_line))


def fit_font_size(bullets: list[str], width_in: float, height_in: float) -> float | None:
    """Largest font size (from FONT_SIZES) at which all bullets fit the box, or None if even the smallest overflows."""
    for size in FONT_SIZES:
        text_w = width_in - 0.45                       # bullet indent
        lines = sum(_estimate_lines(b, text_w, size) for b in bullets)
        needed = lines * size * 1.2 / 72 + len(bullets) * (size * 0.55 / 72)
        if needed <= height_in:
            return size
    return None


def _add_bullets(slide, bullets: list[str], x: float, y: float, w: float, h: float) -> None:
    size = fit_font_size(bullets, w, h) or FONT_SIZES[-1]
    _, tf = _textbox(slide, x, y, w, h)
    indent = Inches(0.35)
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        p.space_after = Pt(size * 0.55)
        p.alignment = PP_ALIGN.LEFT
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(indent)))
        pPr.set("indent", str(-int(indent)))
        clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", str(TEAL))
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")   # must follow spacing elements in the schema
        run = p.add_run()
        run.text = text
        _run_font(run, size, INK)


def _chrome(slide, ctx: SlideContext, number: int, title: str) -> None:
    """Title, accent bar and footer shared by all content slides."""
    _background(slide, WHITE)
    _rect(slide, 0, 0, SLIDE_W, 0.16, NAVY)
    _, tf = _textbox(slide, MARGIN, TITLE_TOP, SLIDE_W - 2 * MARGIN, 0.85, MSO_ANCHOR.MIDDLE)
    run = tf.paragraphs[0].add_run()
    run.text = title
    _run_font(run, 34 if len(title) <= 38 else 28, NAVY, bold=True)
    _rect(slide, MARGIN, 1.38, 0.9, 0.06, TEAL)
    _, ftf = _textbox(slide, MARGIN, 7.0, 9.0, 0.3, MSO_ANCHOR.MIDDLE)
    r = ftf.paragraphs[0].add_run()
    r.text = ctx.title if len(ctx.title) <= 90 else ctx.title[:87] + "…"
    _run_font(r, 10, MUTED)
    _, ntf = _textbox(slide, SLIDE_W - MARGIN - 1.0, 7.0, 1.0, 0.3, MSO_ANCHOR.MIDDLE)
    ntf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    r = ntf.paragraphs[0].add_run()
    r.text = str(number)
    _run_font(r, 10, MUTED)


def _figure_number(fig: Figure) -> int:
    m = re.match(r"\s*fig(?:ure)?\.?\s*(\d+)", fig.caption, re.IGNORECASE)
    return int(m.group(1)) if m else 10_000


def _figure_size(fig: Figure) -> tuple[int, int]:
    if fig.width and fig.height:
        return fig.width, fig.height
    with Image.open(fig.path) as im:
        return im.size


def _add_figure(slide, fig: Figure, x: float, y: float, w: float, h: float) -> None:
    """Figure on a light card with caption underneath, scaled to fit without distortion."""
    caption = fig.caption
    data = f"Chart values (approx.): {fig.data_line}" if fig.data_line else ""
    cap_lines = _estimate_lines(caption, w - 0.3, 12) + (_estimate_lines(data, w - 0.3, 11) if data else 0)
    cap_h = 0.16 + cap_lines * 0.21 if (caption or data) else 0
    card_h = h - cap_h
    _rect(slide, x, y, w, card_h, CARD, CARD_LINE)
    pw, ph = _figure_size(fig)
    pad = 0.15
    scale = min((w - 2 * pad) / pw, (card_h - 2 * pad) / ph)
    iw, ih = pw * scale, ph * scale
    slide.shapes.add_picture(fig.path, Inches(x + (w - iw) / 2), Inches(y + (card_h - ih) / 2), Inches(iw), Inches(ih))
    if caption or data:
        _, tf = _textbox(slide, x, y + card_h + 0.08, w, cap_h - 0.08)
        if caption:
            r = tf.paragraphs[0].add_run()
            r.text = caption
            _run_font(r, 12, MUTED, italic=True)
        if data:
            p = tf.paragraphs[0] if not caption else tf.add_paragraph()
            r = p.add_run()
            r.text = data
            _run_font(r, 11, TEAL)


# ---- slide types -----------------------------------------------------------------------------
def _title_slide(prs: Presentation, ctx: SlideContext) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _background(slide, NAVY)
    _rect(slide, MARGIN, 2.05, 1.1, 0.08, TEAL)
    n = len(ctx.title)
    _, tf = _textbox(slide, MARGIN, 2.3, SLIDE_W - 2 * MARGIN - 1.0, 2.4)
    r = tf.paragraphs[0].add_run()
    r.text = ctx.title
    _run_font(r, 44 if n <= 45 else 38 if n <= 75 else 32 if n <= 110 else 28, WHITE, bold=True)
    if ctx.subtitle:
        _, stf = _textbox(slide, MARGIN, 4.85, SLIDE_W - 2 * MARGIN - 2.0, 1.0)
        r = stf.paragraphs[0].add_run()
        r.text = ctx.subtitle
        _run_font(r, 20, RGBColor(0xB8, 0xE3, 0xE9))
    _, ftf = _textbox(slide, MARGIN, 6.7, 9.0, 0.35)
    r = ftf.paragraphs[0].add_run()
    r.text = f"Automated summary of {ctx.source}"
    _run_font(r, 12, RGBColor(0x9A, 0xA5, 0xB8))


def _agenda_slide(prs: Presentation, ctx: SlideContext, number: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(slide, ctx, number, "Overview")
    per_col = 5
    for col, start in enumerate(range(0, len(ctx.sections), per_col)):
        _, tf = _textbox(slide, MARGIN + col * 5.9, BODY_TOP + 0.1, 5.5, 4.5)
        for i, sec in enumerate(ctx.sections[start:start + per_col]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(16)
            num = p.add_run()
            num.text = f"{start + i + 1:02d}   "
            _run_font(num, 24, TEAL, bold=True)
            r = p.add_run()
            r.text = sec.title
            _run_font(r, 24, INK)


def _notes(slide, text: str) -> None:
    if text:
        slide.notes_slide.notes_text_frame.text = text


def _content_slides(prs: Presentation, ctx: SlideContext, sec: SlideSection, number: int) -> int:
    """One section -> 1-3 slides. Returns the number of slides added."""
    added = 0
    sec = SlideSection(sec.key, sec.title, sec.bullets, sec.notes, sorted(sec.figures, key=_figure_number))
    fig = sec.figures[0] if sec.figures else None
    wide = bool(fig) and (lambda wh: wh[0] / wh[1] >= WIDE_ASPECT)(_figure_size(fig))
    side_by_side = bool(fig) and not wide
    text_w = 5.9 if side_by_side else SLIDE_W - 2 * MARGIN
    body_h = BODY_BOTTOM - BODY_TOP

    groups = [sec.bullets]
    if fit_font_size(sec.bullets, text_w, body_h) is None or (fit_font_size(sec.bullets, text_w, body_h) or 0) < 16:
        mid = math.ceil(len(sec.bullets) / 2)
        groups = [sec.bullets[:mid], sec.bullets[mid:]]

    for gi, bullets in enumerate(groups):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _chrome(slide, ctx, number + added, sec.title if gi == 0 else f"{sec.title} (cont.)")
        _add_bullets(slide, bullets, MARGIN, BODY_TOP, text_w, body_h)
        if side_by_side and gi == 0:
            fx = MARGIN + text_w + 0.4
            _add_figure(slide, fig, fx, BODY_TOP, SLIDE_W - MARGIN - fx, body_h)
        _notes(slide, sec.notes)
        added += 1
    if wide:                                                # dedicated full-width figure slide
        added += _figure_slide(prs, ctx, sec, fig, number + added)
    for extra in sec.figures[1:]:                           # further figures: one full slide each
        added += _figure_slide(prs, ctx, sec, extra, number + added)
    return added


def _figure_slide(prs: Presentation, ctx: SlideContext, sec: SlideSection, fig: Figure, number: int) -> int:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    m = re.match(r"\s*fig(?:ure)?\.?\s*\d+\s*[.:]\s*(.+)", fig.caption, re.IGNORECASE)
    title = m.group(1).strip() if m and len(m.group(1)) <= 60 else f"{sec.title}: Figure"
    _chrome(slide, ctx, number, title)
    _add_figure(slide, fig, MARGIN, BODY_TOP - 0.1, SLIDE_W - 2 * MARGIN, BODY_BOTTOM - BODY_TOP + 0.1)
    _notes(slide, fig.summary)
    return 1


def _closing_slide(prs: Presentation, ctx: SlideContext) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _background(slide, NAVY)
    _rect(slide, MARGIN, 2.9, 1.1, 0.08, TEAL)
    _, tf = _textbox(slide, MARGIN, 3.15, SLIDE_W - 2 * MARGIN, 1.2)
    r = tf.paragraphs[0].add_run()
    r.text = "Thank you"
    _run_font(r, 48, WHITE, bold=True)
    _, stf = _textbox(slide, MARGIN, 4.3, SLIDE_W - 2 * MARGIN, 0.6)
    r = stf.paragraphs[0].add_run()
    r.text = "Questions & discussion"
    _run_font(r, 22, RGBColor(0xB8, 0xE3, 0xE9))


def build_presentation(ctx: SlideContext, out_path: str | Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    _title_slide(prs, ctx)
    number = 2
    _agenda_slide(prs, ctx, number)
    number += 1
    for sec in ctx.sections:
        number += _content_slides(prs, ctx, sec, number)
    _closing_slide(prs, ctx)
    prs.core_properties.title = ctx.title
    prs.core_properties.subject = f"Automated summary of {ctx.source}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    return out_path
