"""Underlines follow translated glyphs; graphic rules retain source pixels."""
from dataclasses import replace

import pymupdf as fitz

from slidetwin.decorations import source_decorations
from slidetwin.extract import native_lines
from slidetwin.models import Region


def source_page(pdf, text='A source heading', background=(1, 1, 1)):
    page = pdf.new_page(width=400, height=240)
    page.draw_rect(page.rect, color=None, fill=background)
    page.insert_text((30, 60), text, fontsize=20)
    lines = native_lines(page)
    chars = lines[0]['chars']
    box = fitz.Rect(chars[0]['bbox'])
    for char in chars[1:]:
        box |= fitz.Rect(char['bbox'])
    region = Region('p0001_r0000', 1, text, list(box), size=20)
    return page, region, lines


def underline(page, chars, y=62.8, fill=False):
    left = chars[0]['bbox'][0]
    right = chars[-1]['bbox'][2]
    if fill:
        page.draw_rect(fitz.Rect(left, y - .5, right, y + .5), color=None, fill=(0, 0, 0))
    else:
        page.draw_line((left, y), (right, y), color=(0, 0, 0), width=1)


def test_full_source_underline_becomes_reflowable_style_on_measured_nonwhite_background():
    with fitz.open() as pdf:
        page, region, lines = source_page(pdf, background=(.12, .25, .4))
        underline(page, lines[0]['chars'])
        styles, plan = source_decorations(page, region, lines)
        assert styles == [] and plan.underline
        assert len(plan.erasures) == 1 and plan.warnings == []
        background = plan.erasures[0]['background']
        assert all(abs(got - expected) < .01 for got, expected in zip(background, (.12, .25, .4)))
        # After source text is removed, covering the old underline with its
        # measured background removes its original fixed width completely.
        box = fitz.Rect(plan.erasures[0]['bbox'])
        page.draw_rect(box, color=None, fill=background)
        pix = page.get_pixmap(clip=box + (1, .3, -1, -.3), matrix=fitz.Matrix(2, 2), alpha=False)
        assert pix.samples[:3] != b'\xff\xff\xff'


def test_full_multiline_source_underline_is_not_left_at_the_old_wrap_locations():
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=240)
        page.insert_text((30, 60), 'First source line', fontsize=20)
        page.insert_text((30, 90), 'Second source line', fontsize=20)
        lines = native_lines(page)
        for line, y in zip(lines, [62.8, 92.8]):
            underline(page, line['chars'], y=y, fill=True)
        region = Region('r', 1, 'First source line\nSecond source line', [30, 30, 230, 98], size=20)
        _, plan = source_decorations(page, region, lines)
        assert plan.underline and len(plan.erasures) == 2
        # The renderer receives a text style plus two old geometric erasures.
        # Its translated line count and width are free to change independently.
        assert all(e['kind'] == 'source_underline' for e in plan.erasures)


def test_partial_underlines_append_style_targets_only_during_extraction():
    with fitz.open() as pdf:
        page, region, lines = source_page(pdf, text='Before ALPHA after')
        chars = lines[0]['chars']
        start = region.source.index('ALPHA')
        selected = chars[start:start + 5]
        underline(page, selected)
        region.inline_styles = [{'source': 'Before', 'color': 0, 'bold': True}]
        render_styles, old = source_decorations(page, region, lines)
        assert render_styles == region.inline_styles and old.erasures == []
        assert old.warnings[0]['kind'] == 'underline_mapping_unavailable'
        extracted, plan = source_decorations(page, region, lines, create_styles=True)
        assert extracted[0] == region.inline_styles[0]
        assert extracted[1]['source'] == 'ALPHA' and extracted[1]['underline']
        assert plan.erasures[0]['style_index'] == 1 and plan.erasures[0]['source'] == 'ALPHA'
        # A subsequent render can reference _s1 without shifting _s0.
        mapped = replace(region, inline_styles=extracted)
        _, second = source_decorations(page, mapped, lines)
        assert len(second.erasures) == 1 and second.warnings == []


def test_chart_rule_and_connected_axis_are_never_erased_as_text_decoration():
    with fitz.open() as pdf:
        page, region, lines = source_page(pdf)
        chars = lines[0]['chars']
        # A remote horizontal rule is not on the measured glyph baseline.
        page.draw_line((30, 150), (300, 150), width=1)
        _, plan = source_decorations(page, region, lines)
        assert not plan.underline and plan.erasures == []
        # Even a line matching the glyph width is retained if it connects to
        # another vector stroke: there is insufficient underline evidence.
        underline(page, chars)
        page.draw_line((chars[0]['bbox'][0], 62.8), (chars[0]['bbox'][0], 150), width=1)
        _, plan = source_decorations(page, region, lines)
        assert not plan.underline and plan.erasures == []
        assert plan.warnings[0]['kind'] == 'underline_background_unsafe'


def test_nonuniform_background_is_preserved_instead_of_white_painted():
    with fitz.open() as pdf:
        page, region, lines = source_page(pdf)
        page.draw_rect(fitz.Rect(70, 63, 110, 80), color=None, fill=(.4, .7, .5))
        underline(page, lines[0]['chars'])
        _, plan = source_decorations(page, region, lines)
        assert plan.erasures == [] and not plan.underline
        assert plan.warnings[0]['kind'] == 'underline_background_unsafe'
