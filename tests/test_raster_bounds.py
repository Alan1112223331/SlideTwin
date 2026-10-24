"""Off-canvas PDF objects must never reach native image code as empty arrays."""
from types import SimpleNamespace

import cv2
import pymupdf as fitz
import pytest

from slidetwin.config import Layout
from slidetwin.models import Document, Page, Region, digest
from slidetwin.render import (
    LayoutError, build_plan, raster_background, raster_label_patch,
    raster_typography, render, visible_native_ink,
)
from slidetwin.raster import raster_container
from slidetwin.raster_refine import caption_body_box, recognize
from slidetwin.formula_ocr import supplement


@pytest.fixture(autouse=True)
def reject_empty_native_inputs(monkeypatch):
    # Fail in Python on the old bug rather than risk terminating pytest in C++.
    for name in ('connectedComponentsWithStats', 'morphologyEx', 'cvtColor', 'findContours'):
        original = getattr(cv2, name)

        def checked(image, *args, _operation=original, **kwargs):
            assert image.size and all(image.shape[:2]), 'Empty image passed to OpenCV'
            return _operation(image, *args, **kwargs)

        monkeypatch.setattr(cv2, name, checked)


@pytest.mark.parametrize('bbox', [
    [50, 816, 85.25, 994],  # Reported PPT export: 141 x 0 pixels at scale 4.
    [50, -100, 85, -10],
    [-100, 30, -10, 80],
    [913, 30, 950, 80],
    [50, 684, 85, 700],   # Touching a page edge is not a visible intersection.
    [912, 30, 950, 80],
    [50, 30, 50, 80],
    [50, 30, 85, 30],
    [85, 30, 50, 80],
    [50, 80, 85, 30],
    [50, float('nan'), 85, 80],
    [50, 30, float('inf'), 80],
    None,
    [],
    [50, 30, 85],
    [50, 'invalid', 85, 80],
])
def test_invisible_native_regions_never_reach_opencv(bbox):
    with fitz.open() as pdf:
        page = pdf.new_page(width=912, height=684)
        region = Region('p0002_r0011', 2, 'Off-canvas source text', bbox)
        assert not visible_native_ink(page, region)


@pytest.mark.parametrize('width,height,samples', [(141, 0, b''), (0, 141, b''), (2, 2, b'')])
def test_empty_raster_after_valid_clip_is_not_analyzed(width, height, samples, tmp_path):
    pix = SimpleNamespace(width=width, height=height, n=3, samples=samples)
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 100), get_pixmap=lambda **kwargs: pix)
    region = Region('empty', 1, 'Visible geometry, empty raster', [10, 10, 80, 30])
    assert not visible_native_ink(page, region)
    assert caption_body_box(page, fitz.Rect(region.bbox)) is None
    assert recognize(page, fitz.Rect(region.bbox), tmp_path) == []
    with pytest.raises(LayoutError):
        raster_typography(page, region)
    with pytest.raises(LayoutError):
        raster_background(page, fitz.Rect(region.bbox))
    with pytest.raises(LayoutError):
        raster_label_patch(page, region, tmp_path)


def test_partially_off_canvas_text_remains_visible_and_clip_is_bounded(monkeypatch):
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=100)
        page.insert_text((2, 30), 'Visible text', fontsize=12)
        region = Region('edge', 1, 'Visible text', [-40, 12, 90, 34], size=12)
        original = page.get_pixmap
        clips = []

        def capture(**kwargs):
            clips.append(kwargs['clip'])
            assert page.rect.contains(kwargs['clip'])
            return original(**kwargs)

        monkeypatch.setattr(page, 'get_pixmap', capture)
        assert visible_native_ink(page, region)
        assert clips and clips[0] == fitz.Rect(0, 12, 90, 34)
        assert region.bbox == [-40, 12, 90, 34]


@pytest.mark.parametrize('bbox', [[50, 816, 85.25, 994], [50, 684, 85, 700], [50, 30, 50, 80]])
def test_invisible_ocr_regions_skip_analysis_even_when_padding_touches_page(bbox, tmp_path, monkeypatch):
    with fitz.open() as pdf:
        page = pdf.new_page(width=912, height=684)

        def forbidden(**kwargs):
            pytest.fail('An invisible region must not rasterize neighboring artwork')

        monkeypatch.setattr(page, 'get_pixmap', forbidden)
        box = fitz.Rect(bbox)
        region = Region('ghost', 1, 'FIG ghost', bbox, native=False)
        assert caption_body_box(page, box) is None
        assert recognize(page, box, tmp_path) == []
        assert supplement(page, [{'role': 'formula', 'bbox': box}], tmp_path) == []
        assert raster_container(page, box) is None
        with pytest.raises(LayoutError):
            raster_typography(page, region)
        with pytest.raises(LayoutError):
            raster_background(page, box)
        with pytest.raises(LayoutError):
            raster_label_patch(page, region, tmp_path)


def test_cached_off_canvas_native_and_ocr_regions_do_not_abort_export(tmp_path):
    source = tmp_path / 'source.pdf'
    pages = []
    with fitz.open() as pdf:
        for number in (1, 2):
            page = pdf.new_page(width=912, height=684)
            page.insert_text((40, 80), 'Visible title', fontsize=18)
            box = list(page.search_for('Visible title')[0])
            chars = [c for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                     for span in line['spans'] for c in span['chars']]
            regions = [Region(f'p{number}_visible', number, 'Visible title', box, size=18,
                              erase=[list(c['bbox']) for c in chars])]
            if number == 2:
                page.insert_text((50, 850), 'Ghost native', fontsize=12)
                regions.extend([
                    Region('p0002_r0011', 2, 'Ghost native', [50, 816, 85.25, 994]),
                    Region('p0002_r0012', 2, 'Ghost OCR', [50, 816, 85.25, 994], native=False),
                ])
            pages.append(Page(number, 912, 684, regions))
        pdf.save(source)
    document = Document(digest(source.read_bytes()), pages)
    # Deliberately omit ghost translations: invisible objects are not required
    # page content and must not become missing-translation failures either.
    plans, failures = build_plan(source, document, {'p1_visible': '可见标题', 'p2_visible': '可见标题'},
                                 [1, 2], Layout(), tmp_path)
    assert not failures and {p.id for p in plans} == {'p1_visible', 'p2_visible'}
    output = tmp_path / 'bilingual.pdf'
    render(source, output, plans, [1, 2], Layout())
    with fitz.open(output) as pdf:
        assert len(pdf) == 4
        for index in (1, 3):
            assert '可见标题' in pdf[index].get_text()
            assert pdf[index].rect == fitz.Rect(0, 0, 912, 684)
