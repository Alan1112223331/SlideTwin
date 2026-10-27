"""Displayed slide coordinates must agree across native text, OCR and edits."""
import sys

import pymupdf as fitz
import pytest

from slidetwin.config import Layout
from slidetwin.extract import EXTRACT_VERSION, enrich_page, extract
from slidetwin.geometry import display_pdf
from slidetwin.models import Document, Page, digest, read_cache, write_json
from slidetwin.qa import verify
from slidetwin.render import build_plan, render


def rotated_source(path, angle, cropped=False):
    width, height = ((240, 400) if angle in (90, 270) else (400, 240))
    with fitz.open() as pdf:
        page = pdf.new_page(width=width + (60 if cropped else 0),
                            height=height + (80 if cropped else 0))
        if cropped:
            page.set_cropbox(fitz.Rect(20, 30, width+20, height+30))
        page.set_rotation(angle)
        matrix = page.derotation_matrix
        for point, text in [((30, 55), 'Visible heading'), ((30, 180), 'Lower visible label')]:
            page.insert_text(fitz.Point(point)*matrix, text, fontsize=16, rotate=angle)
        page.draw_rect(fitz.Rect(250, 95, 350, 150)*matrix, color=(1, 0, 0), fill=(.1, .6, .8))
        # A real bitmap beside the text must survive translation unchanged.
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 24, 18), False)
        pix.clear_with(90)
        page.insert_image(fitz.Rect(270, 170, 330, 215)*matrix, pixmap=pix, rotate=angle)
        pdf.save(path)


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('cropped', [False, True])
def test_rotated_native_slide_preserves_artwork_and_uses_display_coordinates(tmp_path, angle, cropped):
    source = tmp_path/'source.pdf'
    rotated_source(source, angle, cropped)
    before = source.read_bytes()
    with fitz.open(source) as original, display_pdf(source) as normalized:
        page = normalized[0]
        assert page.rotation == 0 and page.rect == fitz.Rect(0, 0, 400, 240)
        assert page.get_pixmap().samples == original[0].get_pixmap().samples
        regions, issues = enrich_page(page, 1, [])
        assert len(regions) == 2 and all(r.direction == [1, 0] for r in regions)
        assert all(page.rect.contains(fitz.Rect(r.bbox)) for r in regions)
        assert regions[1].bbox[1] > 150
        doc = Document(digest(before), [Page(1, 400, 240, regions)], issues)
    values = {regions[0].id: '可见标题', regions[1].id: '下方可见标签'}
    plans, failures = build_plan(source, doc, values, [1], Layout(), tmp_path)
    assert not failures and len(plans) == 2 and all(p.rotate == 0 for p in plans)
    output = tmp_path/'bilingual.pdf'
    render(source, output, plans, [1], Layout())
    qa = verify(source, output, [1], plans, tmp_path)
    assert qa['passed'], qa['failures']
    with fitz.open(output) as pdf, fitz.open(source) as original:
        assert len(pdf) == 2 and pdf[0].rotation == angle and pdf[1].rotation == 0
        assert pdf[0].get_pixmap().samples == original[0].get_pixmap().samples
        assert all(t in pdf[1].get_text() for t in values.values())
        assert 'Lower visible label' not in pdf[1].get_text()
    assert source.read_bytes() == before


def test_rotated_cache_refresh_aligns_docling_and_native_without_duplicate_ocr(tmp_path, monkeypatch):
    source = tmp_path/'source.pdf'
    rotated_source(source, 90)
    sha = digest(source.read_bytes())
    work = tmp_path/'work'
    work.mkdir()
    with display_pdf(source) as pdf:
        texts = [{'text': text, 'label': 'text', 'self_ref': f'#/texts/{index}',
                  'prov': [{'page_no': 1, 'bbox': {'l': box.x0, 't': box.y0,
                                                 'r': box.x1, 'b': box.y1, 'coord_origin': 'TOPLEFT'}}]}
                 for index, text in enumerate(['Visible heading', 'Lower visible label'])
                 for box in pdf[0].search_for(text)]
    raw = {'texts': texts, 'tables': []}
    write_json(work/'docling-document.json', raw)
    Document(sha, [Page(1, 400, 240)], schema_version=1).save(work/'document.json')
    write_json(work/'extraction-key.json', {
        'key': digest(sha+'34'+str([1])), 'source_sha256': sha, 'selected_pages': [1],
        'complete': True, 'docling_complete': True, 'extract_version': '34',
        'docling_sha256': digest((work/'docling-document.json').read_bytes()),
        'document_sha256': digest((work/'document.json').read_bytes()),
    })
    monkeypatch.setitem(sys.modules, 'docling.document_converter', None)
    from slidetwin import formula_ocr, raster_refine
    monkeypatch.setattr(formula_ocr, 'supplement', lambda *args: [])
    monkeypatch.setattr(raster_refine, 'refine', lambda page, descriptors, work: descriptors)
    document = extract(source, work)
    assert document.schema_version == 2
    assert len(document.pages[0].regions) == 2
    assert all(r.native and r.direction == [1, 0] for r in document.pages[0].regions)
    assert read_cache(work/'extraction-key.json')['extract_version'] == EXTRACT_VERSION
    assert read_cache(work/'docling-document.json') == raw
    # Raw Docling remains an extraction of the unmodified source.
    with fitz.open(work/'selected-source.pdf') as subset:
        assert subset[0].rotation == 90
    assert extract(source, work) == document


def test_legacy_rotated_coordinates_are_not_silently_filtered_as_off_canvas(tmp_path):
    source = tmp_path/'source.pdf'
    rotated_source(source, 90)
    document = Document(digest(source.read_bytes()), [Page(1, 400, 240)], schema_version=1)
    with pytest.raises(ValueError, match='rerun extraction'):
        build_plan(source, document, {}, [1], Layout(), tmp_path)


def test_rotated_ocr_patch_is_placed_in_the_same_displayed_frame(tmp_path):
    from slidetwin.models import Region
    source = tmp_path/'scan.pdf'
    with fitz.open() as text_pdf, fitz.open() as pdf:
        text = text_pdf.new_page(width=400, height=240)
        text.insert_text((30, 100), 'Scanned title', fontsize=20)
        box = text.search_for('Scanned title')[0]
        page = pdf.new_page(width=240, height=400)
        page.set_rotation(90)
        page.insert_image(page.rect*page.derotation_matrix, pixmap=text.get_pixmap(), rotate=90)
        pdf.save(source)
    region = Region('scan', 1, 'Scanned title', list(box), size=20, native=False)
    document = Document(digest(source.read_bytes()), [Page(1, 400, 240, [region], ocr_used=True)])
    plans, failures = build_plan(source, document, {'scan': '扫描标题'}, [1], Layout(), tmp_path)
    assert not failures and len(plans) == 1 and plans[0].rotate == 0
    output = tmp_path/'translated.pdf'
    render(source, output, plans, [1], Layout())
    assert verify(source, output, [1], plans, tmp_path)['passed']
    with fitz.open(output) as pdf:
        assert '扫描标题' in pdf[1].get_text(clip=fitz.Rect(plans[0].frame))


@pytest.mark.parametrize('angle', [90, 180, 270])
def test_rotation_with_offset_media_and_crop_preserves_visible_pixels(angle):
    from slidetwin.geometry import normalize_page_rotation
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=300)
        page.set_mediabox(fitz.Rect(-20, -30, 380, 270))
        page.set_cropbox(fitz.Rect(0, 20, 340, 260))
        page.insert_text((30, 50), 'Offset cropped content', fontsize=18)
        page.set_rotation(angle)
        before, bounds = page.get_pixmap(), page.rect
        normalize_page_rotation(page)
        assert page.rect == bounds and page.rotation == 0
        assert page.get_pixmap().samples == before.samples


def test_fallback_list_row_near_page_center_keeps_shared_left_indent():
    from slidetwin.models import Region
    from slidetwin.render import candidate_frame
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=240)
        page.draw_rect(page.rect, fill=(1, 1, 1))
        short = Region('short', 1, 'Short list row', [60, 40, 180, 58], size=18)
        wide = Region('wide', 1, 'Longer list row that happens to be centered', [60, 75, 340, 93], size=18)
        frame, align = candidate_frame(wide, [short, wide], page)
        assert align == 'left' and frame.x0 == 60


def test_quote_operator_after_relative_move_redacts_without_moving_other_text(tmp_path):
    # PyMuPDF 1.28.2 / upstream #5152 rewrites TL incorrectly after Td when
    # handling the single-quote show-text operator. Merely guarding raster
    # bounds cannot prevent the residual/moved text; the fixed engine is needed.
    source = tmp_path/'quoted-text.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=240)
        page.insert_font(fontname='helv')
        stream = pdf.get_new_xref()
        pdf.update_object(stream, '<<>>')
        pdf.update_stream(stream, b"BT /helv 16 Tf 20 190 Td 20 TL (Keep first line) Tj "
                                 b"0 -40 Td (Replace this line) ' ET")
        page.set_contents(stream)
        regions, issues = enrich_page(page, 1, [])
        assert len(regions) == 2
        original_box = tuple(page.search_for('Keep first line')[0])
        pdf.save(source)
    document = Document(digest(source.read_bytes()), [Page(1, 400, 240, regions)], issues)
    values = {r.id: ('替换这一行' if r.source == 'Replace this line' else r.source) for r in regions}
    plans, failures = build_plan(source, document, values, [1], Layout(), tmp_path)
    assert not failures and len(plans) == 1
    output = tmp_path/'result.pdf'
    render(source, output, plans, [1], Layout())
    with fitz.open(output) as pdf:
        assert 'Replace this line' not in pdf[1].get_text()
        assert '替换这一行' in pdf[1].get_text()
        assert tuple(pdf[1].search_for('Keep first line')[0]) == pytest.approx(original_box)
    result = verify(source, output, [1], plans, tmp_path)
    assert result['passed'], result['failures']


def test_optional_layout_import_cannot_change_native_glyph_metrics():
    from slidetwin.extract import native_lines
    from slidetwin.render import css_fonts
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((20, 60), 'Stable geometry', fontsize=16)
        fitz.TOOLS.unset_quad_corrections(True)
        fitz.TOOLS.set_small_glyph_heights(False)
        lines = native_lines(page)
        assert not fitz.TOOLS.unset_quad_corrections()
        assert fitz.TOOLS.set_small_glyph_heights()
        assert all(fitz.Rect(c['bbox']).height == pytest.approx(16, abs=.01)
                   for line in lines for c in line['chars'])
        fitz.TOOLS.unset_quad_corrections(True)
        css_fonts(Layout().fonts())
        assert not fitz.TOOLS.unset_quad_corrections()


@pytest.mark.parametrize('value', ['⟦P000⟧. 第五项内容', '5. 第五项内容', '5.第五项内容', '. 第五项内容', '第五项内容'])
def test_numbered_list_keeps_original_number_and_body_indent(tmp_path, value):
    from slidetwin.extract import native_lines
    from slidetwin.models import Region
    source = tmp_path/'numbered.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=240)
        page.insert_text((30, 60), '5.', fontsize=18)
        page.insert_text((80, 60), 'Fifth list entry', fontsize=18)
        chars = [c for line in native_lines(page) for c in line['chars']]
        boxes = [list(c['bbox']) for c in chars]
        box = fitz.Rect(boxes[0])
        for b in boxes[1:]:
            box.include_rect(b)
        region = Region('number', 1, '⟦P000⟧. Fifth list entry', list(box), size=18,
                        erase=boxes, protected={'⟦P000⟧': '5'})
        pdf.save(source)
    document = Document(digest(source.read_bytes()), [Page(1, 400, 240, [region])])
    unchanged, failures = build_plan(source, document, {'number': region.source}, [1], Layout(), tmp_path)
    assert not unchanged and not failures
    plans, failures = build_plan(source, document, {'number': value}, [1], Layout(), tmp_path)
    assert not failures and len(plans) == 1
    assert plans[0].frame[0] == pytest.approx(80, abs=.1)
    assert plans[0].text == '第五项内容'
    output = tmp_path/'result.pdf'
    render(source, output, plans, [1], Layout())
    with fitz.open(output) as pdf:
        assert pdf[1].get_text().count('5.') == 1
        assert pdf[1].search_for('5.')[0].x0 == pytest.approx(30, abs=.1)
    assert verify(source, output, [1], plans, tmp_path)['passed']
