from pathlib import Path

import pymupdf as fitz
import pytest

from slidetwin.config import Layout
from slidetwin.extract import enrich_page, protect, rect, protect_native_math, repeated_brand_marks
from slidetwin.models import Document, Page, digest
from slidetwin.pipeline import parse_pages
from slidetwin.qa import verify, compact, render_previews
from slidetwin.render import build_plan, candidate_frame, render, raster_background, LayoutError


def test_left_aligned_multiline_heading_stays_left_even_when_overall_box_is_centered(tmp_path):
    from slidetwin.models import Region
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=180)
        region=Region('heading',1,'Long heading\nShort line',[50,30,250,80],role='section_header',size=20,
                      erase=[[50,30,250,50],[50,60,140,80]])
        assert candidate_frame(region,[region],page)[1]=='left'


def test_emphasized_math_asset_is_styled_without_requiring_plain_text_in_html():
    from slidetwin.models import Region
    from slidetwin.render import styled_html
    r=Region('r',1,'Voltage ⟦P000⟧',[0,0,200,40],protected={'⟦P000⟧':'VGS'},
             inline_styles=[{'source':'VGS','color':255,'bold':False,'italic':True}])
    result=styled_html(r,'电压⟦P000⟧',{'r_s0':'VGS'},{'⟦P000⟧':'<img src="original.png">'})
    assert 'original.png' in result and 'font-style:italic' in result
    assert '⟦P000⟧' not in result


def test_symbol_font_math_mu_is_not_treated_as_a_wingdings_bullet():
    from slidetwin.extract import native_bullet
    assert native_bullet({'c':'\uf06d','font':'Wingdings-Regular'})
    assert not native_bullet({'c':'\uf06d','font':'SymbolMT'})


def test_superscript_ordinal_does_not_turn_following_prose_into_a_math_image():
    from slidetwin.extract import native_lines
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'1',fontsize=20);p.insert_text((32,43),'st',fontsize=12)
        p.insert_text((42,50),' Commercially Available',fontsize=20)
        chars=[c for line in native_lines(p) for c in line['chars']]
        text,protected,assets=protect_native_math(chars,'1st Commercially Available')
        assert 'Commercially Available' in text
        assert '1' in protected.values()
        assert not assets


def test_translucent_bitmap_text_shadow_is_removed_without_changing_original_page(tmp_path):
    from PIL import Image,ImageDraw,ImageFilter,ImageFont
    import io
    source=tmp_path/'shadow.pdf'
    mask=Image.new('L',(280,90));draw=ImageDraw.Draw(mask)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32)
    draw.text((8,12),'Shadow title',font=font,fill=110)
    mask=mask.filter(ImageFilter.GaussianBlur(3))
    bitmap=Image.new('RGBA',mask.size,(0,0,0,0));bitmap.putalpha(mask)
    stream=io.BytesIO();bitmap.save(stream,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=240,height=140)
        p.insert_image(fitz.Rect(35,29,175,74),stream=stream.getvalue())
        p.insert_text((40,52),'Shadow title',fontsize=16)
        pdf.save(source)
        regions,issues=enrich_page(p,1,[])
        document=Document(digest(source.read_bytes()),[Page(1,240,140,regions)])
    plans,failures=build_plan(source,document,{regions[0].id:'阴影标题'},[1],Layout(),tmp_path)
    assert not failures
    assert len(plans[0].shadow_boxes)==1
    output=tmp_path/'translated.pdf'
    render(source,output,plans,[1],Layout())
    assert verify(source,output,[1],plans,tmp_path)['passed']
    with fitz.open(output) as pdf:
        assert any(x[1] for x in pdf[0].get_images())
        # Only a transparent placeholder, rather than the original shadow, remains.
        assert not any(x[2]>1 and x[3]>1 and x[1] for x in pdf[1].get_images())


@pytest.fixture
def source(tmp_path):
    path = tmp_path/"source.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=640, height=360)
        page.draw_rect(fitz.Rect(0, 0, 640, 360), color=None, fill=(0.06, 0.12, 0.2))
        page.insert_text((32, 48), "Compute architecture", fontsize=22, color=(1, 1, 1))
        page.draw_circle((39, 99), 2.5, color=None, fill=(0, 0.8, 0.8))
        page.insert_text((52, 104), "First level explanation", fontsize=15, color=(1, 1, 1))
        page.draw_circle((65, 131), 2, color=None, fill=(0, 0.8, 0.8))
        page.insert_text((78, 136), "Nested explanation", fontsize=15, color=(1, 1, 1))
        page.draw_rect(fitz.Rect(32, 182, 304, 256), color=(0.1, 0.8, 0.9), width=2)
        page.draw_line((32, 220), (304, 220), color=(0.1, 0.8, 0.9), width=1)
        page.insert_text((44, 208), "Accumulator", fontsize=15, color=(1, 1, 1))
        page.insert_text((44, 246), "Multiplier", fontsize=15, color=(1, 1, 1))
        page.insert_text((410, 210), "y = x + 2", fontsize=18, color=(1, 1, 1))
        doc.save(path)
    return path


def test_bilingual_render_preserves_bullets_nested_indents_formulas_and_graphics(source, tmp_path):
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open(source) as pdf:
        regions, issues = enrich_page(pdf[0], 1, [])
        assert not issues
        assert len(regions) == 5
        first = next(r for r in regions if "First level" in r.source)
        nested = next(r for r in regions if "Nested" in r.source)
        assert first.bbox[0] == pytest.approx(52)
        assert nested.bbox[0] == pytest.approx(78)
        assert candidate_frame(nested, regions, pdf[0])[0].x0 == pytest.approx(78)
    translations = dict(zip([r.id for r in regions], ["计算架构", "第一级说明", "嵌套说明", "累加器", "乘法器"]))
    doc = Document(digest(source.read_bytes()), [Page(1, 640, 360, regions)])
    plans, failures = build_plan(source, doc, translations, [1], Layout(), tmp_path)
    assert not failures
    output = tmp_path/"output.pdf"
    render(source, output, plans, [1], Layout())
    report = verify(source, output, [1], plans, tmp_path)
    assert report["passed"], report
    with fitz.open(output) as pdf:
        assert len(pdf) == 2
        assert "y = x + 2" in pdf[1].get_text()


def test_numbered_badge_remains_original_and_body_keeps_its_indent(tmp_path):
    source = tmp_path/"numbered-list.pdf"
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        page = pdf.new_page(width=320, height=200)
        page.draw_circle((30, 47), 7, fill=(0.1, 0.2, 0.8))
        page.insert_text((28, 50), "1", fontsize=8, color=(1, 1, 1))
        page.insert_text((45, 51), "Learning Pipeline", fontsize=12)
        pdf.save(source)
        regions, issues = enrich_page(page, 1, [{"role": "list_item", "ref": "list", "text": "1 Learning Pipeline", "bbox": fitz.Rect(20, 36, 210, 60)}])
    assert not issues
    assert len(regions) == 1
    assert regions[0].source == "Learning Pipeline"
    assert regions[0].bbox[0] == pytest.approx(45)
    assert all(b[0] >= 45 for b in regions[0].erase)
    doc = Document(digest(source.read_bytes()), [Page(1, 320, 200, regions)])
    plans, failures = build_plan(source, doc, {regions[0].id: "学习流程"}, [1], Layout(), tmp_path)
    assert not failures
    output = tmp_path/"numbered-output.pdf"
    render(source, output, plans, [1], Layout())
    assert verify(source, output, [1], plans, tmp_path)["passed"]
    with fitz.open(output) as pdf:
        badge = fitz.Rect(22, 39, 38, 55)
        assert pdf[0].get_pixmap(clip=badge).samples == pdf[1].get_pixmap(clip=badge).samples


def test_only_repeated_institution_margin_artwork_is_preserved():
    with fitz.open() as image_doc:
        page = image_doc.new_page(width=80, height=20)
        page.insert_text((2, 13), "UNIVERSITY", fontsize=8)
        image = page.get_pixmap().tobytes("png")
    with fitz.open() as pdf:
        descriptors = {}
        for number in range(1, 4):
            page = pdf.new_page(width=320, height=200)
            page.insert_image(fitz.Rect(10, 175, 90, 195), stream=image)
            page.insert_image(fitz.Rect(100, 70, 180, 90), stream=image)
            descriptors[number] = [
                {"text": "UNIVERSITY", "bbox": fitz.Rect(12, 180, 88, 190), "ref": "logo", "role": "text"},
                {"text": "Diagram", "bbox": fitz.Rect(102, 75, 178, 85), "ref": "body", "role": "text"}]
        marks = repeated_brand_marks(pdf, descriptors, {1: 1, 2: 2, 3: 3})
        assert len(marks[1]) == 1
        regions, issues = enrich_page(pdf[0], 1, descriptors[1], marks[1])
        assert [r.source for r in regions] == ["Diagram"]
        assert issues[0]["kind"] == "preserved_brand_artwork"
        # A single occurrence must remain eligible for OCR translation.
        with fitz.open() as single:
            single.insert_pdf(pdf, from_page=0, to_page=0)
            assert not repeated_brand_marks(single, {1: descriptors[1]}, {1: 1})


@pytest.mark.parametrize("text,is_symbol", [("Ids (mA)", True), ("Vgs (V)", True), ("Time (s)", False), ("Current (mA)", False)])
def test_rotated_symbolic_axes_stay_original_but_english_axes_still_need_translation(text, is_symbol):
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=240)
        page.insert_text((100, 200), text, fontsize=14, rotate=90)
        box = fitz.Rect(page.get_text("blocks")[0][:4])
        regions, issues = enrich_page(page, 1, [{"role": "text", "ref": "axis", "text": text, "bbox": box}])
        if is_symbol:
            assert not regions and not issues
        else:
            assert regions and any(i["kind"] == "rotated_text" and i["blocking"] for i in issues)


def test_overlong_translation_fails_readability_gate_without_erasing_source(source, tmp_path):
    initial = source.read_bytes()
    with fitz.open(source) as pdf:
        regions, _ = enrich_page(pdf[0], 1, [])
    doc = Document(digest(initial), [Page(1, 640, 360, regions)])
    plans, failures = build_plan(source, doc, {r.id: "非常长的译文"*400 for r in regions}, [1], Layout(), tmp_path)
    assert failures and not plans
    assert source.read_bytes() == initial


def test_numbers_and_identifiers_are_programmatically_protected():
    text, protected = protect("Use 128 lanes at 1.5 GHz; x₁ and V_DD remain exact")
    assert "128" in protected.values() and "1.5" in protected.values()
    assert "x₁" in protected.values() and "V_DD" in protected.values()
    assert "⟦P000⟧" in text


def test_bottom_left_coordinates():
    assert rect({"l": 10, "t": 200, "r": 40, "b": 150, "coord_origin": "BOTTOMLEFT"}, 400) == fitz.Rect(10, 200, 40, 250)


@pytest.mark.parametrize("pages", ["0", "1,1", "3-1", "7", "1-2-3"])
def test_invalid_page_selection(pages):
    with pytest.raises(ValueError):
        parse_pages(pages, 6)


def test_explicit_page_order_retained():
    assert parse_pages("4,1-2", 6) == [4, 1, 2]


def test_inline_math_protection_uses_positions_not_ambiguous_string_replacement():
    text = "earlier r then rₜ"
    chars = [{"c": ch, "font": "CambriaMath" if i == len(text)-1 else "Arial", "size": 12,
              "bbox": [i*6, 10, i*6+6, 22], "origin": [i*6, 20]} for i, ch in enumerate(text)]
    protected_text, values, assets = protect_native_math(chars, text)
    assert protected_text == "earlier r then ⟦P000⟧"
    assert values == {"⟦P000⟧": "rₜ"}
    assert assets["⟦P000⟧"]["bbox"][0] == text.rfind("r")*6


def test_native_superscript_preserved_even_with_ordinary_font():
    text = "Voltage VDD"
    chars = [{"c": ch, "font": "Arial", "size": 8 if i >= 9 else 12,
              "bbox": [i*6, 13 if i >= 9 else 10, i*6+6, 25 if i >= 9 else 22],
              "origin": [i*6, 24 if i >= 9 else 20]} for i, ch in enumerate(text)]
    protected_text, values, assets = protect_native_math(chars, text)
    assert protected_text == "Voltage ⟦P000⟧"
    assert values["⟦P000⟧"] == "VDD"


def test_author_name_is_preserved_as_original_glyphs():
    text = "Course | © Dr. Alice Example | AI Models"
    chars = [{"c": ch, "font": "Arial", "size": 12, "flags": 0, "color": 0,
              "bbox": [i*6, 10, i*6+6, 22], "origin": [i*6, 20]} for i, ch in enumerate(text)]
    source, protected, assets = protect_native_math(chars, text)
    assert source == "Course | ⟦P000⟧ AI Models"
    assert protected["⟦P000⟧"] == "© Dr. Alice Example |"
    assert "⟦P000⟧" in assets


def test_text_layer_hyphen_shaping_is_equivalent_but_numbers_are_not():
    assert compact("向量‐矩阵") == compact("向量-矩阵")
    assert compact("V2") != compact("V3")


def test_complex_ocr_background_is_blocked_instead_of_painted_over():
    with fitz.open() as doc:
        page = doc.new_page(width=100, height=100)
        for x in range(100):
            page.draw_rect(fitz.Rect(x, 0, x+1, 100), color=None, fill=(x/100, 0.2, 1-x/100))
        with pytest.raises(LayoutError, match="non-uniform"):
            raster_background(page, fitz.Rect(10, 20, 90, 60))


def test_uniform_ocr_background_can_be_safely_restored():
    with fitz.open() as doc:
        page = doc.new_page(width=100, height=100)
        page.draw_rect(page.rect, color=None, fill=(0.8, 0.9, 1))
        page.insert_text((15, 40), "Label", fontsize=9)
        bg = raster_background(page, fitz.Rect(12, 28, 45, 43))
        assert bg[0] == pytest.approx(0.8, abs=0.02)
        assert bg[2] == pytest.approx(1, abs=0.02)


def test_visibility_distinguishes_small_white_type_from_covered_answers():
    from slidetwin.render import visible_native_ink
    from slidetwin.models import Region
    with fitz.open() as doc:
        p=doc.new_page(width=300,height=200)
        p.draw_rect(fitz.Rect(0,0,300,20),fill=(0,0,0),color=None)
        p.insert_text((20,10),'Data Types',fontsize=6,color=(1,1,1))
        box=p.search_for('Data Types')[0]
        assert visible_native_ink(p,Region('nav',1,'Data Types',list(box),size=6,color=0xffffff))
        p.insert_text((20,100),'Hidden answer',fontsize=14)
        box=p.search_for('Hidden answer')[0]
        p.draw_rect(box+(-1,-1,1,1),fill=(1,1,1),color=None)
        assert not visible_native_ink(p,Region('answer',1,'Hidden answer',list(box),size=14,color=0))


def test_hollow_ocr_bullet_keeps_indent_but_letters_do_not_become_bullets():
    from slidetwin.render import raster_typography
    from slidetwin.models import Region
    with fitz.open() as doc:
        p=doc.new_page(width=300,height=200)
        p.draw_rect(fitz.Rect(21,44,25,48),color=(0,.4,.2),width=.8)
        p.insert_text((36,49),'An example',fontsize=10,color=(0,0,.6))
        r=Region('r',1,'An example',[19,38,100,51],native=False,size=10)
        measured,_,bullet=raster_typography(p,r)
        assert bullet and measured.bbox[0]>=35
        p.insert_text((20,100),'Categorical',fontsize=10,color=(0,0,.6))
        box=p.search_for('Categorical')[0]
        r=Region('r2',1,'Categorical ■',list(box),native=False,size=10)
        measured,_,bullet=raster_typography(p,r)
        assert not bullet and measured.bbox==list(box)


def test_math_axis_units_and_formula_fragments_are_not_english_prose():
    from slidetwin.extract import immutable_math_label
    for source in ['Ids (µA)','Vgs−Vt','C metal','C gate','softmax(QK','Ggd']:
        assert immutable_math_label(source)
    for source in ['Metal gate','Gate capacitance','softmax function']:
        assert not immutable_math_label(source)


def test_native_table_cell_fragments_are_one_complete_translation_target():
    with fitz.open() as pdf:
        p=pdf.new_page(width=350,height=160)
        p.draw_rect(fitz.Rect(15,20,320,65),color=(.5,.7,.9),fill=(1,1,1))
        p.insert_text((20,36),'The model must fit in storage fast',fontsize=12)
        p.insert_text((20,52),'enough.',fontsize=12)
        descriptors=[{'ref':'left','role':'text','bbox':fitz.Rect(18,23,77,56),'text':'The model enough.'},
                     {'ref':'right','role':'text','bbox':fitz.Rect(77,23,318,40),'text':'must fit in storage fast'}]
        regions,_=enrich_page(p,1,descriptors)
        assert len(regions)==1
        assert regions[0].source=='The model must fit in storage fast enough.'


def test_full_width_sentence_and_indented_continuation_are_translated_together():
    first='Q = charge in the space-charge layer when the arbitrary reverse bias from source to bulk'
    second='voltage VSB is zero;'
    size=14
    width=(25+fitz.get_text_length(first,fontsize=size))/.94
    with fitz.open() as pdf:
        p=pdf.new_page(width=width,height=150)
        p.insert_text((25,46),first,fontsize=size)
        p.insert_text((70,60),second,fontsize=size)
        boxes=[p.search_for(value)[0] for value in (first,second)]
        descriptors=[{'ref':str(i),'role':'text','bbox':box,'text':value}
                     for i,(box,value) in enumerate(zip(boxes,(first,second)))]
        regions,_=enrich_page(p,1,descriptors)
        assert len(regions)==1
        from slidetwin.extract import restore
        assert restore(regions[0],regions[0].source)==first+' '+second


def test_cell_merge_does_not_absorb_untranslated_diagram_symbols():
    with fitz.open() as pdf:
        p=pdf.new_page(width=350,height=180)
        p.draw_rect(fitz.Rect(15,20,320,65),color=(0,0,0),fill=(1,1,.8))
        p.insert_text((25,36),'n+',fontsize=12)
        p.insert_text((200,54),'Bulk Si',fontsize=12)
        descriptors=[{'ref':'bulk','role':'text','bbox':fitz.Rect(198,40,225,58),'text':'Bulk'},
                     {'ref':'si','role':'text','bbox':fitz.Rect(225,40,280,58),'text':'Si'}]
        regions,_=enrich_page(p,1,descriptors)
        assert len(regions)==1 and regions[0].source=='Bulk Si'


def test_inline_emphasis_does_not_turn_the_entire_paragraph_blue(tmp_path):
    source = tmp_path/"mixed.pdf"
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=200)
        page.insert_text((30, 50), "Normalization", fontsize=12, color=(0, 0, 1))
        page.insert_text((112, 50), ": adjust data and prevent anomalies in analysis.", fontsize=12, color=(0, 0, 0))
        doc.save(source)
    with fitz.open(source) as doc:
        regions, issues = enrich_page(doc[0], 1, [{"ref": "#/texts/0", "role": "text", "bbox": fitz.Rect(29, 35, 450, 55), "text": "Normalization: adjust data and prevent anomalies in analysis."}])
    assert len(regions) == 1 and not issues
    region = regions[0]
    assert region.color == 0
    assert region.inline_styles[0]["source"] == "Normalization"
    translations = {region.id: "归一化：调整数据并防止分析受到异常值影响。", region.id+"_s0": "归一化"}
    document = Document(digest(source.read_bytes()), [Page(1, 600, 200, regions)])
    placements, failures = build_plan(source, document, translations, [1], Layout(), tmp_path)
    assert not failures
    output = tmp_path/"mixed-translated.pdf"
    render(source, output, placements, [1], Layout())
    with fitz.open(output) as doc:
        spans = [s for b in doc[1].get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    assert any("归一化" in s["text"] and s["color"] == 255 for s in spans)
    assert any("调整数据" in s["text"] and s["color"] == 0 for s in spans)


def test_preview_rebuild_does_not_mix_stale_numbering_or_delete_unrelated_files(tmp_path, monkeypatch):
    import slidetwin.qa as qa
    monkeypatch.setattr(qa.shutil, "which", lambda name: None)
    output = tmp_path/"output.pdf"
    with fitz.open() as doc:
        doc.new_page(width=100, height=100)
        doc.new_page(width=100, height=100)
        doc.save(output)
    folder = tmp_path/"preview"
    folder.mkdir()
    (folder/"page-001.png").write_bytes(b"stale")
    (folder/"pair-0099.png").write_bytes(b"stale")
    (folder/"page-notes.png").write_bytes(b"keep")
    result = render_previews(output, tmp_path, [1], dpi=30)
    assert result["rendered_pages"] == 2
    assert not (folder/"page-001.png").exists()
    assert not (folder/"pair-0099.png").exists()
    assert (folder/"page-notes.png").read_bytes() == b"keep"
