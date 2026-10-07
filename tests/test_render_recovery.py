from pathlib import Path

import pymupdf as fitz

from slidetwin.config import Layout, Settings
from slidetwin.models import Document, Page, Region, digest, write_json
from slidetwin.render import build_plan, render, styled_html, math_font
from slidetwin.best_effort import retained_values, publish_best_effort
from slidetwin.qa import verify


def test_long_translation_uses_verified_empty_rows_and_preserves_neighbor(tmp_path):
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page(width=500,height=300)
        p.insert_text((30,60),'Gate oxide',fontsize=18)
        p.insert_text((300,60),'Neighbor',fontsize=18)
        p.insert_text((30,155),'Next paragraph',fontsize=18)
        from slidetwin.extract import enrich_page
        regions,_=enrich_page(p,1,[])
        pdf.save(source)
    doc=Document(digest(source.read_bytes()),[Page(1,500,300,regions)])
    values={r.id:('形成一层薄的高质量氧化物层，这是晶体管的栅极氧化物。' if 'Gate' in r.source else r.source) for r in regions}
    plans,failures=build_plan(source,doc,values,[1],Layout(),tmp_path)
    assert not failures
    assert len(plans)==1 and plans[0].frame[3]>plans[0].source_bbox[3]+10
    out=tmp_path/'out.pdf';render(source,out,plans,[1],Layout())
    assert verify(source,out,[1],plans,tmp_path)['passed']
    with fitz.open(out) as pdf:
        assert 'Neighbor' in pdf[1].get_text() and 'Next paragraph' in pdf[1].get_text()


def test_legacy_private_glyph_is_preserved_inside_translated_paragraph(tmp_path):
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page(width=500,height=250)
        p.insert_text((30,70),'Delay t',fontsize=18)
        font=p.get_fonts()[0][0];mapping=pdf.get_new_xref()
        pdf.update_object(mapping,'<<>>')
        pdf.update_stream(mapping,b'/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /Private def /CMapType 2 def 1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfchar <74> <F074> endbfchar endcmap CMapName currentdict /CMap defineresource pop end end')
        pdf.xref_set_key(font,'ToUnicode',f'{mapping} 0 R');pdf.save(source)
    with fitz.open(source) as pdf:
        p=pdf[0];chars=[c for b in p.get_text('rawdict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] for c in s['chars']]
        assert any(c['c']=='\uf074' for c in chars)
        box=list(p.get_text('blocks')[0][:4]);erase=[list(c['bbox']) for c in chars]
    r=Region('r',1,'Delay ⟦P000⟧',box,size=18,erase=erase,protected={'⟦P000⟧':'\uf074'})
    doc=Document(digest(source.read_bytes()),[Page(1,500,250,[r])])
    plans,failures=build_plan(source,doc,{'r':'延迟（⟦P000⟧）'},[1],Layout(),tmp_path)
    assert not failures and len(plans)==1 and plans[0].assets
    assert '\uf074' not in plans[0].extractable_text and '<img' in plans[0].html_body
    assert 'white-space:nowrap' in plans[0].html_body
    out=tmp_path/'out.pdf';render(source,out,plans,[1],Layout())
    assert verify(source,out,[1],plans,tmp_path)['passed']
    with fitz.open(out) as pdf:
        assert '延迟' in pdf[1].get_text() and pdf[1].get_image_info()


def test_exact_identifier_retains_source_color_without_style_model_result():
    region=Region('r',1,'Q1: discuss',[0,0,300,30],inline_styles=[
        {'source':'Q1','color':255,'bold':True,'italic':False}])
    html=styled_html(region,'Q1：讨论问题',{'r':'Q1：讨论问题'}, {})
    assert 'color:#0000ff' in html and '>Q1<' in html


def test_repeated_table_phrases_receive_distinct_source_line_breaks():
    style={'color':0,'bold':False,'italic':False,'break_after':True}
    r=Region('r',1,'One\nTwo\nThree',[0,0,200,70],inline_styles=[
        {**style,'source':'One'}, {**style,'source':'Two','break_before':True}, {**style,'source':'Three','break_before':True}])
    html=styled_html(r,'类别 类别 类别',{'r_s0':'类别','r_s1':'类别','r_s2':'类别'}, {})
    assert html.count('>类别</span>')==3
    assert '<br> <br>' not in html


def test_plain_math_literal_recovers_native_asset_style_without_semantic_rewrite():
    from slidetwin.render import original_asset_tokens
    r=Region('r',1,'Test ⟦P000⟧',[0,0,200,40],protected={'⟦P000⟧':'𝑣𝐺𝑆\uf0b3𝑣𝑡𝑜'},
             protected_assets={'⟦P000⟧':{'text':'𝑣𝐺𝑆\uf0b3𝑣𝑡𝑜'}})
    value='电压𝑣𝐺𝑆 ≥ 𝑣𝑡𝑜，重复𝑣𝐺𝑆≥𝑣𝑡𝑜'
    assert original_asset_tokens(r,value)=='电压⟦P000⟧，重复⟦P000⟧'
    r.protected_assets['⟦P000⟧']['text']='x²'
    assert original_asset_tokens(r,'x2')=='x2'
    assert original_asset_tokens(r,'⟦P000⟧')=='⟦P000⟧'


def test_inline_scripts_cannot_center_a_left_aligned_paragraph(tmp_path):
    from slidetwin.render import candidate_frame
    with fitz.open() as pdf:
        p=pdf.new_page(width=380,height=260)
        for baseline,text in [(70,'First long physical text row'),(86,'Second long physical text row'),(102,'Short last row')]:
            p.insert_text((28,baseline),text,fontsize=10)
        p.insert_text((120,81),'2',fontsize=6)
        chars=[c for b in p.get_text('rawdict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] for c in s['chars']]
        boxes=[list(c['bbox']) for c in chars];box=fitz.Rect(boxes[0])
        for b in boxes[1:]:box|=fitz.Rect(b)
        r=Region('r',1,'Paragraph',list(box),size=10,erase=boxes)
        assert candidate_frame(r,[r],p)[1]=='left'


def test_retained_candidate_cannot_bind_to_reused_target_id(tmp_path):
    cfg=Settings();r=Region('r',1,'New source',[0,0,200,40]);doc=Document('source',[Page(1,300,200,[r])])
    meta={'source_sha256':'source','config_fingerprint':cfg.fingerprint()}
    write_json(tmp_path/'translation-candidates.json',{**meta,'targets':{'r':{'source':'Old source','text':'错配文字'}}})
    write_json(tmp_path/'translation-ledger.json',{**meta,'translations':{'r':'旧缓存'},'target_sources':{'r':'Old source'}})
    assert retained_values(tmp_path,doc,cfg)=={}
    write_json(tmp_path/'translation-candidates.json',{**meta,'targets':{'r':{'source':'New source','text':'正确绑定'}}})
    assert retained_values(tmp_path,doc,cfg)=={'r':'正确绑定'}


def test_protected_token_numbers_are_not_sufficient_source_identity(tmp_path):
    cfg=Settings();r=Region('r',1,'Voltage ⟦P000⟧',[0,0,200,40],protected={'⟦P000⟧':'6'})
    doc=Document('source',[Page(1,300,200,[r])]);meta={'source_sha256':'source','config_fingerprint':cfg.fingerprint()}
    write_json(tmp_path/'translation-candidates.json',{**meta,'targets':{'r':{'source':r.source,'source_literal':'Voltage 5','text':'电压5'}}})
    assert retained_values(tmp_path,doc,cfg)=={}
    write_json(tmp_path/'translation-ledger.json',{**meta,'target_sources':[], 'target_source_literals':[], 'translations':{'r':'电压5'}})
    assert retained_values(tmp_path,doc,cfg)=={}


def test_math_font_covers_mathematical_italic_symbols():
    path=math_font()
    if path:
        font=fitz.Font(fontfile=str(path))
        assert all(font.has_glyph(ord(c)) for c in '𝑣𝐺𝑆𝑂𝑁')


def test_cached_glyph_metric_rounding_cannot_block_its_own_redaction(tmp_path):
    from test_best_effort import sample
    source,doc=sample(tmp_path)
    r=doc.pages[0].regions[0]
    r.erase=[[a,b-.011,c,d-.011] for a,b,c,d in r.erase]
    plans,failures=build_plan(source,doc,{'a':'电压 ⟦P000⟧'},[1],Layout(),tmp_path)
    assert not failures
    out=tmp_path/'out.pdf';render(source,out,plans,[1],Layout())
    with fitz.open(out) as pdf:
        assert '电压' in pdf[1].get_text() and 'Voltage' not in pdf[1].get_text()


def test_mixed_superscript_run_uses_explicit_font_and_roundtrips(tmp_path):
    from slidetwin.render import css_fonts,content_html,explicit_fallbacks,unicode_font
    font_cache=Path(__file__).resolve().parents[1]/'.local/fonts'
    regular=font_cache/'NotoSansSC-Regular-Safe.ttf';bold=font_cache/'NotoSansSC-Bold-Safe.ttf'
    if not regular.exists() or not unicode_font():
        import pytest
        pytest.skip('Full Docker font cache absent; mixed-font smoke test is run in build')
    primary=fitz.Font(fontfile=str(regular));fallback=fitz.Font(fontfile=str(unicode_font()))
    body=explicit_fallbacks('中文10⁶W',primary,[('unicode',fallback)])
    assert 'font-family:unicode' in body
    css,archive=css_fonts((regular,bold));out=tmp_path/'roundtrip.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_htmlbox(fitz.Rect(20,20,500,150),content_html('中文10⁶W',18,0,True,'left',1.12,body),css=css,archive=archive)
        pdf.subset_fonts();pdf.save(out)
    with fitz.open(out) as pdf:assert '中文10⁶W' in ''.join(pdf[0].get_text().split())


def test_optional_verification_failure_cannot_hide_published_translation(tmp_path,monkeypatch):
    from test_best_effort import sample
    import slidetwin.best_effort as module
    source,doc=sample(tmp_path)
    def fail(*args,**kwargs):raise OSError('diagnostic storage unavailable')
    monkeypatch.setattr(module,'verify',fail)
    out=tmp_path/'out.pdf'
    result=module.publish_best_effort(source,out,tmp_path/'work',doc,[1],Settings(),'test',{'a':'电压 ⟦P000⟧'},preview=False)
    assert result['final_output_published'] and result['diagnostic_warnings']
    with fitz.open(out) as pdf:assert '电压' in pdf[1].get_text()


def test_final_plan_write_failure_cannot_hide_published_translation(tmp_path,monkeypatch):
    from test_best_effort import sample
    import slidetwin.best_effort as module
    source,doc=sample(tmp_path);original=module.write_json
    def write(path,value):
        if path.name=='final-layout-plan.json':raise OSError('evidence unavailable')
        original(path,value)
    monkeypatch.setattr(module,'write_json',write)
    out=tmp_path/'out.pdf'
    result=module.publish_best_effort(source,out,tmp_path/'work',doc,[1],Settings(),'test',{'a':'电压 ⟦P000⟧'},preview=False)
    assert result['final_output_published'] and any(x['kind']=='final_plan_write_failed' for x in result['diagnostic_warnings'])
    with fitz.open(out) as pdf:assert '电压' in pdf[1].get_text()
