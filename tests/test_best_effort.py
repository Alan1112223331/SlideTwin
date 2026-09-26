import json
from pathlib import Path

import pymupdf as fitz

from slidetwin.best_effort import publish_best_effort
from slidetwin.config import Settings,Provider
from slidetwin.models import Document,Page,Region,digest,write_json


def sample(tmp_path):
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page(width=400,height=250);p.insert_text((30,70),'Voltage 5',fontsize=14)
        box=list(p.search_for('Voltage 5')[0])
        glyphs=[list(c['bbox']) for b in p.get_text('rawdict')['blocks'] for line in b.get('lines',[]) for s in line['spans'] for c in s['chars']]
        pdf.save(source)
    document=Document(digest(source.read_bytes()),[Page(1,400,250,[Region('a',1,'Voltage ⟦P000⟧',box,size=14,erase=glyphs,protected={'⟦P000⟧':'5'})],'Voltage 5')])
    return source,document


def test_export_keeps_rejected_translation_and_replaces_output_with_backup(tmp_path):
    source,doc=sample(tmp_path);cfg=Settings();work=tmp_path/'work';work.mkdir()
    write_json(work/'translation-candidates.json',{'source_sha256':doc.source_sha256,'config_fingerprint':cfg.fingerprint(),
               'targets':{'a':{'text':'电压为 6'}}})
    output=tmp_path/'output.pdf';output.write_bytes(b'previous-output')
    before=source.read_bytes()
    report=publish_best_effort(source,output,work,doc,[1],cfg,'Number validation failed',preview=False)
    with fitz.open(output) as pdf:
        assert len(pdf)==2
        assert '6' in pdf[1].get_text() and 'Voltage 5' not in pdf[1].get_text()
        assert pdf[1].rect==fitz.Rect(0,0,400,250)
        assert pdf.metadata['title']=='source - SlideTwin'
    assert report['fully_validated'] is False
    assert Path(report['previous_output_backup']).read_bytes()==b'previous-output'
    assert source.read_bytes()==before


def test_failed_layout_keeps_slide_size_and_retains_model_text_in_sidecar(tmp_path,monkeypatch):
    import slidetwin.best_effort as module
    source,doc=sample(tmp_path);cfg=Settings();work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(module,'build_plan',lambda *a,**k:([],[{'page':1,'id':'a','kind':'layout_blocked'}]))
    out=tmp_path/'out.pdf'
    text='电压为 6。'+('这是模型已经返回、必须完整保留的内容。'*100)
    report=publish_best_effort(source,out,work,doc,[1],cfg,'layout failure',{'a':text},preview=False)
    assert report['pages'][0]['layout']=='source_layout'
    assert report['status']=='completed_with_warnings'
    assert report['pages'][0]['unplaced_translations']==[{'id':'a','bbox':doc.pages[0].regions[0].bbox,'translation':text}]
    assert json.loads(out.with_suffix('.issues.json').read_text(encoding='utf8'))==report
    with fitz.open(out) as pdf:
        assert len(pdf)==2
        assert pdf[0].rect==pdf[1].rect==fitz.Rect(0,0,400,250)
        assert 'Voltage 5' not in pdf[1].get_text()
        assert all(value not in pdf[1].get_text() for value in ('局部排版待检查','原位置','这是模型'))


def test_extraction_failure_never_adds_diagnostic_page_or_footer(tmp_path):
    source,doc=sample(tmp_path);doc.pages[0].regions=[]
    doc.diagnostics=[{'page':1,'kind':'page_enrichment_failed','blocking':True}]
    out=tmp_path/'out.pdf'
    report=publish_best_effort(source,out,tmp_path/'work',doc,[1],Settings(),'extraction failed',preview=False)
    assert report['pages'][0]['extraction_failed']
    with fitz.open(source) as original,fitz.open(out) as pdf:
        assert len(pdf)==2
        assert pdf[1].rect==original[0].rect
        assert pdf[1].get_pixmap().samples==original[0].get_pixmap().samples
        assert '提取未完成' not in pdf[1].get_text()


def test_unsafe_ocr_overflow_retains_text_only_in_report(tmp_path,monkeypatch):
    import slidetwin.best_effort as module
    source,doc=sample(tmp_path);doc.pages[0].regions[0].native=False
    monkeypatch.setattr(module,'build_plan',lambda *a,**k:([],[{'page':1,'id':'a','kind':'layout_blocked'}]))
    out=tmp_path/'out.pdf'
    report=publish_best_effort(source,out,tmp_path/'work',doc,[1],Settings(),'unsafe erase',{'a':'电压 ⟦P000⟧'},preview=False)
    assert report['pages'][0]['unplaced_translations'][0]['translation']=='电压 5'
    assert any(i['kind']=='source_pixels_retained' for i in report['pages'][0]['issues'])
    with fitz.open(source) as original,fitz.open(out) as pdf:
        assert pdf[1].rect==original[0].rect
        assert pdf[1].get_pixmap().samples==original[0].get_pixmap().samples


def test_pipeline_exports_after_retry_limit_without_hiding_unvalidated_text(tmp_path,monkeypatch):
    import slidetwin.pipeline as pipeline
    from slidetwin.protocol import ProtocolError
    source,doc=sample(tmp_path);cfg=Settings(provider=Provider(base_url='https://test.invalid/v1',model='test'))
    work=tmp_path/'work';out=tmp_path/'out.pdf'
    monkeypatch.setenv('SLIDETWIN_API_KEY','unit-test-only')
    monkeypatch.setattr(pipeline,'extract',lambda *a,**k:doc)
    class Failing:
        def __init__(self,*a,**k):pass
        async def run_async(self,*a,**k):
            write_json(work/'translation-candidates.json',{'source_sha256':doc.source_sha256,'config_fingerprint':cfg.fingerprint(),'targets':{'a':{'text':'电压为 6'}}})
            raise ProtocolError('Retries exhausted')
    monkeypatch.setattr(pipeline,'AsyncTranslator',Failing)
    report=pipeline.run(source,out,work,cfg,preview=False)
    assert report['status']=='completed_with_warnings' and out.exists()
    with fitz.open(out) as pdf:assert '6' in pdf[1].get_text()
