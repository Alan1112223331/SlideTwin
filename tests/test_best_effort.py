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
        assert '待检查译稿' in pdf.metadata['title']
    assert report['fully_validated'] is False
    assert Path(report['previous_output_backup']).read_bytes()==b'previous-output'
    assert source.read_bytes()==before


def test_failed_layout_still_exports_model_text_in_readable_page(tmp_path,monkeypatch):
    import slidetwin.best_effort as module
    source,doc=sample(tmp_path);cfg=Settings();work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(module,'build_plan',lambda *a,**k:([],[{'page':1,'id':'a','kind':'layout_blocked'}]))
    out=tmp_path/'out.pdf'
    report=publish_best_effort(source,out,work,doc,[1],cfg,'layout failure',{'a':'电压为 6'},preview=False)
    assert report['pages'][0]['layout']=='local_overflow_notes'
    with fitz.open(out) as pdf:
        assert '6' in pdf[1].get_text() and 'Voltage 5' not in pdf[1].get_text()


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
