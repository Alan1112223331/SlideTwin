"""Fault injection: local failures must not discard unrelated completed work."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pymupdf as fitz
import pytest

from slidetwin.async_client import AsyncModelClient
from slidetwin.async_translate import AsyncTranslator, ASYNC_VERSION
from slidetwin.best_effort import publish_best_effort, retained_values
from slidetwin.client import ProviderError
from slidetwin.config import Settings, Provider
from slidetwin.models import Document, Page, Region, digest, write_json, read_cache
from slidetwin.protocol import ProtocolError
from slidetwin.render import LayoutError


def translator(tmp_path,count=3):
    cfg=Settings();cfg.translation.glossary=False;cfg.provider.vision=False
    doc=Document('hash',[Page(n,600,400,[Region(f'p{n}',n,'Input memory',[20,60,300,100])],f'Context for page {n}') for n in range(1,count+1)])
    class Client:
        usage={}
        async def complete(self,messages,response_format=None,**kw):
            targets=json.loads(messages[1]['content'].split('TARGETS TO RETURN:\n')[1].split('\n')[0])
            return ''.join(f'<<<{key}>>>译文{key}<<<END>>>' for key in targets)
    return AsyncTranslator(cfg,Client(),doc,Path('unused'),tmp_path)


def test_oversized_page_does_not_prevent_later_pages(tmp_path,monkeypatch):
    t=translator(tmp_path)
    estimate=t.estimate_batch
    def with_oversize(numbers):
        value=estimate(numbers)
        if 2 in numbers:value['fits']=False
        return value
    monkeypatch.setattr(t,'estimate_batch',with_oversize)
    original=t.call
    async def call(page,*args,**kwargs):
        if page.number==2:raise ProtocolError('Request exceeds capacity')
        return await original(page,*args,**kwargs)
    monkeypatch.setattr(t,'call',call)
    with pytest.raises(ProtocolError):asyncio.run(t.run_async([1,2,3]))
    ledger=read_cache(tmp_path/'translation-ledger.json')
    assert ledger['completed_pages']==[1,3]
    assert [f['page'] for f in ledger['failures']]==[2]
    assert set(ledger['translations'])=={'p1','p3'}


def test_glossary_failure_does_not_cancel_translation(tmp_path,monkeypatch):
    t=translator(tmp_path)
    async def fail():raise ProviderError('terminology unavailable',503)
    monkeypatch.setattr(t,'glossary_async',fail)
    assert set(asyncio.run(t.run_async([1,2,3])))=={'p1','p2','p3'}
    assert read_cache(tmp_path/'translation-ledger.json')['preparation_warnings']


def test_global_glossary_configuration_error_does_not_fan_out_calls(tmp_path,monkeypatch):
    t=translator(tmp_path)
    async def fail():raise ProviderError('model not found',400)
    monkeypatch.setattr(t,'glossary_async',fail)
    with pytest.raises(ProviderError):asyncio.run(t.run_async([1,2,3]))
    assert not (tmp_path/'batch-plan.json').exists()


def test_partial_glossary_keeps_successful_chunks_and_retries_only_failed(tmp_path):
    t=translator(tmp_path);t.config.translation.glossary=True;t.config.provider.context_window_tokens=6000
    t.context=''.join(f'Complete course memory context line {n}\n' for n in range(1500))
    calls=[];fail=True
    async def complete(messages,**kw):
        part=kw['label']['context_part'];calls.append(part)
        if fail and part==2:raise ProviderError('part two failed',503)
        return f'guide {part}'
    t.client.complete=complete
    asyncio.run(t.glossary_async())
    saved=read_cache(t.cache/'glossary.json')
    assert not saved['complete'] and 'guide 1' in saved['text'] and 'guide 2' not in saved['text']
    first_calls=list(calls);fail=False;calls.clear()
    asyncio.run(t.glossary_async())
    assert len(first_calls)>2 and calls==[2]
    assert read_cache(t.cache/'glossary.json')['complete']


def test_corrupt_one_page_cache_preserves_other_page_checkpoints(tmp_path):
    t=translator(tmp_path)
    asyncio.run(t.run_async([1,2,3]))
    (t.cache/'page-0002.json').write_text('{broken')
    # With no network available, the matching grouped checkpoint can recover
    # page 2 while pages 1 and 3 remain cache hits.
    async def fail(*a,**k):raise AssertionError('Unrelated completed work was called again')
    t.client.complete=fail
    with pytest.warns(RuntimeWarning,match='invalid cache'):
        values=asyncio.run(t.run_async([1,2,3]))
    assert set(values)=={'p1','p2','p3'}


def test_corrupt_candidate_file_does_not_hide_valid_ledger(tmp_path):
    from test_best_effort import sample
    source,doc=sample(tmp_path);cfg=Settings()
    (tmp_path/'translation-candidates.json').write_text('broken')
    write_json(tmp_path/'translation-ledger.json',{'source_sha256':doc.source_sha256,'config_fingerprint':cfg.fingerprint(),'translations':{'a':'已完成内容'}})
    with pytest.warns(RuntimeWarning):assert retained_values(tmp_path,doc,cfg)=={'a':'已完成内容'}


def test_remaining_page_failure_does_not_mark_cached_siblings_failed(tmp_path):
    t=translator(tmp_path)
    asyncio.run(t.run_async([1,2,3]))
    # A changed page legitimately invalidates only its own cache.
    t.document.pages[1].regions[0].source='Updated target'
    async def fail(*a,**k):raise ProviderError('only pending page failed',400)
    t.client.complete=fail
    with pytest.raises(ProtocolError):asyncio.run(t.run_async([1,2,3]))
    ledger=read_cache(tmp_path/'translation-ledger.json')
    assert ledger['completed_pages']==[1,3]
    assert [f['page'] for f in ledger['failures']]==[2]


def test_improved_optional_glossary_does_not_invalidate_completed_pages(tmp_path):
    t=translator(tmp_path)
    t.glossary='partial terminology guide'
    asyncio.run(t.run_async([1,2,3]))
    t.glossary='completed terminology guide'
    async def fail(*a,**k):raise AssertionError('Completed pages were retranslated after optional preparation changed')
    t.client.complete=fail
    assert set(asyncio.run(t.run_async([1,2,3])))=={'p1','p2','p3'}


def test_plain_parent_survives_failed_style_child(tmp_path):
    t=translator(tmp_path,1);t.config.provider.protocol='plain'
    p=t.document.pages[0];p.regions[0].inline_styles=[{'source':'memory','bold':True}]
    async def plain(page,key,source,draft):
        if key.endswith('_s0'):raise ProviderError('child failed',503)
        return '主句已经完成'
    t.plain=plain
    with pytest.raises(ProviderError):asyncio.run(t.batch_async(p,t.sources(p),'plain'))
    assert t.batch_checkpoint(p,t.sources(p),'plain')=={'p1':'主句已经完成'}


def test_split_review_retains_completed_half(tmp_path):
    t=translator(tmp_path,2)
    class Counter:
        admission_capacity=100
        def request(self,messages,fmt):return 101 if len(fmt or {})>100 else 50
        def output(self,sources,ratio):return len(sources)*30
    # Force just the two-page draft over capacity.
    counter=Counter()
    counter.request=lambda messages,fmt:50
    t.counter=counter
    t.config.translation.context_mode='document'
    async def batch(page,sources,mode,draft):
        if page.number==2:raise ProviderError('second half failed',503)
        return {'p1':'复核完成'}
    t.batch_async=batch
    group=t.grouped_page([1,2]);sources=t.sources(group);draft={'p1':'初稿1','p2':'初稿2'}
    with pytest.raises(ProviderError):asyncio.run(t.review_async(group,sources,draft))
    assert t.batch_checkpoint(group,sources,t.config.provider.protocol,draft)=={'p1':'复核完成'}


@pytest.mark.parametrize('status',[400,413,422])
def test_rejected_group_retries_pages_with_changed_payload(tmp_path,status):
    t=translator(tmp_path)
    original=t.client.complete;calls=[]
    async def complete(messages,response_format=None,**kw):
        targets=json.loads(messages[1]['content'].split('TARGETS TO RETURN:\n')[1].split('\n')[0]);calls.append(list(targets))
        if len(targets)>1:raise ProviderError('request content rejected',status)
        return await original(messages,response_format,**kw)
    t.client.complete=complete
    assert set(asyncio.run(t.run_async([1,2,3])))=={'p1','p2','p3'}
    assert sum(len(c)>1 for c in calls)==1


@pytest.mark.parametrize('status,message',[(401,'unauthorized'),(403,'forbidden'),(400,'model not found'),(400,'unsupported parameter')])
def test_global_errors_do_not_trigger_page_fanout(tmp_path,status,message):
    t=translator(tmp_path);calls=[]
    async def complete(*a,**k):calls.append(1);raise ProviderError(message,status)
    t.client.complete=complete
    with pytest.raises(ProtocolError):asyncio.run(t.run_async([1,2,3]))
    assert len(calls)==1


def test_structured_http_error_is_preserved_without_key_leak(tmp_path,monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        async def handler(request):return httpx.Response(400,json={'error':{'message':'Too much input; key private-secret Bearer abc123 sk-hidden123'}})
        client=AsyncModelClient(Provider(base_url='https://test.invalid',model='test',retries=1),httpx.MockTransport(handler),tmp_path/'trace.jsonl')
        try:
            with pytest.raises(ProviderError,match='Too much input') as err:await client.complete([])
            assert err.value.allows_page_recovery
        finally:await client.close()
    asyncio.run(scenario())
    trace=(tmp_path/'trace.jsonl').read_text()
    assert all(s not in trace for s in ['private-secret','abc123','sk-hidden123'])


def test_one_bad_placement_keeps_artwork_and_other_translations(tmp_path,monkeypatch):
    import slidetwin.best_effort as module
    from slidetwin.extract import enrich_page
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=500,height=300)
        page.insert_text((25,60),'Working title',fontsize=16)
        page.insert_text((25,110),'Failing label',fontsize=14)
        page.draw_rect(fitz.Rect(300,130,450,250),fill=(.2,.4,.8),color=None)
        regions,_=enrich_page(page,1,[])
        pdf.save(source)
    doc=Document(digest(source.read_bytes()),[Page(1,500,300,regions)])
    good,bad=regions[0].id,regions[1].id
    values={good:'正常标题',bad:'模型返回了很长很长的标签，必须完整保留。'*8}
    original_render=module.render
    def injected(source,dest,plans,*args,**kw):
        if any(p.id==bad and p.text for p in plans):raise LayoutError('injected failure',bad)
        return original_render(source,dest,plans,*args,**kw)
    monkeypatch.setattr(module,'render',injected)
    out=tmp_path/'out.pdf';report=publish_best_effort(source,out,tmp_path/'work',doc,[1],Settings(),'test',values,preview=False)
    with fitz.open(source) as src,fitz.open(out) as result:
        assert len(result)==2
        assert '正常标题' in result[1].get_text()
        assert 'Failing label' not in result[1].get_text()
        assert values[bad].replace(' ','') in ''.join(result[1].get_text().split())
        clip=fitz.Rect(310,140,440,240)
        assert src[0].get_pixmap(clip=clip).samples==result[1].get_pixmap(clip=clip).samples
    assert report['pages'][0]['overflow_targets']==[bad]


def test_preview_failure_still_publishes_checked_pdf(tmp_path,monkeypatch):
    import slidetwin.pipeline as pipeline
    from test_best_effort import sample
    source,doc=sample(tmp_path);cfg=Settings(provider=Provider(base_url='https://test.invalid',model='test'))
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    monkeypatch.setattr(pipeline,'extract',lambda *a,**k:doc)
    class Content:
        def __init__(self,*a,**k):pass
        async def run_async(self,*a,**k):return {'a':'电压 6'}
    monkeypatch.setattr(pipeline,'AsyncTranslator',Content)
    monkeypatch.setattr(pipeline,'render_previews',lambda *a,**k:(_ for _ in ()).throw(subprocess.CalledProcessError(1,'pdftoppm')))
    out=tmp_path/'out.pdf';report=pipeline.run(source,out,tmp_path/'work',cfg)
    assert out.exists() and report['final_output_published'] and 'preview_error' in report
    with fitz.open(out) as pdf:assert len(pdf)==2


def test_local_export_reuses_completed_preflight_without_replanning(tmp_path,monkeypatch):
    import slidetwin.best_effort as module
    from test_best_effort import sample
    from slidetwin.render import build_plan
    source,doc=sample(tmp_path);cfg=Settings();work=tmp_path/'work';work.mkdir()
    values={'a':'电压 6'}
    preflight=build_plan(source,doc,values,[1],cfg.layout,work)
    monkeypatch.setattr(module,'build_plan',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Completed geometry was rebuilt')))
    out=tmp_path/'out.pdf'
    publish_best_effort(source,out,work,doc,[1],cfg,'local recovery',values,preview=False,preflight=preflight)
    with fitz.open(out) as pdf:assert len(pdf)==2 and '电压 6' in pdf[1].get_text()


def test_bad_region_preflight_does_not_drop_other_placements(tmp_path,monkeypatch):
    import slidetwin.render as module
    from slidetwin.extract import enrich_page
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page(width=500,height=300)
        p.insert_text((20,60),'First title');p.insert_text((20,100),'Second label')
        regions,_=enrich_page(p,1,[]);pdf.save(source)
    doc=Document(digest(source.read_bytes()),[Page(1,500,300,regions)])
    original=module.visible_native_ink
    def fail(page,region):
        if region.id==regions[0].id:raise ValueError('broken text object')
        return original(page,region)
    monkeypatch.setattr(module,'visible_native_ink',fail)
    plans,failures=module.build_plan(source,doc,{r.id:'译文' for r in regions},[1],Settings().layout,tmp_path)
    assert [p.id for p in plans]==[regions[1].id]
    assert failures[0]['id']==regions[0].id


def test_non_json_provider_reply_is_a_retryable_transport_failure(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        client=AsyncModelClient(Provider(base_url='https://test.invalid',model='test',retries=1),httpx.MockTransport(lambda request:httpx.Response(200,text='<html>gateway error</html>')))
        try:
            with pytest.raises(ProviderError,match='Invalid JSON'):await client.complete([])
        finally:await client.close()
    asyncio.run(scenario())


def test_optional_ocr_failure_keeps_native_pages(tmp_path,monkeypatch):
    import slidetwin.extract as module
    source=tmp_path/'source.pdf';work=tmp_path/'work';work.mkdir()
    with fitz.open() as pdf:
        for n in range(2):pdf.new_page().insert_text((20,60),f'Native page text {n}')
        pdf.save(source)
    raw=work/'docling-document.json';write_json(raw,{'texts':[],'tables':[]})
    write_json(work/'extraction-key.json',{'source_sha256':digest(source.read_bytes()),'selected_pages':[1,2],'docling_sha256':digest(raw.read_bytes())})
    def supplement(page,*a):
        if page.number==0:raise RuntimeError('OCR worker failed on page 1')
        return []
    monkeypatch.setitem(sys.modules,'slidetwin.formula_ocr',SimpleNamespace(supplement=supplement))
    monkeypatch.setitem(sys.modules,'slidetwin.raster_refine',SimpleNamespace(refine=lambda page,descriptors,work:descriptors))
    doc=module.extract(source,work,[1,2])
    assert all(p.regions for p in doc.pages)
    assert [d['page'] for d in doc.diagnostics if d['kind']=='supplemental_ocr_failed']==[1]
    assert read_cache(work/'extraction-key.json')['complete'] is False


def test_broken_native_enrichment_is_confined_to_one_page(tmp_path,monkeypatch):
    import slidetwin.extract as module
    source=tmp_path/'source.pdf';work=tmp_path/'work';work.mkdir()
    with fitz.open() as pdf:
        for n in range(2):pdf.new_page().insert_text((20,60),f'Native page text {n}')
        pdf.save(source)
    raw=work/'docling-document.json';write_json(raw,{'texts':[],'tables':[]})
    write_json(work/'extraction-key.json',{'source_sha256':digest(source.read_bytes()),'selected_pages':[1,2],'docling_sha256':digest(raw.read_bytes())})
    monkeypatch.setitem(sys.modules,'slidetwin.formula_ocr',SimpleNamespace(supplement=lambda *a:[]))
    monkeypatch.setitem(sys.modules,'slidetwin.raster_refine',SimpleNamespace(refine=lambda page,descriptors,work:descriptors))
    enrich=module.enrich_page
    def broken(page,number,*a):
        if number==1:raise RuntimeError('broken page font data')
        return enrich(page,number,*a)
    monkeypatch.setattr(module,'enrich_page',broken)
    doc=module.extract(source,work,[1,2])
    assert not doc.pages[0].regions and doc.pages[1].regions
    out=tmp_path/'out.pdf'
    report=publish_best_effort(source,out,work,doc,[1,2],Settings(),'extraction failure',{r.id:'正常页面译文' for r in doc.pages[1].regions},preview=False)
    with fitz.open(out) as pdf:
        assert len(pdf)==4 and '该页文字提取未完成' in pdf[1].get_text()
        assert '正常页面译文' in pdf[3].get_text()


def test_docling_partial_success_keeps_pages_and_original_ocr_page_numbers(tmp_path,monkeypatch):
    import slidetwin.extract as module
    source=tmp_path/'source.pdf';work=tmp_path/'work'
    with fitz.open() as pdf:
        for n in range(3):pdf.new_page().insert_text((20,60),f'Native text page {n}')
        pdf.save(source)
    result=SimpleNamespace(status='partial_success',errors=[SimpleNamespace(page_no=2)],
        document=SimpleNamespace(save_as_json=lambda path:write_json(path,{'texts':[],'tables':[]})),
        pages=[SimpleNamespace(page_no=1,cells=[]),SimpleNamespace(page_no=3,cells=[SimpleNamespace(from_ocr=True)])])
    class Converter:
        def __init__(self,**kwargs):pass
        def convert(self,path,raises_on_error):
            assert raises_on_error is False
            return result
    monkeypatch.setitem(sys.modules,'docling.document_converter',SimpleNamespace(DocumentConverter=Converter,PdfFormatOption=lambda **k:k))
    monkeypatch.setitem(sys.modules,'docling.datamodel.base_models',SimpleNamespace(InputFormat=SimpleNamespace(PDF='pdf')))
    monkeypatch.setitem(sys.modules,'docling.datamodel.pipeline_options',SimpleNamespace(PdfPipelineOptions=lambda:SimpleNamespace(table_structure_options=SimpleNamespace())))
    monkeypatch.setitem(sys.modules,'slidetwin.formula_ocr',SimpleNamespace(supplement=lambda *a:[]))
    monkeypatch.setitem(sys.modules,'slidetwin.raster_refine',SimpleNamespace(refine=lambda page,descriptors,work:descriptors))
    doc=module.extract(source,work,[1,2,3])
    assert all(p.regions for p in doc.pages)
    assert [p.ocr_used for p in doc.pages]==[False,False,True]
    assert [d['page'] for d in doc.diagnostics if d['kind']=='docling_partial_result']==[2]
    assert not read_cache(work/'extraction-key.json')['docling_complete']


def test_repeated_footer_is_not_forced_onto_two_rows_or_hidden_in_white(tmp_path):
    from slidetwin.extract import enrich_page
    from slidetwin.render import build_plan,render
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=960,height=540)
        page.insert_text((884,525),'Slide 8',fontsize=14,color=(1,1,1))
        page.insert_text((887,525),'Slide 8',fontsize=14,color=(.25,.25,.25))
        regions,_=enrich_page(page,1,[])
        pdf.save(source)
    # Reproduce the merged Docling region in the actual failing course page.
    r=regions[0];r.role='page_footer';r.source='Slide 8Slide ⟦P000⟧'
    r.bbox=list(fitz.Rect(regions[0].bbox)|fitz.Rect(regions[1].bbox))
    r.erase+=regions[1].erase
    r.inline_styles=[{'source':'Slide 8','color':0x404040,'bold':False,'italic':False}]
    regions=[r]
    doc=Document(digest(source.read_bytes()),[Page(1,960,540,regions)])
    values={r.id:'第8页第8页',f'{r.id}_s0':'第8页'}
    plans,failures=build_plan(source,doc,values,[1],Settings().layout,tmp_path)
    assert not failures and len(plans)==1
    assert '<br>' not in plans[0].html_body and plans[0].color!=0xffffff
    out=tmp_path/'out.pdf';render(source,out,plans,[1],Settings().layout)
    with fitz.open(out) as pdf:assert '第8页第8页' in ''.join(pdf[1].get_text().split())
