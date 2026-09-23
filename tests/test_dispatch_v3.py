import asyncio
import json
from pathlib import Path
import time

import httpx
import pymupdf as fitz
import pytest

from slidetwin.async_translate import AsyncTranslator
from slidetwin.config import Provider, Settings
from slidetwin.model_pool import AsyncModelPool
from slidetwin.models import Document, Page, Region
from slidetwin.protocol import ProtocolError, parse_content_response
from slidetwin.rate_limit import RollingRateLimiter
from slidetwin.render import styled_html


def document(count=100,words=10):
    return Document('test-hash',[Page(n,600,400,[Region(f'p{n}',n,'input output memory '*words,[10,60,500,150])],
                                    f'PAGE {n} context:\n'+('input output memory '*words),ocr_used=False) for n in range(1,count+1)])


def test_parser_only_requires_addressable_nonempty_content():
    sources={'a':'Voltage 5','b':'Taylor A. Example'}
    result=parse_content_response('Here are the translations:\n<<<a>>>电压 6>>>END>>>\n<<<b>>>Taylor A. Example>>>END>>>',sources,'tagged')
    assert result=={'a':'电压 6','b':'Taylor A. Example'}
    assert parse_content_response('{"a":"English prose","extra":"ignored"}',{'a':'Original prose'},'json')=={'a':'English prose'}
    with pytest.raises(ProtocolError,match='Missing/empty'):
        parse_content_response('<<<a>>> <<<END>>>',sources,'tagged')


def test_unmatched_emphasis_keeps_sentence_and_reports_style_warning():
    r=Region('a',1,'5 hours',[0,0,100,30],inline_styles=[{'source':'5','color':0xff0000,'bold':True,'italic':False}])
    warnings=[]
    html=styled_html(r,'需要六小时',{'a_s0':'五'}, {},warnings)
    assert html=='需要六小时'
    assert warnings[0]['kind']=='unmatched_emphasis'


def test_native_pages_never_attach_images_even_as_ocr_neighbors(tmp_path):
    pdf=tmp_path/'source.pdf'
    with fitz.open() as src:
        for _ in range(3):src.new_page(width=600,height=400)
        src.save(pdf)
    doc=document(3);doc.pages[1].ocr_used=True
    cfg=Settings();cfg.provider.vision=True;cfg.translation.image_neighbors=1
    t=AsyncTranslator(cfg,None,doc,pdf,tmp_path/'work')
    messages=t.messages(t.grouped_page([1,2,3]),t.sources(t.grouped_page([1,2,3])),'tagged')
    parts=messages[1]['content']
    assert sum(p['type']=='image_url' for p in parts)==1
    assert [p['text'] for p in parts if p['type']=='text' and p['text'].startswith('Original source')]==['Original source PAGE 2']
    cfg.translation.image_neighbors=0
    assert isinstance(t.messages(doc.pages[0],t.sources(doc.pages[0]),'tagged')[1]['content'],str)
    assert t.estimate_batch([1,2,3])['ocr_image_pages']==[2]


def test_all_100_pages_fit_one_group_when_tokens_fit(tmp_path):
    cfg=Settings();cfg.translation.glossary=False
    t=AsyncTranslator(cfg,None,document(),Path('unused'),tmp_path)
    groups=t.plan_batches(list(range(1,101)))
    assert groups==[list(range(1,101))]
    plan=json.loads((tmp_path/'batch-plan.json').read_text())['batches'][0]
    assert plan['context_total_estimated']<=int(262144*.8)
    assert plan['review_input_tokens_estimated']>plan['input_tokens_estimated']


def test_token_planner_packs_until_next_page_exceeds_80_percent(tmp_path):
    cfg=Settings();cfg.translation.glossary=False;cfg.translation.context_mode='hierarchical'
    cfg.provider.context_window_tokens=24000
    t=AsyncTranslator(cfg,None,document(40,30),Path('unused'),tmp_path)
    groups=t.plan_batches(list(range(1,41)))
    assert len(groups)>1 and len(groups[0])>4
    for i,group in enumerate(groups):
        estimate=t.estimate_batch(group)
        assert estimate['fits'] and estimate['context_total_estimated']<=19200
        if i<len(groups)-1:
            assert not t.estimate_batch(group+[groups[i+1][0]])['fits']
    cfg.provider.tokens_per_minute=18000
    assert t.estimate_batch(groups[0])['admission_capacity']==14400


def test_rolling_limiter_uses_full_rpm_and_80_percent_tpm():
    async def scenario():
        limiter=RollingRateLimiter(rpm=8,tpm=1000,window_seconds=.06)
        begin=time.monotonic()
        tickets=await asyncio.gather(*(limiter.acquire(100) for _ in range(8)))
        assert time.monotonic()-begin<.04
        assert len(tickets)==8 and sum(t['tokens'] for t in tickets)==800
        with pytest.raises(ValueError):await limiter.acquire(801)
        ninth=asyncio.create_task(limiter.acquire(100))
        await asyncio.sleep(.015);assert not ninth.done()
        # Even reclaiming TPM must not bypass RPM.
        await limiter.settle(tickets[0],0)
        await asyncio.sleep(.005);assert not ninth.done()
        await asyncio.wait_for(ninth,.2)
    asyncio.run(scenario())


def test_settlement_immediately_releases_excess_tokens_while_other_request_runs():
    async def scenario():
        limiter=RollingRateLimiter(rpm=20,tpm=1000,window_seconds=2)
        first=await limiter.acquire(400)
        other=await limiter.acquire(400)
        waiting=asyncio.create_task(limiter.acquire(300))
        await asyncio.sleep(.01);assert not waiting.done()
        await limiter.settle(first,100)
        await asyncio.wait_for(waiting,.1)
        assert other['tokens']==400
    asyncio.run(scenario())


def test_eight_concurrent_primary_calls_fill_freed_slot_before_slowest_finishes(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        release=asyncio.Event();started=[];active=peak=0
        async def handler(request):
            nonlocal active,peak
            payload=json.loads(request.content);i=int(payload['messages'][0]['content'])
            assert payload['model']=='primary'
            started.append(i);active+=1;peak=max(peak,active)
            if i==0:await release.wait()
            else:await asyncio.sleep(.01)
            active-=1
            return httpx.Response(200,json={'choices':[{'message':{'content':'translation'},'finish_reason':'stop'}]})
        client=AsyncModelPool(Provider(base_url='https://test.invalid',model='primary',fallback_models=['backup'],concurrency=8),httpx.MockTransport(handler))
        task=asyncio.gather(*(client.complete([{'role':'user','content':str(i)}]) for i in range(10)))
        try:
            for _ in range(50):
                if len(started)==10:break
                await asyncio.sleep(.005)
            assert len(started)==10 and not release.is_set() and peak==8
            release.set();assert len(await task)==10
            assert client.usage['by_model']['backup']['requests']==0
        finally:release.set();await client.close()
    asyncio.run(scenario())


def test_queued_requests_recheck_primary_health_before_starting(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        models=[]
        async def handler(request):
            model=json.loads(request.content)['model'];models.append(model)
            await asyncio.sleep(.005)
            if model=='primary':return httpx.Response(503)
            return httpx.Response(200,json={'choices':[{'message':{'content':'translation'},'finish_reason':'stop'}]})
        client=AsyncModelPool(Provider(base_url='https://test.invalid',model='primary',fallback_models=['backup'],concurrency=1,retries=2),httpx.MockTransport(handler))
        try:
            assert len(await asyncio.gather(*(client.complete([]) for _ in range(4))))==4
            assert models.count('primary')==1 and models.count('backup')==4
        finally:await client.close()
    asyncio.run(scenario())


def test_later_page_checkpoint_is_written_without_waiting_for_first_page(tmp_path):
    cfg=Settings();cfg.translation.glossary=False;cfg.translation.review=False
    cfg.translation.batch_mode='fixed';cfg.translation.pages_per_request=1
    async def scenario():
        release=asyncio.Event()
        class Client:
            usage={}
            async def complete(self,*args,**kwargs):
                page=kwargs['label']['page']
                if page==1:await release.wait()
                return f'<<<p{page}>>>模型译文<<<END>>>'
        t=AsyncTranslator(cfg,Client(),document(3),Path('unused'),tmp_path)
        task=asyncio.create_task(t.run_async([1,2,3]))
        for _ in range(50):
            if (tmp_path/'translation-ledger.json').exists():break
            await asyncio.sleep(.005)
        ledger=json.loads((tmp_path/'translation-ledger.json').read_text(encoding='utf-8'))
        assert 1 not in ledger['completed_pages'] and 2 in ledger['completed_pages']
        assert not task.done()
        release.set();assert len(await task)==3
    asyncio.run(scenario())


def test_single_slot_cli_still_uses_async_content_policy(tmp_path,monkeypatch):
    import slidetwin.pipeline as pipeline
    from test_best_effort import sample
    source,doc=sample(tmp_path)
    cfg=Settings(provider=Provider(base_url='https://test.invalid',model='test',concurrency=1))
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    monkeypatch.setattr(pipeline,'extract',lambda *args,**kwargs:doc)
    class Content:
        def __init__(self,*args,**kwargs):pass
        async def run_async(self,selected):return {'a':'电压 6'}
    monkeypatch.setattr(pipeline,'AsyncTranslator',Content)
    result=pipeline.run(source,tmp_path/'out.pdf',tmp_path/'work',cfg,preview=False)
    assert result['status'] in {'automated_checks_passed','completed_with_warnings'}
    # This synthetic fixture omits exact redaction glyphs; geometry fallback
    # is allowed. The changed number must still be exported without a new call.
    with fitz.open(tmp_path/'out.pdf') as pdf:assert '6' in pdf[1].get_text()


def test_json_schema_counts_against_input_budget():
    from slidetwin.budget import TokenCounter
    from slidetwin.protocol import response_format
    counter=TokenCounter(Provider())
    messages=[{'role':'user','content':'translate'}]
    schema=response_format({'target_'+str(i):'source' for i in range(100)},'json_schema')
    assert counter.request(messages,schema)>counter.messages(messages)+1000


def test_source_hollow_square_is_program_formatting():
    from slidetwin.extract import native_bullet
    assert native_bullet({'c':'❑','font':'Wingdings'})


def test_connected_raster_word_uses_letter_height_instead_of_tiny_noise():
    from slidetwin.render import raster_typography
    with fitz.open() as vector:
        p=vector.new_page(width=220,height=90)
        p.insert_text((30,55),'MMMMMMM',fontsize=24,color=(.75,.05,.05))
        box=p.search_for('MMMMMMM')[0]
        # A scan can join neighboring letters across one row of pixels.
        p.draw_line((box.x0+1,49),(box.x1-1,49),color=(.75,.05,.05),width=.5)
        pix=p.get_pixmap(matrix=fitz.Matrix(3,3))
        with fitz.open() as raster:
            page=raster.new_page(width=220,height=90)
            page.insert_image(page.rect,stream=pix.tobytes('png'))
            region=Region('word',1,'MMMMMMM',list(box),size=24,native=False)
            measured,_,_=raster_typography(page,region)
            assert measured.size>16


def test_large_actual_draft_splits_review_without_repeating_initial_translation(tmp_path):
    cfg=Settings();cfg.translation.glossary=False;cfg.provider.context_window_tokens=20000
    doc=document(10,2)
    class Client:
        usage={};calls=[]
        async def complete(self,messages,response_format=None,**kwargs):
            self.calls.append(kwargs['label'])
            assert kwargs['label']['has_draft']
            assert t.counter.request(messages,response_format)+kwargs['max_output_tokens']<=16000
            sources=json.loads(messages[1]['content'].split('TARGETS TO RETURN:\n')[1].split('\n')[0])
            return ''.join(f'<<<{key}>>>译文<<<END>>>' for key in sources)
    c=Client();t=AsyncTranslator(cfg,c,doc,Path('unused'),tmp_path)
    group=t.grouped_page(list(range(1,11)));sources=t.sources(group)
    result=asyncio.run(t.review_async(group,sources,{key:'译文'*750 for key in sources}))
    assert set(result)==set(sources) and len(c.calls)>1
    assert any(e['kind']=='review_capacity_split' for e in t.events)
    assert not any(e['kind']=='targeted_repair' for e in t.events)


def test_oversized_terminology_context_is_complete_across_budgeted_calls(tmp_path):
    cfg=Settings();cfg.provider.context_window_tokens=6000
    doc=document(15,40)
    class Client:
        pieces=[]
        async def complete(self,messages,**kwargs):
            assert t.counter.messages(messages)+kwargs['max_output_tokens']<=4800
            self.pieces.append(messages[1]['content'].split('COURSE CONTEXT:\n')[1])
            return 'memory -> 存储器'
    c=Client();t=AsyncTranslator(cfg,c,doc,Path('unused'),tmp_path)
    asyncio.run(t.glossary_async())
    assert len(c.pieces)>1 and ''.join(c.pieces)==t.context


def test_adaptive_context_accounts_for_tpm_admission_limit(tmp_path):
    cfg=Settings();cfg.provider.tokens_per_minute=12000
    t=AsyncTranslator(cfg,None,document(40,30),Path('unused'),tmp_path)
    assert t.counter.text(t.context)<t.counter.capacity*.5
    context,_=t.prompt_context(t.document.pages[0])
    assert context!=t.context and context.startswith('FULL COURSE INDEX')


def test_timeout_after_streamed_content_retains_received_targets(monkeypatch):
    from slidetwin.async_client import AsyncModelClient
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        class Interrupted(httpx.AsyncByteStream):
            async def __aiter__(self):
                chunk={'choices':[{'delta':{'content':'<<<a>>>已经收到的译文<<<END>>>'}}]}
                yield ('data: '+json.dumps(chunk)+'\n\n').encode()
                await asyncio.sleep(.2)
        calls=[]
        async def handler(request):
            calls.append(request)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Interrupted())
        client=AsyncModelClient(Provider(base_url='https://test.invalid',model='test',request_deadline_seconds=.02),httpx.MockTransport(handler))
        try:
            result=await client.complete([])
            assert parse_content_response(result,{'a':'received','b':'missing'},'tagged',allow_missing=True)=={'a':'已经收到的译文'}
            assert len(calls)==1
        finally:await client.close()
    asyncio.run(scenario())
