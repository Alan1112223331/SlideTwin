import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from slidetwin.async_client import AsyncModelClient
from slidetwin.async_translate import AsyncTranslator
from slidetwin.client import ProviderError
from slidetwin.config import Settings,Provider
from slidetwin.models import Document,Page,Region


def test_verified_recurring_footer_is_reused_with_document_scoped_provenance(tmp_path):
    from dataclasses import asdict
    from slidetwin.models import digest,write_json
    from slidetwin.async_translate import ASYNC_VERSION
    pages=[Page(n,600,400,[Region(f'p{n}',n,'Learning Pipeline',[10,382,150,397])],f'Context {n}') for n in range(1,4)]
    cfg=Settings();cfg.translation.glossary=False;cfg.translation.pages_per_request=3
    class NoCalls:
        usage={}
        async def complete(self,*args,**kwargs):raise AssertionError('Verified recurring footer called the provider')
    t=AsyncTranslator(cfg,NoCalls(),Document('hash',pages),Path('unused'),tmp_path)
    key=digest(ASYNC_VERSION+cfg.fingerprint()+'hash'+t.context+json.dumps(asdict(pages[0]),sort_keys=True))
    write_json(t.cache/'page-0001.json',{'key':key,'reviewed':True,'translations':{'p1':'学习流程'}})
    result=asyncio.run(t.run_async([1,2,3]));assert result=={'p1':'学习流程','p2':'学习流程','p3':'学习流程'}
    provenance=json.loads((tmp_path/'recurring-translation-provenance.json').read_text(encoding='utf8'))
    assert set(provenance)=={'p2','p3'}
    assert all(v['cache_sha256'] for v in provenance.values())


def test_recurring_reuse_never_admits_nonrecurring_body_none_key(tmp_path):
    from slidetwin.models import write_json
    pages=[Page(n,600,400,[Region(f'body{n}',n,'Different body text',[20,100,400,150]),Region(f'foot{n}',n,'Footer',[20,385,100,398])]) for n in range(1,4)]
    t=AsyncTranslator(Settings(),None,Document('hash',pages),Path('unused'),tmp_path)
    t.load_recurring();cache=tmp_path/'record.json';write_json(cache,{})
    t.remember_recurring(pages[0],{'body1':'正文一','foot1':'页脚'},cache)
    assert None not in t.recurring
    assert t.recurring_values(pages[1],t.sources(pages[1]))=={'foot2':'页脚'}


def test_async_transport_overlaps_requests_and_enforces_limit(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        active,peak=0,0
        async def handler(request):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(0.03)
            active-=1
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'译文'}}]})
        c=AsyncModelClient(Provider(base_url='https://test.invalid/v1',model='test',concurrency=3),httpx.MockTransport(handler))
        try:
            results=await asyncio.gather(*(c.complete([]) for _ in range(9)))
            assert results==['译文']*9
            assert peak==3
        finally:await c.close()
    asyncio.run(scenario())


def test_healthy_model_pool_uses_only_primary_and_shares_global_limit(monkeypatch):
    from slidetwin.model_pool import AsyncModelPool
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        active,peak=0,0;models=[]
        async def handler(request):
            nonlocal active,peak
            model=json.loads(request.content)['model'];models.append(model)
            active+=1;peak=max(peak,active);await asyncio.sleep(.03);active-=1
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'译文'}}],'usage':{'prompt_tokens':5,'completion_tokens':2}})
        config=Provider(base_url='https://test.invalid/v1',model='a',worker_models=['a','b'],allowed_models=['a','b'],concurrency=3)
        c=AsyncModelPool(config,httpx.MockTransport(handler))
        try:
            assert await asyncio.gather(*(c.complete([]) for _ in range(8)))==['译文']*8
            assert peak==3 and set(models)=={'a'}
            assert c.usage['requests']==8 and c.usage['prompt_tokens']==40
            assert set(c.usage['by_model'])=={'a','b'}
            assert c.usage['by_model']['b']['requests']==0
        finally:await c.close()
        config.worker_models.append('forbidden')
        with pytest.raises(ValueError,match='allowed_models'):AsyncModelPool(config)
    asyncio.run(scenario())


def test_model_pool_fails_over_transient_error_to_another_allowed_model(monkeypatch):
    from slidetwin.model_pool import AsyncModelPool
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        models=[]
        async def handler(request):
            model=json.loads(request.content)['model'];models.append(model)
            if model=='a':return httpx.Response(503)
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'译文'}}]})
        config=Provider(base_url='https://test.invalid/v1',model='a',worker_models=['a','b'],allowed_models=['a','b'],retries=3)
        client=AsyncModelPool(config,httpx.MockTransport(handler))
        try:
            assert await client.complete([])=='译文'
            assert models==['a','b']
        finally:await client.close()
    asyncio.run(scenario())


def test_disallowed_model_is_rejected_before_any_network_request(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        called=False
        async def handler(request):
            nonlocal called
            called=True
            return httpx.Response(200,json={})
        config=Provider(base_url='https://test.invalid/v1',model='not-allowed',allowed_models=['approved-model'])
        client=AsyncModelClient(config,httpx.MockTransport(handler))
        try:
            with pytest.raises(ValueError,match='allowed_models'):await client.complete([])
            assert not called
        finally:await client.close()
    asyncio.run(scenario())


def test_token_admission_allows_immediate_burst_within_budget(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        starts=[]
        async def handler(request):
            starts.append(time.monotonic())
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'译文'}}]})
        c=AsyncModelClient(Provider(base_url='https://test.invalid/v1',model='test',concurrency=3,tokens_per_minute=60000,max_output_tokens=60),httpx.MockTransport(handler))
        try:
            assert await asyncio.gather(*(c.complete([]) for _ in range(3)))==['译文']*3
            assert starts[-1]-starts[0]<.05
        finally:await c.close()
    asyncio.run(scenario())


def test_total_deadline_discards_incomplete_request(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','private-secret')
    async def scenario():
        async def handler(request):
            await asyncio.sleep(0.2)
            return httpx.Response(200,json={})
        c=AsyncModelClient(Provider(base_url='https://test.invalid/v1',model='test',retries=1,request_deadline_seconds=0.02),httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError,match='deadline'):await c.complete([])
        finally:await c.close()
    asyncio.run(scenario())


def test_language_suspicion_is_only_a_model_review_hint(tmp_path):
    doc=Document('hash',[Page(1,600,400,[Region('a',1,'Accumulator',[1,1,20,20]),Region('b',1,'Multiplier',[1,30,20,50])],'The complete page and its definitions'),
                         Page(2,600,400,[],'A later example explaining both terms')])
    class Client:
        calls=[]
        async def complete(self,messages,response_format=None,**kwargs):
            prompt=messages[1]['content'];self.calls.append(prompt)
            assert 'A later example' in prompt
            if len(self.calls)==1:return '<<<a>>>累加器<<<END>>><<<b>>>Multiplier<<<END>>>'
            targets=json.loads(prompt.split('TARGETS TO RETURN:\n')[1].split('\n')[0])
            assert targets=={'a':'Accumulator','b':'Multiplier'}
            assert 'PROGRAM REVIEW HINTS' in prompt and 'untranslated English' in prompt
            return '<<<a>>>累加器<<<END>>><<<b>>>Multiplier<<<END>>>'
    config=Settings();config.translation.glossary=False
    c=Client();t=AsyncTranslator(config,c,doc,Path('unused'),tmp_path)
    async def scenario():
        values=await t.batch_async(doc.pages[0],t.sources(doc.pages[0]),'auto')
        assert values=={'a':'累加器','b':'Multiplier'}
        assert len(c.calls)==1
        assert await t.batch_async(doc.pages[0],t.sources(doc.pages[0]),'auto')==values
        assert len(c.calls)==1
        assert await t.review_async(doc.pages[0],t.sources(doc.pages[0]),values)==values
        assert len(c.calls)==2
        assert not any(e['kind'] in {'targeted_repair','plain_fallback'} for e in t.events)
    asyncio.run(scenario())


def test_one_failed_page_does_not_cancel_other_pages_and_resume_keeps_draft(tmp_path):
    doc=Document('hash',[Page(i,600,400,[Region(f'p{i}',i,'Accumulator',[1,1,100,20])],f'Full page {i}') for i in [1,2]])
    class Client:
        usage={};calls=[];fail=True
        async def complete(self,messages,response_format=None,**kwargs):
            prompt=messages[1]['content'];self.calls.append(kwargs['label'])
            n=kwargs['label']['page']
            if n==1 and kwargs['label']['has_draft'] and self.fail:raise ProviderError('outage')
            return f'<<<p{n}>>>累加器<<<END>>>'
    config=Settings();config.translation.glossary=False
    c=Client()
    async def scenario():
        t=AsyncTranslator(config,c,doc,Path('unused'),tmp_path)
        with pytest.raises(ValueError,match='1 pages'):await t.run_async([1,2])
        assert (tmp_path/'translations/page-0002.json').exists()
        before=len(c.calls);c.fail=False
        assert len(await AsyncTranslator(config,c,doc,Path('unused'),tmp_path).run_async([1,2]))==2
        assert len(c.calls)==before+1
    asyncio.run(scenario())


def test_adjacent_pages_share_translation_and_review_but_keep_individual_checkpoints(tmp_path):
    doc=Document('hash',[Page(i,600,400,[Region(f'p{i}',i,'Accumulator',[1,1,100,20])],f'Complete page {i}') for i in range(1,4)])
    class Client:
        usage={};calls=[]
        async def complete(self,messages,response_format=None,**kwargs):
            prompt=messages[1]['content'];self.calls.append(prompt)
            assert all(f'Complete page {i}' in prompt for i in range(1,4))
            return ''.join(f'<<<p{i}>>>累加器<<<END>>>' for i in range(1,4))
    config=Settings();config.translation.glossary=False;config.translation.pages_per_request=3
    c=Client()
    async def scenario():
        t=AsyncTranslator(config,c,doc,Path('unused'),tmp_path)
        assert len(await t.run_async([1,2,3]))==3
        assert len(c.calls)==2
        assert 'TRANSLATION REVIEW' in c.calls[1]
        assert len(list((tmp_path/'translations').glob('page-*.json')))==3
        assert len(await t.run_async([1,2,3]))==3
        assert len(c.calls)==2
    asyncio.run(scenario())


def test_group_failure_recovers_pages_once_and_reports_only_actual_failed_page(tmp_path):
    from slidetwin.protocol import ProtocolError
    from slidetwin.models import write_json
    pages=[Page(i,600,400,[Region(f'p{i}',i,'Accumulator',[10,20,200,40])],f'Complete page {i}') for i in range(1,4)]
    cfg=Settings();cfg.translation.glossary=False;cfg.translation.pages_per_request=3
    class Client:
        usage={};calls=[]
        async def complete(self,messages,response_format=None,**kwargs):
            prompt=messages[1]['content'];self.calls.append(prompt)
            ids=json.loads(prompt.split('TARGETS TO RETURN:\n')[1].split('\n')[0])
            if 'p2' in ids:raise ProviderError('unavailable')
            return ''.join(f'<<<{k}>>>累加器<<<END>>>' for k in ids)
    client=Client();t=AsyncTranslator(cfg,client,Document('hash',pages),Path('unused'),tmp_path)
    group=t.grouped_page([1,2,3])
    key,path=t.batch_identity(group,t.sources(group),'auto')
    write_json(path,{'key':key,'translations':{'p1':'累加器','p3':'累加器'}})
    with pytest.raises(ProtocolError,match='1 pages remain incomplete'):
        asyncio.run(t.run_async([1,2,3]))
    failures=json.loads((tmp_path/'translation-failures.json').read_text())
    assert [x['page'] for x in failures]==[2]
    assert (tmp_path/'translations/page-0001.json').exists()
    assert (tmp_path/'translations/page-0003.json').exists()
    assert any('automatic_page_recovery'==x['kind'] for x in t.events)
    # Good pages only need their reviews; their initial grouped translations survive.
    for prompt in client.calls:
        ids=json.loads(prompt.split('TARGETS TO RETURN:\n')[1].split('\n')[0])
        if 'p1' in ids or 'p3' in ids:assert 'TRANSLATION REVIEW' in prompt


def test_rejected_model_text_is_retained_before_validation(tmp_path):
    cfg=Settings();cfg.translation.glossary=False
    page=Page(1,600,400,[Region('p1',1,'Voltage ⟦P000⟧',[10,20,200,40],protected={'⟦P000⟧':'5'})],'Voltage 5')
    class Client:
        usage={}
        async def complete(self,*args,**kwargs):return '<<<p1>>>电压为 6<<<END>>>'
    t=AsyncTranslator(cfg,Client(),Document('hash',[page]),Path('unused'),tmp_path)
    asyncio.run(t.call(page,t.sources(page),'tagged'))
    obj=json.loads((tmp_path/'translation-candidates.json').read_text(encoding='utf8'))
    assert obj['targets']['p1']['text']=='电压为 6'
