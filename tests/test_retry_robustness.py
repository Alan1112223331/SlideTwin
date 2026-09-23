import asyncio
from datetime import datetime,timedelta,timezone
from email.utils import format_datetime
import json
import time

import httpx
import pytest

from slidetwin.async_client import AsyncModelClient
from slidetwin.client import ProviderError,retry_after_seconds
from slidetwin.config import Provider
from slidetwin.model_pool import AsyncModelPool


def settings(**kwargs):
    return Provider(base_url='https://test.invalid',model='primary',retries=3,
                    retry_base_delay_seconds=0,retry_max_delay_seconds=0,**kwargs)


def success(content='译文',**fields):
    return httpx.Response(200,json={'choices':[{'message':{'content':content},'finish_reason':'stop'}],**fields})


def test_single_timeout_retries_primary_without_disabling_it(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        models=[]
        async def handler(request):
            models.append(json.loads(request.content)['model'])
            if len(models)==1:raise httpx.ReadTimeout('not for logging')
            return success()
        pool=AsyncModelPool(settings(fallback_models=['backup']),httpx.MockTransport(handler))
        try:
            assert await pool.complete([])=='译文'
            assert models==['primary','primary']
            assert pool.unavailable_until[0]==0
        finally:await pool.close()
    asyncio.run(scenario())


def test_repeated_primary_timeouts_use_ordered_fallback(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        models=[]
        async def handler(request):
            model=json.loads(request.content)['model'];models.append(model)
            if model=='primary':raise httpx.ConnectError('network unavailable')
            return success()
        pool=AsyncModelPool(settings(fallback_models=['backup']),httpx.MockTransport(handler))
        try:
            assert await pool.complete([])=='译文'
            assert models==['primary','primary','backup']
        finally:await pool.close()
    asyncio.run(scenario())


def test_empty_reply_retries_without_crashing_protocol_layer(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        calls=[]
        async def handler(request):
            calls.append(1);return success('' if len(calls)==1 else '正常内容')
        client=AsyncModelClient(settings(),httpx.MockTransport(handler))
        try:assert await client.complete([])=='正常内容' and len(calls)==2
        finally:await client.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('usage',[{'prompt_tokens':'unknown','completion_tokens':12},{'prompt_tokens':None},[3,4],{'prompt_tokens':-1,'completion_tokens':2}])
def test_invalid_usage_does_not_discard_model_content(monkeypatch,tmp_path,usage):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        client=AsyncModelClient(settings(),httpx.MockTransport(lambda r:success(usage=usage)),tmp_path/'trace.jsonl')
        try:
            assert await client.complete([])=='译文'
            assert client.usage['requests']==1
            assert client.usage['usage_reported_requests']==0
        finally:await client.close()
    asyncio.run(scenario())
    assert json.loads((tmp_path/'trace.jsonl').read_text())['usage_warning']


def test_malformed_stream_tail_preserves_received_targets(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        chunk={'choices':[{'delta':{'content':'<<<a>>>已收到的译文<<<END>>>'}}]}
        response='data: '+json.dumps(chunk)+'\n\ndata: []\n\n'
        client=AsyncModelClient(settings(),httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/event-stream'},text=response)))
        try:
            assert await client.complete([])=='<<<a>>>已收到的译文<<<END>>>'
            assert client.usage['requests']==1
        finally:await client.close()
    asyncio.run(scenario())


def test_retry_after_supports_dates_and_does_not_truncate_seconds():
    assert retry_after_seconds('180')==180
    assert retry_after_seconds('1.5')==1.5
    assert retry_after_seconds('NaN')==0
    assert retry_after_seconds('garbage')==0
    date=format_datetime(datetime.now(timezone.utc)+timedelta(seconds=100))
    assert 98<=retry_after_seconds(date)<=100


def test_provider_cooldown_longer_than_deadline_returns_without_retry(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        calls=[]
        def handler(request):calls.append(1);return httpx.Response(429,headers={'retry-after':'180'},json={'error':{'message':'busy'}})
        pool=AsyncModelPool(settings(request_deadline_seconds=.2),httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError,match='deadline'):await pool.complete([])
            assert len(calls)==1
        finally:await pool.close()
    asyncio.run(scenario())


def test_retry_backoff_releases_slot_for_healthy_work(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        order=[]
        async def handler(request):
            label=json.loads(request.content)['messages'][0]['content'];order.append(label)
            if label=='slow' and order.count('slow')==1:raise httpx.ReadTimeout('temporary')
            return success(label)
        cfg=settings(concurrency=1);cfg.retry_base_delay_seconds=.08;cfg.retry_max_delay_seconds=.08
        pool=AsyncModelPool(cfg,httpx.MockTransport(handler))
        try:
            slow=asyncio.create_task(pool.complete([{'role':'user','content':'slow'}]))
            await asyncio.sleep(.01)
            assert await pool.complete([{'role':'user','content':'healthy'}])=='healthy'
            assert await slow=='slow'
            assert order==['slow','healthy','slow']
        finally:await pool.close()
    asyncio.run(scenario())


def test_authentication_error_is_not_retried_or_failed_over(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        pool=AsyncModelPool(settings(fallback_models=['backup']),httpx.MockTransport(lambda r:httpx.Response(401)))
        try:
            with pytest.raises(ProviderError):await pool.complete([])
            assert pool.usage['requests']==1
        finally:await pool.close()
    asyncio.run(scenario())


def test_failed_optional_trace_write_does_not_discard_response(monkeypatch,tmp_path):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        path=tmp_path/'log.jsonl';path.mkdir()
        client=AsyncModelClient(settings(),httpx.MockTransport(lambda r:success()),path)
        try:
            with pytest.warns(RuntimeWarning,match='timing log'):assert await client.complete([])=='译文'
            assert client.usage['requests']==1 and client.trace_errors==1
        finally:await client.close()
    asyncio.run(scenario())


def test_connection_cleanup_does_not_replace_success_with_error(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY','test-only')
    async def scenario():
        pool=AsyncModelPool(settings(fallback_models=['backup']),httpx.MockTransport(lambda r:success()))
        original_close=pool.clients[0].close
        async def fail():
            await original_close()
            raise RuntimeError('cleanup failure')
        pool.clients[0].close=fail
        assert await pool.complete([])=='译文'
        with pytest.warns(RuntimeWarning,match='cleanup'):await pool.close()
        assert pool.clients[1].http.is_closed
    asyncio.run(scenario())
