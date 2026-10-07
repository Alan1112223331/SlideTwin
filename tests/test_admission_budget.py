"""Admission waits are local scheduling, separate from model transport deadlines."""
import asyncio
import json
import time

import httpx
import pytest

from slidetwin.async_client import AsyncModelClient
from slidetwin.client import ProviderError
from slidetwin.config import Provider, Settings
from slidetwin.model_pool import AsyncModelPool


def config(**kwargs):
    return Provider(base_url='https://test.invalid', model='test', retries=1,
                    request_deadline_seconds=.025, max_output_tokens=10, **kwargs)


def success(request):
    return httpx.Response(200, json={'choices': [{'message': {'content': '译文'}, 'finish_reason': 'stop'}]})


@pytest.mark.parametrize('pooled', [False, True])
def test_concurrency_queue_cannot_consume_transport_budget(monkeypatch, tmp_path, pooled):
    monkeypatch.setenv('SLIDETWIN_API_KEY', 'test-only')
    async def scenario():
        cls = AsyncModelPool if pooled else AsyncModelClient
        client = cls(config(concurrency=1), httpx.MockTransport(success), tmp_path/'trace.jsonl')
        await client.semaphore.acquire()
        try:
            task = asyncio.create_task(client.complete([]))
            await asyncio.sleep(.07)
            assert not task.done()
            client.semaphore.release()
            assert await task == '译文'
            assert client.usage['requests'] == 1
        finally:
            await client.close()
    asyncio.run(scenario())
    entry = json.loads((tmp_path/'trace.jsonl').read_text(encoding='utf8'))
    assert entry['pool_queue_seconds' if pooled else 'queue_seconds'] >= .06
    assert entry['network_seconds'] < .025


def test_tpm_wait_can_exceed_transport_deadline_without_model_retry(monkeypatch, tmp_path):
    monkeypatch.setenv('SLIDETWIN_API_KEY', 'test-only')
    async def scenario():
        client = AsyncModelClient(config(tokens_per_minute=100, token_rate_utilization=1),
                                  httpx.MockTransport(success), tmp_path/'trace.jsonl')
        client.limiter.window = .08
        await client.limiter.acquire(100)
        try:
            assert await client.complete([]) == '译文'
            assert client.usage['requests'] == 1
        finally:
            await client.close()
    asyncio.run(scenario())
    entry = json.loads((tmp_path/'trace.jsonl').read_text(encoding='utf8'))
    assert entry['queue_seconds'] >= .07 and entry['http_status'] == 200


def test_independent_queue_timeout_does_not_send_or_disable_model(monkeypatch, tmp_path):
    monkeypatch.setenv('SLIDETWIN_API_KEY', 'test-only')
    async def scenario():
        client = AsyncModelClient(config(queue_timeout_seconds=.025, concurrency=1),
                                  httpx.MockTransport(success), tmp_path/'trace.jsonl')
        await client.semaphore.acquire()
        try:
            with pytest.raises(ProviderError) as error:
                await client.complete([])
            assert error.value.kind == 'queue_timeout' and not error.value.retryable
            assert client.usage['requests'] == 0
            client.semaphore.release()
            assert await client.complete([]) == '译文'
        finally:
            await client.close()
    asyncio.run(scenario())
    failed, succeeded = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text(encoding='utf8').splitlines()]
    assert failed['status'] == 'queue_timeout' and failed['network_seconds'] == 0
    assert 'http_status' not in failed and succeeded['status'] == 'ok'


def test_cancelled_local_queue_releases_slots_without_sending(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY', 'test-only')
    async def scenario():
        client = AsyncModelClient(config(concurrency=1), httpx.MockTransport(success))
        await client.semaphore.acquire()
        task = asyncio.create_task(client.complete([]))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        client.semaphore.release()
        try:
            assert await client.complete([]) == '译文'
            assert client.usage['requests'] == 1
        finally:
            await client.close()
    asyncio.run(scenario())


def test_retries_share_transport_budget_even_though_queues_are_excluded(monkeypatch):
    monkeypatch.setenv('SLIDETWIN_API_KEY', 'test-only')
    async def scenario():
        calls = []
        async def handler(request):
            calls.append(time.monotonic())
            await asyncio.sleep(.04 if len(calls)==1 else .2)
            raise httpx.ReadTimeout('temporary')
        provider = config()
        provider.request_deadline_seconds = .15
        provider.retries = 5
        provider.retry_base_delay_seconds = 0
        provider.retry_max_delay_seconds = 0
        pool = AsyncModelPool(provider, httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError):
                await pool.complete([])
            assert len(calls) == 2
        finally:
            await pool.close()
    asyncio.run(scenario())


def test_fallback_shares_primary_account_quota():
    pool = AsyncModelPool(config(fallback_models=['backup']))
    assert pool.clients[0].limiter is pool.clients[1].limiter
    asyncio.run(pool.close())


def test_queue_budget_is_validated_and_does_not_invalidate_translation_cache():
    settings = Settings(provider=config())
    before = settings.fingerprint()
    settings.provider.queue_timeout_seconds = 3600
    settings.validate()
    assert settings.fingerprint() == before
    settings.provider.queue_timeout_seconds = -1
    with pytest.raises(ValueError):
        settings.validate()
