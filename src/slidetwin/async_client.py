"""Bounded asynchronous OpenAI-compatible transport, shared across documents."""
from __future__ import annotations

import asyncio
import json
import random
import time
import re
import base64
import io
import warnings
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .client import ProviderError, endpoint_error
from .config import Provider
from .budget import TokenCounter
from .rate_limit import RollingRateLimiter


class RequestBudget:
    """Logical transport/retry budget; local admission consumes a separate budget."""

    def __init__(self, seconds, queue_seconds=0):
        self.remaining = float(seconds)
        self.queue_remaining = float(queue_seconds) if queue_seconds else None

    async def sleep(self, seconds):
        if seconds >= self.remaining:
            raise ProviderError('Model retry deadline exhausted before next attempt', kind='deadline')
        started = time.monotonic()
        try:
            await asyncio.sleep(seconds)
        finally:
            self.remaining -= time.monotonic() - started

    async def wait(self, awaitable):
        """Count only admission time, including cancellation-safe semaphore waits."""
        started = time.monotonic()
        try:
            if self.queue_remaining is None:
                return await awaitable
            if self.queue_remaining <= 0:
                if hasattr(awaitable, 'close'):
                    awaitable.close()
                raise TimeoutError
            async with asyncio.timeout(max(.001, self.queue_remaining)):
                return await awaitable
        except TimeoutError:
            raise ProviderError('Local admission queue timeout; model request was not sent',
                                kind='queue_timeout', retryable=False) from None
        finally:
            if self.queue_remaining is not None:
                self.queue_remaining -= time.monotonic() - started


class AsyncModelClient:
    def __init__(self, config: Provider, transport=None, trace_path: Path | None = None):
        self.config = config
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(config.read_timeout_seconds,connect=config.connect_timeout_seconds), transport=transport,
                                     limits=httpx.Limits(max_connections=config.max_in_flight, max_keepalive_connections=config.max_in_flight),
                                     follow_redirects=False)
        self.semaphore = asyncio.Semaphore(config.max_in_flight)
        self.cooldown_until = 0.0
        self.limiter=RollingRateLimiter(config.requests_per_minute,config.tokens_per_minute,config.token_rate_utilization)
        self.counter=TokenCounter(config)
        self.effective_tpm = self.limiter.tpm
        self.trace_path = trace_path
        self.trace_errors = 0
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "usage_reported_requests": 0}

    async def close(self):
        await self.http.aclose()

    def estimate_tokens(self,messages,output_tokens):
        return self.counter.messages(messages)+output_tokens

    async def dispatch(self,estimated):
        return await self.limiter.acquire(estimated)

    async def _stream(self, response, entry=None, started=None, partial=None):
        if "text/event-stream" not in response.headers.get("content-type", ""):
            await response.aread()
            try:return response.json()
            except ValueError:raise ProviderError('Invalid JSON response from model endpoint',kind='invalid_response') from None
        pieces, reason, usage, done = partial if partial is not None else [], None, {}, False
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(body)
            except ValueError:
                raise ProviderError("Invalid event-stream JSON from model endpoint",kind='invalid_response') from None
            if not isinstance(obj,dict):raise ProviderError('Invalid event-stream object',kind='invalid_response')
            if "error" in obj:
                raise ProviderError("Model endpoint reported a streaming error")
            if obj.get("usage"):
                usage = obj["usage"]
            choices=obj.get('choices',[])
            if not isinstance(choices,list):raise ProviderError('Invalid event-stream choices',kind='invalid_response')
            for choice in choices:
                if not isinstance(choice,dict) or not isinstance(choice.get('delta',{}),dict):
                    raise ProviderError('Invalid event-stream choice',kind='invalid_response')
                if choice.get("index", 0) == 0:
                    value = choice.get("delta", {}).get("content")
                    if isinstance(value, str):
                        pieces.append(value)
                        if entry is not None and value and 'first_token_seconds' not in entry:
                            entry['first_token_seconds'] = round(time.monotonic()-started,3)
                    if choice.get("finish_reason"):
                        reason = choice["finish_reason"]
        if not done and reason is None and not ''.join(pieces).strip():
            raise ProviderError("Interrupted event stream; partial translation discarded")
        if not done and reason is None:reason='length'
        return {"choices": [{"finish_reason": reason, "message": {"content": "".join(pieces)}}], "usage": usage}

    async def complete(self, messages, response_format=None, *, max_output_tokens=None, expected_output_tokens=None, label=None, deadline_seconds=None, _budget=None):
        c = self.config
        c.validate_model()
        payload = {"model": c.model, "messages": messages, c.token_parameter: max_output_tokens or c.max_output_tokens,
                   "stream": c.stream, **c.extra_body}
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        if response_format is not None:
            payload["response_format"] = response_format
        estimated=self.counter.request(messages,response_format)+(expected_output_tokens if expected_output_tokens is not None else payload[c.token_parameter])
        budget = _budget or RequestBudget(deadline_seconds if deadline_seconds is not None else c.request_deadline_seconds,
                                          c.queue_timeout_seconds)
        for attempt in range(c.retries):
            delay = self.cooldown_until-time.monotonic()
            if delay > 0:
                await budget.sleep(delay)
            started = time.monotonic()
            queued_at = started
            entry = {"queued_at": datetime.now(timezone.utc).isoformat(), "model": c.model, "attempt": attempt+1, **(label or {})}
            retry_delay = 0.0
            ticket=None;actual_tokens=None;partial=[];acquired=False;sent_at=None
            try:
                await budget.wait(self.semaphore.acquire())
                acquired = True
                ticket = await budget.wait(self.dispatch(estimated))
                if budget.remaining <= 0:
                    raise ProviderError('Model transport/retry deadline exhausted', kind='deadline')
                if acquired:
                    entry['queue_seconds']=round(time.monotonic()-started,3)
                    entry['reserved_tokens']=estimated
                    entry['context_budget_tokens']=self.counter.capacity
                    entry['image_count']=sum(p.get('type')=='image_url' for m in messages if isinstance(m.get('content'),list) for p in m['content'])
                    started = time.monotonic()
                    sent_at = started
                    entry['started_at']=datetime.now(timezone.utc).isoformat()
                    self.usage["requests"] += 1
                    async with asyncio.timeout(max(.001,budget.remaining)):
                        async with self.http.stream("POST", c.base_url.rstrip("/")+"/chat/completions", json=payload,
                                                    headers={"Authorization": "Bearer "+c.key()}) as response:
                            entry["http_status"] = response.status_code
                            entry['headers_seconds']=round(time.monotonic()-started,3)
                            if response.status_code in {408, 429, 500, 502, 503, 504}:
                                await response.aread()
                                error=endpoint_error(response,c.key())
                                retry_delay=max(error.retry_after,min(c.retry_max_delay_seconds,c.retry_base_delay_seconds*2**attempt)*(1+random.random()*.2))
                                if response.status_code == 429:
                                    retry_delay=max(retry_delay,c.rate_limit_backoff_seconds)
                                    entry['effective_tokens_per_minute']=self.effective_tpm
                                    self.cooldown_until = max(self.cooldown_until, time.monotonic()+retry_delay)
                                    await self.limiter.cooldown(retry_delay)
                                actual_tokens=0
                                error.retry_after=retry_delay
                                raise error
                            if not response.is_success:
                                await response.aread()
                                actual_tokens=0
                                error=endpoint_error(response,c.key())
                                entry['error']=str(error)
                                raise error
                            data = await self._stream(response,entry,started,partial)
                            entry['response_finished_at']=datetime.now(timezone.utc).isoformat()
                    try:
                        choice = data["choices"][0]
                        if choice.get("finish_reason") not in {None, "stop", "eos", "length"}:
                            raise ProviderError("Incomplete/refused model response",kind='refused_response',retryable=False)
                        value = choice["message"].get("content")
                        if isinstance(value, list):
                            value = "".join(x.get("text", "") for x in value if x.get("type") == "text")
                        if not isinstance(value, str) or not value.strip():
                            raise ProviderError("Model returned no text; reasoning/tool-only output is unsupported",kind='empty_response')
                        if data.get("usage"):
                            # Optional accounting must never invalidate content.
                            try:
                                usage={k:int(data['usage'][k]) for k in ('prompt_tokens','completion_tokens')}
                                if any(v<0 for v in usage.values()):raise ValueError('negative usage')
                            except (ValueError,TypeError,KeyError,AttributeError,OverflowError):
                                entry['usage_warning']='Missing/invalid token accounting; retained response and conservative quota reservation'
                            else:
                                actual_tokens=sum(usage.values());entry['usage']=usage
                                self.usage['usage_reported_requests']+=1
                                for key,tokens in usage.items():self.usage[key]+=tokens
                        entry["status"] = "truncated" if choice.get('finish_reason')=='length' else "ok"
                        entry['finish_reason']=choice.get('finish_reason')
                        return value
                    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                        raise ProviderError("Invalid model response",kind='invalid_response') from None
            except (TimeoutError, httpx.TransportError) as exc:
                kind='deadline' if isinstance(exc,TimeoutError) else type(exc).__name__
                entry['error_kind']=kind
                entry['error']='Request deadline or transport timeout/error'
                if ''.join(partial).strip():
                    entry['status']='partial_transport_failure'
                    # Let the protocol layer retain usable IDs and request only
                    # missing ones. Never discard received translation text.
                    return ''.join(partial)
                entry["status"] = "timeout"
                if attempt+1 >= c.retries:
                    raise ProviderError(f"Model request exceeded its deadline or connection failed ({kind})",kind=kind) from None
                retry_delay=min(c.retry_max_delay_seconds,c.retry_base_delay_seconds*2**attempt)*(1+random.random()*.2)
            except ProviderError as exc:
                entry['error_kind']=exc.kind;entry['error']=str(exc)
                if ''.join(partial).strip():
                    entry['status']='partial_stream_failure'
                    return ''.join(partial)
                entry["status"] = "queue_timeout" if exc.kind=='queue_timeout' else "failed"
                if not exc.retryable or attempt+1 >= c.retries:
                    raise
                retry_delay=max(retry_delay,exc.retry_after,min(c.retry_max_delay_seconds,c.retry_base_delay_seconds*2**attempt)*(1+random.random()*.2))
            finally:
                if sent_at is not None:
                    budget.remaining -= time.monotonic() - sent_at
                if acquired:self.semaphore.release()
                if ticket is not None:await self.limiter.settle(ticket,actual_tokens)
                entry.setdefault('queue_seconds', round(time.monotonic()-queued_at,3))
                entry['network_seconds'] = round(time.monotonic()-sent_at,3) if sent_at is not None else 0
                entry['total_seconds'] = round(time.monotonic()-queued_at+entry.get('pool_queue_seconds',0),3)
                entry["seconds"] = entry['network_seconds'] if sent_at is not None else entry['queue_seconds']
                if retry_delay:entry['retry_delay_seconds']=round(retry_delay,3)
                if self.trace_path:
                    try:
                        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                        with self.trace_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(entry, ensure_ascii=False)+"\n")
                    except OSError:
                        self.trace_errors+=1
                        warnings.warn('Request timing log could not be written; model response retained',RuntimeWarning)
            if retry_delay:
                await budget.sleep(retry_delay)
        raise ProviderError("Model retry limit reached")
