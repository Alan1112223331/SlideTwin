"""One primary model, bounded transport retries, and ordered fallback."""
import asyncio
import copy
import random
import time
import warnings

from .async_client import AsyncModelClient
from .client import ProviderError


class AsyncModelPool:
    def __init__(self,config,transport=None,trace_path=None):
        config.validate_model();self.config=config
        self.semaphore=asyncio.Semaphore(config.max_in_flight)
        legacy=[m for m in config.worker_models if m!=config.model]
        models=list(dict.fromkeys([config.model,*(config.fallback_models or legacy)]))
        self.clients=[];self.unavailable_until=[0.]*len(models)
        self.failure_streak=[0]*len(models)
        for model in models:
            provider=copy.deepcopy(config);provider.model=model;provider.worker_models=[];provider.fallback_models=[];provider.retries=1
            if 'Qwen3-VL' in model or 'Qwen3-Omni' in model:provider.extra_body.pop('enable_thinking',None)
            self.clients.append(AsyncModelClient(provider,transport,trace_path))

    @property
    def usage(self):
        keys=('requests','prompt_tokens','completion_tokens','usage_reported_requests')
        return {**{k:sum(c.usage[k] for c in self.clients) for k in keys},
                'by_model':{c.config.model:dict(c.usage) for c in self.clients}}

    def choose(self,attempts):
        now=time.monotonic()
        ready=[i for i in range(len(self.clients)) if self.unavailable_until[i]<=now]
        if 0 in ready and attempts[0]<self.config.primary_attempts:return 0
        fallback=[i for i in ready if i!=0]
        if fallback:return min(fallback,key=lambda i:(attempts[i],i))
        if ready:return ready[0]
        return min(range(len(self.clients)),key=lambda i:self.unavailable_until[i])

    async def complete(self,messages,response_format=None,**kwargs):
        attempts=[0]*len(self.clients);last_error=None;backoff=0.
        deadline=time.monotonic()+self.config.request_deadline_seconds
        for attempt in range(self.config.retries):
            if backoff:
                if time.monotonic()+backoff>=deadline:
                    raise ProviderError('Logical request retry deadline exhausted',kind='deadline') from last_error
                await asyncio.sleep(backoff)
            while True:
                # Model health is checked after acquiring a slot. Cooldowns and
                # retry sleeps happen outside that slot so unrelated work runs.
                async with self.semaphore:
                    index=self.choose(attempts)
                    wait=max(0.,self.unavailable_until[index]-time.monotonic())
                    remaining=deadline-time.monotonic()
                    if remaining<=wait:
                        raise ProviderError('Logical request deadline cannot accommodate provider cooldown',kind='deadline') from last_error
                    if not wait:
                        call_kwargs=dict(kwargs)
                        call_kwargs['deadline_seconds']=remaining
                        call_kwargs['label']={**(kwargs.get('label') or {}),'pool_attempt':attempt+1,
                            'fallback':index!=0,'retry_wait_seconds':round(backoff,3),
                            'previous_error_kind':last_error.kind if last_error else None}
                        attempts[index]+=1
                        try:
                            value=await self.clients[index].complete(messages,response_format,**call_kwargs)
                            self.failure_streak[index]=0;self.unavailable_until[index]=0.
                            return value
                        except ProviderError as exc:
                            if not exc.retryable:raise
                            last_error=exc;self.failure_streak[index]+=1
                            # A single request timeout or malformed response does
                            # not declare the entire primary model unavailable.
                            if exc.status==429:
                                cooldown=max(exc.retry_after,self.config.rate_limit_backoff_seconds)
                            elif exc.status in {500,502,503,504} or self.failure_streak[index]>=2:
                                cooldown=max(exc.retry_after,self.config.fallback_cooldown_seconds)
                            else:cooldown=exc.retry_after
                            if cooldown:self.unavailable_until[index]=max(self.unavailable_until[index],time.monotonic()+cooldown)
                            if attempt+1>=self.config.retries:raise
                            backoff=min(self.config.retry_max_delay_seconds,self.config.retry_base_delay_seconds*2**attempt)*(1+random.random()*.2)
                        break
                # Every model is cooling down. Release the concurrency slot and
                # respect the server's delay rather than truncating Retry-After.
                await asyncio.sleep(wait)
        raise last_error or ProviderError('Model retry limit reached')

    async def close(self):
        results=await asyncio.gather(*(c.close() for c in self.clients),return_exceptions=True)
        self.close_errors=[type(r).__name__ for r in results if isinstance(r,Exception)]
        if self.close_errors:warnings.warn('Model connection cleanup failed; completed translations retained',RuntimeWarning)
