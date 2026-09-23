"""Burst-capable rolling RPM/TPM admission; finished requests release excess reserve."""
from __future__ import annotations

import asyncio
import time


class RollingRateLimiter:
    def __init__(self,rpm=0,tpm=0,utilization=.8,window_seconds=60):
        self.rpm=rpm;self.tpm=int(tpm*utilization);self.window=window_seconds
        self.entries=[];self.condition=asyncio.Condition();self.cooldown_until=0.

    async def acquire(self,tokens):
        if self.tpm and tokens>self.tpm:
            raise ValueError(f'One request reserves {tokens} tokens, exceeding the configured {self.tpm} token/minute budget')
        async with self.condition:
            while True:
                now=time.monotonic();self.entries=[e for e in self.entries if e['time']+self.window>now]
                count=len(self.entries);used=sum(e['tokens'] for e in self.entries)
                rpm_ok=not self.rpm or count<self.rpm
                tpm_ok=not self.tpm or used+tokens<=self.tpm
                if now>=self.cooldown_until and rpm_ok and tpm_ok:
                    item={'time':now,'tokens':tokens};self.entries.append(item);return item
                deadlines=[self.cooldown_until] if now<self.cooldown_until else []
                if not rpm_ok or not tpm_ok:deadlines.extend(e['time']+self.window for e in self.entries)
                delay=max(.001,min(deadlines)-now)
                try:await asyncio.wait_for(self.condition.wait(),timeout=delay)
                except TimeoutError:pass

    async def settle(self,item,actual_tokens):
        async with self.condition:
            if actual_tokens is not None:item['tokens']=max(0,int(actual_tokens))
            self.condition.notify_all()

    async def cooldown(self,seconds):
        async with self.condition:
            self.cooldown_until=max(self.cooldown_until,time.monotonic()+seconds)
            self.condition.notify_all()
