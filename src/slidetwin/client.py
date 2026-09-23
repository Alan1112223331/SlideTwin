from __future__ import annotations

import time
import json
import re
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import httpx

from .config import Provider


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, *, kind='provider', retry_after=0., retryable=None):
        super().__init__(message)
        self.status = status
        self.kind=kind
        self.retry_after=retry_after
        self.retryable=(status in {None,408,429,500,502,503,504}) if retryable is None else retryable

    @property
    def allows_page_recovery(self):
        if self.status is None and not self.retryable:return False
        if self.status in {None, 408, 413, 422, 429, 500, 502, 503, 504}:
            return True
        if self.status != 400:
            return False
        # Splitting cannot fix invalid account/model/request configuration.
        return not re.search(r"api.?key|unauthori[sz]ed|balance|credit|payment|model.{0,50}(not found|not exist|invalid|unsupported)|unsupported.{0,30}(parameter|model)|unknown parameter|invalid.{0,30}(temperature|response_format|max_tokens)", str(self), re.I)


def retry_after_seconds(value):
    try:
        seconds=float(value)
        return max(0.,seconds) if math.isfinite(seconds) else 0.
    except (ValueError,TypeError):
        try:
            date=parsedate_to_datetime(value)
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            return max(0.,(date-datetime.now(timezone.utc)).total_seconds())
        except (ValueError,TypeError,OverflowError):return 0.


def endpoint_error(response, key=""):
    """Only expose a bounded, credential-redacted structured provider error."""
    detail = ""
    try:
        data = response.json()
        error = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(error, dict):
            detail = str(error.get("message", ""))
    except (ValueError, TypeError):
        pass
    if key:
        detail = detail.replace(key, "[redacted]")
    detail = re.sub(r"(?i)bearer\s+\S+|sk-[A-Za-z0-9_-]+|data:[^\s]+", "[redacted]", detail)
    detail = " ".join(detail.split())[:800]
    return ProviderError(f"Model endpoint returned HTTP {response.status_code}" + (f": {detail}" if detail else ""), response.status_code,
                         kind='http',retry_after=retry_after_seconds(response.headers.get('retry-after','')))


class ModelClient:
    """Chat Completions transport with redacted structured error messages."""
    def __init__(self, config: Provider, transport=None, sleep=time.sleep):
        self.config = config
        self.http = httpx.Client(timeout=config.timeout_seconds, transport=transport, follow_redirects=False)
        self.sleep = sleep
        self.usage = {"requests": 0, "prompt_tokens": None, "completion_tokens": None, "usage_reported_requests": 0}

    def close(self):
        self.http.close()

    def _stream_data(self, response: httpx.Response) -> dict:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            response.read()
            return response.json()
        pieces, reason, usage, done = [], None, {}, False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(body)
            except ValueError:
                raise ProviderError("Invalid event-stream JSON from model endpoint") from None
            if "error" in obj:
                raise ProviderError("Model endpoint reported an error during streaming")
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                value = choice.get("delta", {}).get("content")
                if isinstance(value, str):
                    pieces.append(value)
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
        if not done and reason is None:
            raise ProviderError("Interrupted event stream; partial translation discarded")
        return {"choices": [{"finish_reason": reason, "message": {"content": "".join(pieces)}}], "usage": usage}

    def complete(self, messages: list[dict], response_format: dict | None = None) -> str:
        c = self.config
        c.validate_model()
        payload = {"model": c.model, "messages": messages, c.token_parameter: c.max_output_tokens, "stream": c.stream, **c.extra_body}
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        if response_format is not None:
            payload["response_format"] = response_format
        for attempt in range(c.retries):
            try:
                self.usage["requests"] += 1
                with self.http.stream("POST", c.base_url.rstrip("/") + "/chat/completions", json=payload,
                                      headers={"Authorization": "Bearer " + c.key()}) as response:
                    if not response.is_success:
                        response.read()
                    if response.is_success:
                        try:
                            data = self._stream_data(response)
                        except (ValueError, TypeError, KeyError, IndexError):
                            raise ProviderError("Invalid model response") from None
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt + 1 < c.retries:
                    self.sleep(min(2 ** attempt, 8))
                    continue
                raise ProviderError("Model endpoint timed out or connection failed; no response recorded") from None
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt + 1 < c.retries:
                retry = response.headers.get("retry-after", "")
                self.sleep(min(float(retry), 30) if retry.isdigit() else min(2 ** attempt, 8))
                continue
            if not response.is_success:
                raise endpoint_error(response, c.key())
            try:
                choice = data["choices"][0]
                if choice.get("finish_reason") not in {None, "stop", "eos"}:
                    raise ProviderError("Incomplete/refused model response (finish_reason=" + str(choice.get("finish_reason")) + ")")
                value = choice["message"].get("content")
                if isinstance(value, list):
                    value = "".join(x.get("text", "") for x in value if x.get("type") == "text")
                if not isinstance(value, str) or not value.strip():
                    raise ProviderError("Model returned no text; reasoning/tool-only output is unsupported")
                if data.get("usage"):
                    self.usage["usage_reported_requests"] += 1
                    for key in ("prompt_tokens", "completion_tokens"):
                        if data["usage"].get(key) is not None:
                            self.usage[key] = (self.usage[key] or 0) + int(data["usage"][key])
                return value
            except (KeyError, IndexError, TypeError, ValueError):
                raise ProviderError("Model endpoint returned an invalid Chat Completions response") from None
        raise ProviderError("Model retry limit reached")
