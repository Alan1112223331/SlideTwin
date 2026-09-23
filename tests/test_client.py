import json
import httpx
import pytest

from slidetwin.client import ModelClient, ProviderError
from slidetwin.config import Provider, Settings


def test_plain_request_never_sends_tools_or_response_format(monkeypatch):
    monkeypatch.setenv("SLIDETWIN_API_KEY", "private-secret")
    def handler(request):
        payload = json.loads(request.content)
        assert not {"tools", "tool_choice", "response_format"} & payload.keys()
        assert request.headers["Authorization"] == "Bearer private-secret"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "译文"}}]})
    client = ModelClient(Provider(base_url="https://test.invalid/v1", model="arbitrary"), httpx.MockTransport(handler))
    try:
        assert client.complete([{"role": "user", "content": "test"}]) == "译文"
    finally:
        client.close()


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404])
def test_provider_errors_do_not_echo_credentials_or_response_bodies(monkeypatch, status):
    monkeypatch.setenv("SLIDETWIN_API_KEY", "private-secret")
    client = ModelClient(Provider(base_url="https://test.invalid/v1", model="x"),
                         httpx.MockTransport(lambda r: httpx.Response(status, text="private-secret")))
    with pytest.raises(ProviderError) as exc:
        client.complete([])
    assert "private-secret" not in str(exc.value)
    assert str(status) in str(exc.value)
    client.close()


def test_rate_limit_retries_and_truncation_rejected(monkeypatch):
    monkeypatch.setenv("SLIDETWIN_API_KEY", "secret")
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    client = ModelClient(Provider(base_url="https://test.invalid/v1", model="x"), httpx.MockTransport(handler), sleep=lambda t: None)
    with pytest.raises(ProviderError, match="Incomplete"):
        client.complete([])
    assert len(calls) == 2
    client.close()


@pytest.mark.parametrize("url", ["http://remote.example/v1", "https://key@example.com/v1", "https://example.com/v1?key=secret"])
def test_unsafe_endpoint_config_rejected(url):
    with pytest.raises(ValueError):
        Settings(provider=Provider(base_url=url, model="x")).validate()


def test_streaming_translation_ignores_reasoning_and_counts_usage(monkeypatch):
    monkeypatch.setenv("SLIDETWIN_API_KEY", "secret")
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "internal reasoning"}}]},
        {"choices": [{"index": 0, "delta": {"content": "中文"}}]},
        {"choices": [{"index": 0, "delta": {"content": "译文"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
    ]
    data = "".join("data: " + json.dumps(x, ensure_ascii=False) + "\n\n" for x in chunks) + "data: [DONE]\n\n"
    client = ModelClient(Provider(base_url="https://test.invalid/v1", model="x"),
                         httpx.MockTransport(lambda r: httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})))
    assert client.complete([]) == "中文译文"
    assert client.usage["prompt_tokens"] == 10
    client.close()


def test_truncated_stream_is_never_treated_as_complete(monkeypatch):
    monkeypatch.setenv("SLIDETWIN_API_KEY", "secret")
    data = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    client = ModelClient(Provider(base_url="https://test.invalid/v1", model="x"),
                         httpx.MockTransport(lambda r: httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})))
    with pytest.raises(ProviderError, match="partial translation discarded"):
        client.complete([])
    client.close()
