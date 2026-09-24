from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import re
import tomllib
from urllib.parse import urlsplit

from .models import digest


@dataclass
class Provider:
    base_url: str = ""
    model: str = ""
    api_key_env: str = "SLIDETWIN_API_KEY"
    api_key_file: str = ""
    vision: bool = False
    protocol: str = "auto"
    timeout_seconds: float = 180
    max_output_tokens: int = 131072
    temperature: float | None = 0.1
    retries: int = 3
    stream: bool = True
    token_parameter: str = "max_tokens"
    extra_body: dict = field(default_factory=dict)
    concurrency: int = 8
    request_deadline_seconds: float = 1800
    allowed_models: list[str] = field(default_factory=list)
    tokens_per_minute: int = 0
    rate_limit_backoff_seconds: float = 60
    worker_models: list[str] = field(default_factory=list)
    fallback_models: list[str] = field(default_factory=list)
    requests_per_minute: int = 0
    token_rate_utilization: float = 0.8
    context_window_tokens: int = 262144
    context_utilization: float = 0.8
    model_max_output_tokens: int = 262144
    tokenizer_file: str = ""
    read_timeout_seconds: float = 120
    connect_timeout_seconds: float = 10
    fallback_cooldown_seconds: float = 120
    rate_limit_source: str = "unconfigured"
    primary_attempts: int = 2
    retry_base_delay_seconds: float = 1
    retry_max_delay_seconds: float = 15

    @property
    def max_in_flight(self) -> int:
        # 0 means no extra concurrency cap below the configured RPM capacity.
        return self.concurrency or self.requests_per_minute or 8

    def validate_model(self):
        for model in [self.model,*self.worker_models,*self.fallback_models]:
            if self.allowed_models and model not in self.allowed_models:
                raise ValueError(f'Model is not in provider.allowed_models: {model}')

    def key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if value:
            return value
        if self.api_key_file:
            text = Path(self.api_key_file).read_text(encoding="utf-8-sig").strip()
            match = re.search(r"\bsk-[A-Za-z0-9_-]+", text)
            if match:
                return match.group()
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "http")):
                    return line.split("=", 1)[-1].strip().strip("\"'")
        raise ValueError(f"Missing API key: set {self.api_key_env} or provider.api_key_file")


@dataclass
class Translation:
    target_language: str = "Simplified Chinese"
    neighbor_pages: int = 2
    glossary: bool = True
    review: bool = True
    max_context_characters: int = 0
    image_dpi: int = 120
    image_neighbors: int = 0
    context_mode: str = "adaptive"
    related_pages: int = 3
    pages_per_request: int = 1
    preserve_terms: list[str] = field(default_factory=list)
    batch_mode: str = "auto"
    output_expansion_ratio: float = 1.5
    image_policy: str = "ocr_only"


@dataclass
class Layout:
    font_regular: str = ""
    font_bold: str = ""
    min_font_scale: float = 0.75
    line_height: float = 1.12
    render_dpi: int = 110
    strict: bool = True

    def fonts(self) -> tuple[Path, Path]:
        pairs = [
            (self.font_regular, self.font_bold or self.font_regular),
            ("C:/Windows/Fonts/Deng.ttf", "C:/Windows/Fonts/Dengb.ttf"),
            ("/usr/share/fonts/truetype/slidetwin/NotoSansSC-Regular.ttf", "/usr/share/fonts/truetype/slidetwin/NotoSansSC-Bold.ttf"),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc"),
        ]
        if self.font_regular:
            pairs = pairs[:1]
        for regular, bold in pairs:
            if regular and Path(regular).is_file() and Path(bold).is_file():
                return Path(regular).resolve(), Path(bold).resolve()
        raise ValueError("CJK fonts not found; set layout.font_regular and layout.font_bold")


@dataclass
class Settings:
    provider: Provider = field(default_factory=Provider)
    translation: Translation = field(default_factory=Translation)
    layout: Layout = field(default_factory=Layout)

    @classmethod
    def load(cls, path: Path) -> Settings:
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        unknown = set(data) - {"provider", "translation", "layout"}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        if data.get("provider", {}).get("temperature") == "omit":
            data["provider"]["temperature"] = None
        obj = cls(Provider(**data.get("provider", {})), Translation(**data.get("translation", {})), Layout(**data.get("layout", {})))
        for target, attr in [(obj.provider, "api_key_file"), (obj.provider,"tokenizer_file"), (obj.layout, "font_regular"), (obj.layout, "font_bold")]:
            value = getattr(target, attr)
            if value and not Path(value).is_absolute():
                setattr(target, attr, str((path.parent / value).resolve()))
        obj.validate()
        return obj

    def validate(self):
        p = self.provider
        p.validate_model()
        parsed = urlsplit(p.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) API base URL without credentials, query or fragment")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote endpoints require HTTPS")
        if not p.model or p.protocol not in {"auto", "tagged", "plain", "json", "json_schema"}:
            raise ValueError("Specify model and a supported protocol")
        if p.token_parameter not in {"max_tokens", "max_completion_tokens"} or p.retries < 1:
            raise ValueError("Invalid token_parameter or retries")
        if not 0 <= p.concurrency <= 10000 or p.request_deadline_seconds <= 0:
            raise ValueError("Invalid concurrency or request deadline")
        if min(p.tokens_per_minute,p.requests_per_minute,p.rate_limit_backoff_seconds) < 0:
            raise ValueError('Invalid provider rate limits')
        if p.primary_attempts<1 or p.retry_base_delay_seconds<0 or p.retry_max_delay_seconds<p.retry_base_delay_seconds:
            raise ValueError('Invalid retry attempts/backoff settings')
        if not 0 < p.token_rate_utilization <= 1 or not 0 < p.context_utilization <= 1:
            raise ValueError('Invalid context/token utilization')
        if min(p.context_window_tokens,p.model_max_output_tokens,p.max_output_tokens,p.read_timeout_seconds,p.connect_timeout_seconds)<=0:
            raise ValueError('Model capacities and timeouts must be positive')
        if set(p.extra_body) & {"model", "messages", "tools", "tool_choice", "response_format", "stream"}:
            raise ValueError("extra_body may not override model, messages, tools, protocol or streaming")
        if not 0.5 <= self.layout.min_font_scale <= 1 or not 0.9 <= self.layout.line_height <= 2:
            raise ValueError("Invalid layout scaling/line height")
        if min(self.translation.neighbor_pages, self.translation.image_neighbors) < 0:
            raise ValueError("Neighbor counts must be nonnegative")
        if self.translation.context_mode not in {"document", "hierarchical", "adaptive"} or self.translation.related_pages < 0:
            raise ValueError("Invalid translation context settings")
        if self.translation.pages_per_request < 1 or self.translation.batch_mode not in {'auto','fixed'}:
            raise ValueError('Invalid batch mode/page count')
        if self.translation.image_policy not in {'ocr_only','never'} or self.translation.output_expansion_ratio < 1:
            raise ValueError('Invalid image policy/output estimate')

    def fingerprint(self) -> str:
        data = asdict(self)
        data["provider"].pop("api_key_file")
        data["provider"].pop("api_key_env")
        for key in ('primary_attempts','retry_base_delay_seconds','retry_max_delay_seconds'):data['provider'].pop(key,None)
        for key in ("concurrency", "request_deadline_seconds", "timeout_seconds", "retries", "stream", "allowed_models", "tokens_per_minute", "rate_limit_backoff_seconds", "requests_per_minute", "token_rate_utilization", "read_timeout_seconds", "connect_timeout_seconds", "fallback_cooldown_seconds", "rate_limit_source"):
            data["provider"].pop(key, None)
        if not data['provider']['worker_models']:data['provider'].pop('worker_models')
        if not data['translation']['preserve_terms']:data['translation'].pop('preserve_terms')
        # No secret values ever enter a ledger or cache filename.
        return digest(str(data))
