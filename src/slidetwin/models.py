from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import os
import tempfile
import warnings


def digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def read_cache(path: Path) -> dict:
    """An invalid optional checkpoint is a cache miss, never a document failure.

    Leave the original bytes on disk for diagnosis. Authoritative inputs and
    filesystem permission errors still fail normally.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except FileNotFoundError:
        return {}
    except (ValueError, UnicodeError):
        warnings.warn(f"Ignoring invalid cache: {path}", RuntimeWarning, stacklevel=2)
        return {}


def write_json(path: Path, value) -> None:
    """Atomic writes: an interrupted run never turns a partial ledger into a cache hit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@dataclass
class Region:
    id: str
    page: int
    source: str
    bbox: list[float]
    role: str = "text"
    size: float = 12.0
    color: int = 0
    bold: bool = False
    align: str = "left"
    native: bool = True
    erase: list[list[float]] = field(default_factory=list)
    protected: dict[str, str] = field(default_factory=dict)
    protected_assets: dict[str, dict] = field(default_factory=dict)
    inline_styles: list[dict] = field(default_factory=list)
    direction: list[float] = field(default_factory=lambda: [1.0, 0.0])
    docling_ref: str = ""


@dataclass
class Page:
    number: int
    width: float
    height: float
    regions: list[Region] = field(default_factory=list)
    context: str = ""
    ocr_used: bool | None = None


@dataclass
class Document:
    source_sha256: str
    pages: list[Page]
    diagnostics: list[dict] = field(default_factory=list)
    schema_version: int = 1

    def save(self, path: Path):
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> Document:
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["pages"] = [Page(**{**p, "regions": [Region(**r) for r in p["regions"]]}) for p in obj["pages"]]
        return cls(**obj)
