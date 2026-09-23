import json
import pymupdf as fitz
import pytest

from slidetwin.config import Provider, Settings
from slidetwin.models import Document, Page, digest
from slidetwin.pipeline import run
import slidetwin.pipeline as pipeline


def test_failure_never_overwrites_existing_good_output_or_input(tmp_path, monkeypatch):
    source = tmp_path/"input.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((30, 30), "test document")
        doc.save(source)
    source_before = source.read_bytes()
    output = tmp_path/"output.pdf"
    output.write_bytes(b"previously reviewed output")
    monkeypatch.setenv("SLIDETWIN_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "extract", lambda *a, **k: Document(digest(source_before), [Page(1, 595, 842)]))
    class FailingTranslator:
        def __init__(self, *a, **k): pass
        def run(self, *a, **k): raise RuntimeError("Simulated upstream failure")
        async def run_async(self, *a, **k): raise RuntimeError("Simulated upstream failure")
    monkeypatch.setattr(pipeline, "Translator", FailingTranslator)
    monkeypatch.setattr(pipeline, "AsyncTranslator", FailingTranslator)
    config = Settings(provider=Provider(base_url="https://example.invalid/v1", model="test"))
    with pytest.raises(RuntimeError, match="upstream"):
        run(source, output, tmp_path/"work", config)
    assert output.read_bytes() == b"previously reviewed output"
    assert source.read_bytes() == source_before
    status = json.loads((tmp_path/"work/run.json").read_text())
    assert status["status"] == "failed"
    assert status["final_output_published"] is False


def test_rejects_pptx_instead_of_silently_flattening(tmp_path):
    with pytest.raises(ValueError, match="PPTX"):
        run(tmp_path/"input.pptx", tmp_path/"out.pdf", tmp_path/"work", Settings())
