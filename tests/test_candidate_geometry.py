"""Positional target IDs must not recover unrelated pre-refresh candidates."""

import asyncio
from pathlib import Path
import pytest
from slidetwin.async_translate import AsyncTranslator
from slidetwin.config import Settings
from slidetwin.models import Document, Page, Region, read_cache, write_json


@pytest.mark.parametrize("metadata", [True, False, None])
def test_accepted_candidate_resume_requires_the_current_source(tmp_path, monkeypatch, metadata):
    config = Settings()
    config.translation.glossary = False
    regions = [Region("same", 1, "Current label", [10, 20, 160, 40]),
               Region("moved", 1, "A different field", [10, 55, 160, 75])]
    document = Document("input-hash", [Page(1, 200, 100, regions)])

    class NoProvider:
        usage = {}

    translator = AsyncTranslator(config, NoProvider(), document, Path("unused"), tmp_path)
    ledger = {"source_sha256": document.source_sha256, "config_fingerprint": config.fingerprint(),
              "accepted_candidates": {"same": "同一标签", "moved": "以前的其它字段", "gone": "已删除"}}
    if metadata is True:
        ledger["target_sources"] = {"same": "Current label", "moved": "Previous label", "gone": "Old region"}
    elif metadata is None:
        ledger["target_sources"] = None
        ledger["accepted_candidates"] = None
    write_json(tmp_path/"translation-ledger.json", ledger)

    async def complete(number):
        return {"same": "本次标签", "moved": "不同字段"}

    monkeypatch.setattr(translator, "page_async", complete)
    monkeypatch.setattr(translator, "plan_batches", lambda selected: [[1]])
    assert asyncio.run(translator.run_async([1])) == {"same": "本次标签", "moved": "不同字段"}
    saved = read_cache(tmp_path/"translation-ledger.json")
    assert saved["accepted_candidates"] == ({"same": "同一标签"} if metadata is True else {})
    assert saved["target_sources"] == translator.sources(document.pages[0])
    assert saved["target_source_literals"] == translator.source_literals(document.pages[0])


@pytest.mark.parametrize("previous_literal", ["Value Vin", None, "Value Vout"])
def test_same_placeholder_id_cannot_rebind_a_different_math_literal(tmp_path, monkeypatch, previous_literal):
    config = Settings()
    config.translation.glossary = False
    region = Region("math", 1, "Value ⟦P000⟧", [10, 20, 160, 40], protected={"⟦P000⟧": "Vout"})
    document = Document("input-hash", [Page(1, 200, 100, [region])])

    class NoProvider:
        usage = {}

    translator = AsyncTranslator(config, NoProvider(), document, Path("unused"), tmp_path)
    ledger = {"source_sha256": document.source_sha256, "config_fingerprint": config.fingerprint(),
              "accepted_candidates": {"math": "数值 ⟦P000⟧"}, "target_sources": {"math": region.source}}
    if previous_literal is not None:
        ledger["target_source_literals"] = {"math": previous_literal}
    write_json(tmp_path/"translation-ledger.json", ledger)

    async def complete(number):
        return {"math": "数值 ⟦P000⟧"}

    monkeypatch.setattr(translator, "page_async", complete)
    monkeypatch.setattr(translator, "plan_batches", lambda selected: [[1]])
    asyncio.run(translator.run_async([1]))
    saved = read_cache(tmp_path/"translation-ledger.json")
    assert saved["accepted_candidates"] == ({"math": "数值 ⟦P000⟧"} if previous_literal == "Value Vout" else {})
    assert saved["target_source_literals"] == {"math": "Value Vout"}


def test_retained_style_candidate_records_its_parent_protected_literal(tmp_path):
    from slidetwin.translate import Translator

    region = Region("math", 1, "Value ⟦P000⟧", [10, 20, 160, 40], protected={"⟦P000⟧": "Vout"},
                    inline_styles=[{"source": "Value ⟦P000⟧", "color": 0xFF0000, "bold": True, "italic": False}])
    page = Page(1, 200, 100, [region])
    translator = Translator(Settings(), None, Document("input-hash", [page]), Path("unused"), tmp_path)
    translator.retain_candidate(page, translator.sources(page), "tagged", "<<<math_s0>>>数值 ⟦P000⟧<<<END>>>")
    record = read_cache(tmp_path/"translation-candidates.json")["targets"]["math_s0"]
    assert record["source"] == "Value ⟦P000⟧"
    assert record["source_literal"] == "Value Vout"
