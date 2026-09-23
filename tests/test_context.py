from pathlib import Path
import json
import pytest

from slidetwin.config import Settings, Provider
from slidetwin.models import Document, Page, Region
from slidetwin.translate import Translator


def document():
    return Document("source-hash", [
        Page(1, 600, 400, [], "Earlier definition of the accumulator"),
        Page(2, 600, 400, [Region("a", 2, "Accumulator", [10, 10, 100, 30]), Region("b", 2, "Multiplier", [10, 40, 100, 60])], "Entire target page with both terms"),
        Page(3, 600, 400, [], "Later example with the same term"),
    ])


def test_plain_mode_keeps_document_page_and_neighbors(tmp_path):
    config = Settings(provider=Provider(protocol="plain"))
    t = Translator(config, None, document(), Path("unused.pdf"), tmp_path)
    message = t.messages(document().pages[1], {"a": "Accumulator"}, "plain")[1]["content"]
    for text in ["Earlier definition", "Entire target page", "Later example", "Multiplier"]:
        assert text in message
    assert "TARGETS TO RETURN" in message
    assert "geometry" not in json.dumps({"a": "Accumulator"})


def test_large_context_is_never_silently_truncated(tmp_path):
    config = Settings()
    config.translation.max_context_characters = 4
    with pytest.raises(ValueError, match="NOT silently truncated"):
        Translator(config, None, document(), Path("unused.pdf"), tmp_path)


def test_auto_falls_back_for_model_that_cannot_produce_structured_output(tmp_path):
    class PlainOnly:
        def complete(self, messages, response_format=None):
            assert response_format is None
            prompt = messages[1]["content"]
            if "Return ONLY" not in prompt:
                return "I do not output IDs."
            return "累加器" if 'TARGETS TO RETURN:\n{"a"' in prompt else "乘法器"
    translator = Translator(Settings(), PlainOnly(), document(), Path("unused.pdf"), tmp_path)
    assert translator.translate_page(document().pages[1]) == {"a": "累加器", "b": "乘法器"}
    assert translator.events[-1]["kind"] == "plain_fallback"


@pytest.mark.parametrize("source", ["What is Data?", "Data Types", "Decision and Performance Evaluation", "DEMO101 | © Dr. Alice Example | AI Models to Hardware"])
def test_unchanged_english_headers_and_footers_are_not_mistaken_for_complete_translation(tmp_path, source):
    from slidetwin.protocol import ProtocolError
    translator = Translator(Settings(), None, document(), Path("unused.pdf"), tmp_path)
    with pytest.raises(ProtocolError, match="untranslated"):
        translator.validate_meaning_coverage({"a": source}, {"a": source})


def test_acronyms_and_named_authors_can_remain_intact(tmp_path):
    translator = Translator(Settings(), None, document(), Path("unused.pdf"), tmp_path)
    values = {"a": "SIMD", "b": "Dr. Alice Example"}
    translator.validate_meaning_coverage(values, values)


@pytest.mark.parametrize("mode", ["tagged", "plain"])
def test_numeric_emphasis_aligns_against_restored_parent(tmp_path, mode):
    from slidetwin.protocol import ProtocolError
    page = Page(1, 600, 400, [Region("a", 1, "Review (⟦P000⟧ hrs)", [1, 1, 300, 30],
                protected={"⟦P000⟧": "5"}, inline_styles=[{"source": "(5 hrs)"}])], "Review (5 hrs)")
    class Model:
        child = "（5小时）"
        def complete(self, messages, response_format=None):
            if mode == "plain":
                return self.child
            return f"<<<a>>>回顾（⟦P000⟧小时）<<<END>>><<<a_s0>>>{self.child}<<<END>>>"
    model = Model()
    translator = Translator(Settings(), model, Document("hash", [page]), Path("unused"), tmp_path)
    sources = {"a_s0": "(5 hrs)"} if mode == "plain" else translator.sources(page)
    draft = {"a": "回顾（⟦P000⟧小时）"} if mode == "plain" else None
    assert translator.batch(page, sources, mode, draft)["a_s0"] == "（5小时）"
    model.child = "（6小时）"
    for path in (translator.cache/"batches").glob("*.json"):
        path.unlink()
    changed=translator.batch(page, sources, mode, draft)
    assert changed['a_s0']=='（6小时）'
    assert translator.language_hints(page,translator.sources(page),{**(draft or {}),**changed})


def test_validated_first_pass_survives_interruption_before_review(tmp_path):
    from slidetwin.client import ProviderError
    class Model:
        calls = 0
        def complete(self, messages, response_format=None):
            self.calls += 1
            if 'TRANSLATION REVIEW:' in messages[1]['content']:
                raise ProviderError('temporary outage')
            return '<<<a>>>累加器<<<END>>><<<b>>>乘法器<<<END>>>'
    config = Settings()
    config.translation.glossary = False
    model = Model()
    translator = Translator(config, model, document(), Path('unused'), tmp_path)
    with pytest.raises(ProviderError):
        translator.run([2])
    assert model.calls == 2
    # Reconstruct the translator like a restarted process. Only review calls the
    # endpoint again; the complete initial translation remains validated.
    resumed = Translator(config, model, document(), Path('unused'), tmp_path)
    with pytest.raises(ProviderError):
        resumed.run([2])
    assert model.calls == 3
    assert not (tmp_path/'translations/page-0002.json').exists()


def test_plain_footer_never_asks_model_to_recreate_immutable_author(tmp_path):
    page = Page(1, 600, 400, [Region("footer", 1, "IC for AI | ⟦P000⟧ AI Models to Hardware", [1, 1, 500, 30],
                protected={"⟦P000⟧": "© Dr. Alice Example |"},
                protected_assets={"⟦P000⟧": {"text": "© Dr. Alice Example |", "bbox": [50, 1, 200, 30]}})],
                "Full original page context: IC for AI | © Dr. Alice Example | AI Models to Hardware")
    class Model:
        def complete(self, messages, response_format=None):
            prompt = messages[1]["content"]
            assert "Full original page context" in prompt
            target = prompt.split("TARGETS TO RETURN:\n")[1].split("\n")[0]
            value = json.loads(target)["footer"]
            assert "Example" not in value
            return {"IC for AI": "人工智能集成电路", "AI Models to Hardware": "从AI模型到硬件"}[value]
    translator = Translator(Settings(provider=Provider(protocol="plain")), Model(), Document("hash", [page]), Path("unused"), tmp_path)
    assert translator.translate_page(page)["footer"] == "人工智能集成电路 | ⟦P000⟧ 从AI模型到硬件"


def test_mathematical_timing_labels_are_not_mistaken_for_english_prose():
    from slidetwin.extract import immutable_math_label
    assert immutable_math_label('10ns ≤tpff≤ 40ns')
    assert immutable_math_label('tphl')
    assert immutable_math_label('⟦P000⟧~40ns')
    assert not immutable_math_label('Clock 5ns')
    assert not immutable_math_label('Time 5ns')
    assert not immutable_math_label('Hold time')


def test_calendar_semantics_require_month_day_year_and_pm(tmp_path):
    from slidetwin.protocol import ProtocolError
    t=Translator(Settings(),None,document(),Path('unused'),tmp_path)
    source={'a':'⟦P000⟧ Dec ⟦P001⟧ (Tuesday), ⟦P002⟧ PM'}
    t.validate_meaning_coverage(source,{'a':'⟦P001⟧年十二月⟦P000⟧日（星期二），下午⟦P002⟧'})
    for wrong in ['⟦P000⟧年十二月⟦P001⟧日，下午⟦P002⟧','⟦P000⟧月⟦P001⟧日，下午⟦P002⟧','⟦P001⟧年十二月⟦P000⟧日，⟦P002⟧']:
        with pytest.raises(ProtocolError):t.validate_meaning_coverage(source,{'a':wrong})
    t.validate_meaning_coverage({'a':'Some layers may retain ⟦P000⟧ bits'},{'a':'某些层可能保留 ⟦P000⟧ 位'})
    t.validate_meaning_coverage({'a':'batch ⟦P000⟧ may reuse weights less'},{'a':'批大小为 ⟦P000⟧ 时，权重复用可能较少'})


def test_protected_math_style_and_invented_neighbor_equation(tmp_path):
    from slidetwin.protocol import ProtocolError
    region=Region('a',1,'thickness ⟦P000⟧',[1,1,100,20],protected={'⟦P000⟧':'tox'},inline_styles=[{'source':'tox'}])
    p=Page(1,200,100,[region]);t=Translator(Settings(),None,Document('s',[p]),Path('unused'),tmp_path)
    t.validate_values(p,t.sources(p),{'a':'厚度 ⟦P000⟧','a_s0':'tox'},None)
    with pytest.raises(ProtocolError,match='invented an equation'):
        t.validate_meaning_coverage({'b':'Sample mean:'},{'b':'样本均值：μ = Σ xi / M'})


def test_reference_author_emphasis_is_not_untranslated_prose_but_book_title_is(tmp_path):
    from slidetwin.protocol import ProtocolError
    suffix=', Example Press, Taylor A. Example'
    region=Region('book',1,'Understanding Data Communication and Networks'+suffix,[1,1,400,30],inline_styles=[{'source':suffix}])
    page=Page(1,600,400,[region],'Books\nReference books:')
    t=Translator(Settings(),None,Document('hash',[page]),Path('unused'),tmp_path)
    t.validate_values(page,t.sources(page),{'book':'理解数据通信与网络'+suffix,'book_s0':suffix},None)
    with pytest.raises(ProtocolError,match='untranslated'):
        t.validate_values(page,t.sources(page),{'book':region.source,'book_s0':suffix},None)
    with pytest.raises(ProtocolError,match='Phrase'):
        t.validate_values(page,t.sources(page),{'book':'理解数据通信与网络, Another Author','book_s0':suffix},None)
    page.context='Ordinary lecture content'
    with pytest.raises(ProtocolError,match='untranslated'):
        t.validate_values(page,t.sources(page),{'book':'理解数据通信与网络'+suffix,'book_s0':suffix},None)
