import pytest


def test_legacy_symbol_private_code_is_matched_by_its_readable_math_value():
    from slidetwin.protocol import restore_plain_tokens
    assert restore_plain_tokens('当 VDS ≤VGS 时',{'⟦P000⟧':'VDS','⟦P001⟧':'\uf0a3VGS'})=='当 ⟦P000⟧ ⟦P001⟧ 时'
    with pytest.raises(ValueError):restore_plain_tokens('当 VDS ≥VGS 时',{'⟦P000⟧':'VDS','⟦P001⟧':'\uf0a3VGS'})

from slidetwin.protocol import ProtocolError, parse_response, response_format
from slidetwin.protocol import aligned_phrase, restore_plain_tokens


@pytest.mark.parametrize("mode,text", [
    ("tagged", "<<<a>>>乘法器 ⟦P000⟧<<<END>>>\n<<<b>>>累加器<<<END>>>"),
    ("json", '{"a":"乘法器 ⟦P000⟧","b":"累加器"}'),
    ("json_schema", '```json\n{"a":"乘法器 ⟦P000⟧","b":"累加器"}\n```'),
])
def test_all_protocols_match_ids_and_protected_content(mode, text):
    assert parse_response(text, {"a": "multiplier ⟦P000⟧", "b": "accumulator"}, mode) == {"a": "乘法器 ⟦P000⟧", "b": "累加器"}


@pytest.mark.parametrize("text", [
    "<<<a>>>译文<<<END>>>",
    "<<<a>>>译文 ⟦P000⟧ ⟦P000⟧<<<END>>>",
    "<<<a>>>译文 ⟦P000⟧<<<END>>><<<a>>>重复 ⟦P000⟧<<<END>>>",
    "Explanation\n<<<a>>>译文 ⟦P000⟧<<<END>>>",
    "<<<x>>>译文 ⟦P000⟧<<<END>>>",
    "<<<a>>>- 译文 ⟦P000⟧<<<END>>>",
    "<<<a>>>译文 � ⟦P000⟧<<<END>>>",
])
def test_bad_tagged_responses_never_silently_drop_content(text):
    with pytest.raises(ProtocolError):
        parse_response(text, {"a": "text ⟦P000⟧"}, "tagged")


def test_duplicate_json_keys_rejected():
    with pytest.raises(ProtocolError):
        parse_response('{"a":"第一","a":"第二"}', {"a": "source"}, "json")


def test_existing_source_dash_is_preserved_without_allowing_new_lists():
    assert parse_response('- 延迟适用于两种情况', {'a':'- Delay applies to both cases'}, 'plain')['a']=='- 延迟适用于两种情况'
    with pytest.raises(ProtocolError):
        parse_response('- 延迟适用于两种情况\n- 新增项目', {'a':'- Delay applies to both cases'}, 'plain')


def test_model_emphasis_delimiters_are_not_printed_as_document_text():
    assert parse_response('**静态噪声**与**随机**变化',{'a':'static noise and random variation'},'plain')['a']=='静态噪声与随机变化'
    assert parse_response('x**2',{'a':'x**2'},'plain')['a']=='x**2'


def test_plain_has_no_structured_output_dependency_and_normalizes_model_indents():
    assert parse_response("<think>reasoning</think>\n   第一行\n    第二行", {"a": "source"}, "plain") == {"a": "第一行 第二行"}
    assert response_format({"a": "source"}, "plain") is None
    assert response_format({"a": "source"}, "tagged") is None


def test_plain_cannot_ambiguously_map_multiple_targets():
    with pytest.raises(ProtocolError):
        parse_response("译文", {"a": "A", "b": "B"}, "plain")


def test_structured_reply_can_spell_exact_numbers_without_internal_tokens():
    sources = {"a": "Use ⟦P000⟧ hours and ⟦P001⟧ exercises"}
    protected = {"a": {"⟦P000⟧": "5", "⟦P001⟧": "2"}}
    assert parse_response('<<<a>>>用5小时完成2道练习<<<END>>>', sources, 'tagged', protected) == {"a": "用⟦P000⟧小时完成⟦P001⟧道练习"}
    assert parse_response('<<<a>>>用⟦P000⟧小时完成2道练习<<<END>>>', sources, 'tagged', protected) == {"a": "用⟦P000⟧小时完成⟦P001⟧道练习"}
    for invalid in ['用6小时完成2道练习', '用5小时再用5小时完成2道练习', '用⟦P999⟧小时完成2道练习']:
        with pytest.raises(ProtocolError):
            parse_response(f'<<<a>>>{invalid}<<<END>>>', sources, 'tagged', protected)


def test_phrase_alignment_preserves_original_spacing_but_never_accepts_different_meaning():
    assert aligned_phrase("DEMO101：人工智能集成电路 | © Alice Example博士", "DEMO101: 人工智能集成电路") == "DEMO101：人工智能集成电路"
    assert aligned_phrase("数据归一化", "数据标准化") is None
    assert aligned_phrase("主成分分析（PCA）、独立成分分析（ICA）", "(PCA),") == "（PCA）、"
    assert aligned_phrase("主成分分析（PCA）、独立成分分析（ICA）", "(ICA),") is None


def test_plain_mode_accepts_normal_written_numbers_without_special_tokens():
    assert restore_plain_tokens("INT8 + 50%剪枝", {"⟦P000⟧": "50%"}) == "INT8 + ⟦P000⟧剪枝"
    assert restore_plain_tokens("用1个加上1个", {"⟦P000⟧": "1", "⟦P001⟧": "1"}) == "用⟦P000⟧个加上⟦P001⟧个"
    with pytest.raises(ProtocolError):
        restore_plain_tokens("INT8 + 60%剪枝", {"⟦P000⟧": "50%"})
    with pytest.raises(ProtocolError):
        restore_plain_tokens("值为2.01", {"⟦P000⟧": "2"})
    assert restore_plain_tokens("三维张量中的二维网格", {"⟦P000⟧": "3", "⟦P001⟧": "2"}, "⟦P000⟧-D tensor with a ⟦P001⟧-D grid") == "⟦P000⟧维张量中的⟦P001⟧维网格"


def test_clock_colon_is_equivalent_only_in_time_context():
    assert restore_plain_tokens('下午5:00',{'⟦P000⟧':'5.00'},'⟦P000⟧ PM')=='下午⟦P000⟧'
    with pytest.raises(ProtocolError):restore_plain_tokens('值为5:00',{'⟦P000⟧':'5.00'},'Voltage ⟦P000⟧')


def test_decimal_split_by_original_font_assets_is_verified_as_one_number():
    protected={'⟦P000⟧':'1','⟦P001⟧':'ni','⟦P002⟧':'45×1010','⟦P003⟧':'cm-3'}
    source='⟦P001⟧ = concentration = ⟦P000⟧.⟦P002⟧ ⟦P003⟧'
    assert restore_plain_tokens('ni = 本征载流子浓度 = 1.45×1010 cm-3',protected,source)=='⟦P001⟧ = 本征载流子浓度 = ⟦P000⟧.⟦P002⟧ ⟦P003⟧'
    with pytest.raises(ProtocolError):restore_plain_tokens('ni = 浓度 = 1.46×1010 cm-3',protected,source)
def test_inline_latex_variable_cannot_leak_into_pdf():
    from slidetwin.protocol import validate_text,ProtocolError
    assert validate_text('velocity v','速度 $v$')=='速度 v'
    with pytest.raises(ProtocolError):
        validate_text('velocity v',r'速度 $\\frac{a}{b}$')
