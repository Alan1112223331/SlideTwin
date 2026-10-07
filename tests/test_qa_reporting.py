"""Character/geometry checks and final-product diagnostic provenance."""
from dataclasses import replace

import pymupdf as fitz
import pytest

from slidetwin.models import Document, Page, digest, read_cache, write_json
from slidetwin.qa import compact, merge_final_checks, text_layer_check, verify
from slidetwin.render import Placement
from slidetwin.worker import public_diagnostics, public_issue


def placement(text='Expected', frame=(20, 20, 150, 50)):
    return Placement('p0001_r0000', 1, text, list(frame), list(frame), 12, 1,
                     'left', False, 0, True, [], extractable_text=text)


def test_explicit_font_presentation_forms_preserve_math_and_numeric_distinctions():
    assert compact('ﬁnd a ﬂow・动态⸺续') == compact('find a flow·动态——续')
    assert compact('V2') != compact('V3')
    assert compact('x²') != compact('x2')
    assert compact('𝒙') != compact('x')
    assert compact('a × b') != compact('a x b')
    assert compact('①') != compact('1')
    assert compact('&gt;') != compact('>')


def test_text_elsewhere_cannot_hide_a_missing_target(tmp_path):
    source = tmp_path / 'source.pdf'
    output = tmp_path / 'output.pdf'
    with fitz.open() as pdf:
        pdf.new_page(width=300, height=200)
        pdf.save(source)
        page = pdf.new_page(width=300, height=200)
        page.insert_text((20, 130), 'Expected')
        pdf.save(output)
    report = verify(source, output, [1], [placement()], tmp_path)
    failure = next(i for i in report['failures'] if i['kind'] == 'inserted_text_not_extractable')
    assert failure['source_page'] == 1 and failure['translated_page'] == 2
    assert failure['region_ids'] == ['p0001_r0000']
    assert failure['reason_code'] == 'target_characters_not_recovered'


def test_stream_order_does_not_override_character_geometry():
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        # The second word is emitted first in the PDF content stream.
        page.insert_text((65, 40), 'WORLD', fontsize=12)
        page.insert_text((20, 40), 'Hello', fontsize=12)
        assert 'HelloWORLD' not in page.get_text().replace('\n', '')
        assert text_layer_check(page, placement('Hello WORLD')) is None
        assert text_layer_check(page, placement('Hello WORL3')) is not None


def test_unmapped_font_glyph_is_a_text_layer_problem_not_an_assumed_visual_loss():
    class Page:
        def get_text(self, mode=None, clip=None):
            if mode == 'rawdict':
                return {'blocks': []}
            return 'Q睠'

        def get_texttrace(self):
            return [{'chars': [(0xFFFD, 30560, (35, 35), (35, 20, 42, 40))]}]

    result = text_layer_check(Page(), placement('Q1'))
    assert result['reason_code'] == 'invalid_unicode_mapping'
    assert result['unmapped_glyphs'] == 1
    assert result['missing_characters'] == ['1']
    assert result['category'] == 'text_layer'


def test_asset_only_formula_does_not_require_literal_protected_tokens():
    with fitz.open() as pdf:
        page = pdf.new_page()
        entry = replace(placement('⟦P000⟧'), extractable_text='', assets={'math.png': 'math.png'})
        assert text_layer_check(page, entry) is None


def test_final_recheck_replaces_intermediate_qa_without_dropping_layout_evidence():
    original = {'pages': [{'source_page': 7, 'issues': [
        {'kind': 'original_page_pixels_changed', 'page': 7},
        {'kind': 'missing_font_glyph', 'id': 'p0007_r0000', 'characters': ['∑']},
    ]}]}
    final = {'passed': True, 'phase': 'final', 'failures': [],
             'pages': [{'source_page': 7, 'chinese_page': 1, 'bilingual_original_page': 1,
                        'bilingual_translated_page': 2, 'translated_page': 2}]}
    merged = merge_final_checks(original, final)
    assert merged['pages'][0]['issues'][0]['kind'] == 'missing_font_glyph'
    historical = merged['pages'][0]['intermediate_diagnostics']
    assert historical == [{'kind': 'original_page_pixels_changed', 'page': 7, 'phase': 'intermediate'}]
    assert merged['final_qa']['passed']
    assert len(original['pages'][0]['issues']) == 2


def test_report_includes_qa_and_style_pages_without_unplaced_targets(tmp_path):
    result = {'pages': [{'source_page': 8, 'issues': [
        {'id': 'p0008_r0002', 'kind': 'inserted_text_not_extractable',
         'reason_code': 'invalid_unicode_mapping', 'unmapped_glyphs': 3,
         'reason': '/private/log.txt', 'private_path': '/private/key.txt'}]}]}
    write_json(tmp_path / 'layout-plan.json', {'warnings': [
        {'page': 3, 'id': 'p0003_r0000_s0', 'kind': 'unmatched_emphasis', 'reason': 'private'}]})
    report = public_diagnostics(result, [3, 8], tmp_path)
    first, second = report['page_issues']
    assert first['source_page'] == 3 and first['warnings'][0]['category'] == 'text_style'
    assert second['source_page'] == 8 and second['chinese_page'] == 2
    assert second['bilingual_original_page'] == 3 and second['bilingual_translated_page'] == 4
    assert second['unplaced_targets'] == []
    assert second['issues'][0]['unmapped_glyphs'] == 3
    assert '/private/' not in str(report)


def test_report_preserves_each_non_overflow_failure_type_and_ids(tmp_path):
    result = {'pages': [{'source_page': 1, 'overflow_targets': ['a', 'b'], 'issues': [
        {'kind': 'missing_font_glyph', 'id': 'a', 'characters': ['𝔽']},
        {'kind': 'translated_text_overlap', 'ids': ['a', 'b']},
    ]}]}
    page = public_diagnostics(result, [1], tmp_path)['page_issues'][0]
    assert [i['category'] for i in page['issues']] == ['font', 'layout_overlap']
    assert len(page['unplaced_reasons']['a']) == 2
    assert page['unplaced_reasons']['b'][0]['kind'] == 'translated_text_overlap'
    safe = public_issue({'kind': 'local_render_failure', 'reason': 'Traceback /private/secret',
                         'phase': '/private/secret'}, {})
    assert safe['phase'] == 'final' and 'Traceback' not in str(safe)


def test_product_specific_issues_use_the_actual_pdf_page_index(tmp_path):
    issue = {'kind': 'inserted_text_not_extractable', 'page': 8, 'id': 'target'}
    result = {'product_qa': {'chinese': {'failures': [issue]}, 'bilingual': {'failures': [issue]}}}
    page = public_diagnostics(result, [3, 8], tmp_path)['page_issues'][0]
    chinese, bilingual = page['issues']
    assert chinese['output_mode'] == 'chinese' and chinese['translated_page'] == 2
    assert bilingual['output_mode'] == 'bilingual' and bilingual['translated_page'] == 4
    assert chinese['bilingual_original_page'] == bilingual['bilingual_original_page'] == 3


@pytest.mark.parametrize('placement_hash_matches', [True, False])
def test_worker_checks_final_product_bytes_and_saves_both_hashes(tmp_path, monkeypatch, placement_hash_matches):
    from slidetwin.config import Settings
    import slidetwin.worker as worker

    job = tmp_path / 'job'
    job.mkdir()
    with fitz.open() as pdf:
        for text in ('First', 'Second'):
            pdf.new_page(width=300, height=200).insert_text((20, 50), text)
        pdf.save(job / 'input.pdf')
    write_json(job / 'job.json', {'outputs': ['chinese', 'bilingual'], 'status': 'queued'})

    def run(source, output, work, *args, **kwargs):
        with fitz.open(source) as original, fitz.open() as result:
            for number in range(2):
                for _ in range(2):
                    result.insert_pdf(original, from_page=number, to_page=number)
            result.save(output)
        Document(digest(source.read_bytes()), [Page(1, 300, 200, []), Page(2, 300, 200, [])]).save(work / 'document.json')
        write_json(work / 'final-layout-plan.json', {'placements': [],
                   'output_sha256': digest(output.read_bytes()) if placement_hash_matches else 'stale-plan-hash'})
        return {'status': 'automated_checks_passed'}

    monkeypatch.setattr(worker, 'convert_docling', lambda *a, **k: {'warnings': []})
    monkeypatch.setattr(worker, '_load_config', lambda: Settings())
    monkeypatch.setattr(worker, 'run', run)
    report = worker.run_job(job)
    if not placement_hash_matches:
        assert report['status'] == 'completed_with_warnings'
        assert report['outputs'] == {'chinese': 'ready', 'bilingual': 'ready'}
        assert report['product_qa'] == {}
        assert 'final_placement_evidence_unavailable' in report['warnings']
        return
    assert report['status'] == 'completed'
    assert set(report['product_qa']) == {'chinese', 'bilingual'}
    assert report['page_issues'] == []
    for mode, qa in report['product_qa'].items():
        assert qa['passed'] and qa['phase'] == 'final'
        assert qa['output_sha256'] == digest((job / 'artifacts' / f'{mode}.pdf').read_bytes())
        assert read_cache(job / 'work' / f'{mode}-final-qa' / 'qa.json')['output_sha256'] == qa['output_sha256']
