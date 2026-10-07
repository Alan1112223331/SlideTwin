"""Run one persisted job in its own process (MuPDF is not shared across threads)."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import traceback

import pymupdf as fitz

from .config import Settings
from .docling_export import convert_docling
from .models import Document, digest, read_cache, write_json
from .pipeline import parse_pages, run
from .service_job import collect_artifacts, job_path, now, update
from .qa import page_indices, verify
from .render import Placement


ISSUE_REASONS = {
    'layout_blocked': ('layout_fit', 'Translated text did not fit the available frame.'),
    'missing_font_glyph': ('font', 'No configured font could render the listed characters.'),
    'translated_text_overlap': ('layout_overlap', 'Translated frames overlap each other.'),
    'local_render_failure': ('rendering', 'Local text rendering failed; other blocks were retained.'),
    'region_preflight_failed': ('rendering', 'This region could not be prepared for rendering.'),
    'source_pixels_retained': ('content_preservation', 'Source pixels were retained because no safe erase geometry was available.'),
    'inserted_text_not_extractable': ('text_layer', 'The expected character sequence could not be recovered within this target frame.'),
    'original_page_pixels_changed': ('graphics', 'The copied original page differs from the source at the verification resolution.'),
    'graphics_outside_edit_regions_changed': ('graphics', 'Pixels changed outside the declared edit regions.'),
    'page_count': ('document_structure', 'The output page count differs from the requested page count.'),
    'page_dimensions': ('document_structure', 'The translated page dimensions differ from the source.'),
    'raster_dimensions': ('document_structure', 'The rendered page dimensions differ from the source.'),
    'missing_translation': ('missing_content', 'No model translation was available for this target.'),
    'unsupported_text_rotation': ('layout_rotation', 'This text rotation could not be rendered safely.'),
    'unmatched_emphasis': ('text_style', 'A source emphasis span could not be placed; base styling was used.'),
    'underline_mapping_unavailable': ('text_style', 'The partial source underline has no matching translated style target; the source line was retained.'),
    'underline_background_unsafe': ('text_style', 'The source underline could not be cleared without risking adjacent graphics or a nonuniform background.'),
    'rotated_text': ('layout_rotation', 'Rotated source text requires a supported local rendering path.'),
    'docling_partial_result': ('extraction', 'Docling returned an incomplete extraction for this page.'),
    'supplemental_ocr_failed': ('extraction', 'Supplemental OCR failed on this page.'),
    'ocr_refinement_failed': ('extraction', 'OCR refinement failed on this page.'),
    'page_enrichment_failed': ('extraction', 'Native and extracted text could not be fully reconciled on this page.'),
    'page_text_unavailable': ('extraction', 'No usable text extraction was available for this page.'),
}


def public_issue(issue: dict, indices: dict[int, dict], *, phase='final', output_mode=None) -> dict:
    """Allowlist diagnostics rather than exposing exceptions or private paths."""
    kind = issue.get('kind', 'unspecified')
    if not isinstance(kind, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', kind):
        kind = 'unspecified'
    category, reason = ISSUE_REASONS.get(kind, ('processing', 'This diagnostic requires review; see its kind and target IDs.'))
    number = issue.get('source_page', issue.get('page'))
    result = {'kind': kind, 'category': category, 'reason': reason,
              'phase': issue.get('phase') if issue.get('phase') in {'final', 'intermediate'} else phase}
    if isinstance(number, int):
        result.update(indices.get(number, {'source_page': number}))
    if output_mode in {'chinese', 'bilingual'}:
        result['output_mode'] = output_mode
    ids = issue.get('region_ids') or issue.get('ids') or ([issue['id']] if issue.get('id') else [])
    result['region_ids'] = [v for v in ids if isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', v)]
    if issue.get('id') in result['region_ids']:
        result['id'] = issue['id']
    if issue.get('ids'):
        result['ids'] = result['region_ids']
    for key in ('pixels', 'expected', 'actual', 'unmapped_glyphs', 'missing_character_count',
                'expected_character_count', 'observed_character_count', 'blocking', 'recovered'):
        if isinstance(issue.get(key), (int, float, bool)):
            result[key] = issue[key]
    for key in ('characters', 'missing_characters'):
        if isinstance(issue.get(key), list):
            result[key] = [v for v in issue[key] if isinstance(v, str) and len(v) == 1][:256]
    for key in ('reason_code', 'check', 'action'):
        value = issue.get(key)
        if isinstance(value, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,120}', value):
            result[key] = value
    if kind == 'inserted_text_not_extractable' and result.get('reason_code') == 'invalid_unicode_mapping':
        result['reason'] = 'The PDF text layer contains unmapped glyphs; this check does not establish visual text loss.'
    return result


def public_diagnostics(result: dict, selected: list[int], work: Path) -> dict:
    """Expose every page diagnostic, including pages without unplaced text."""
    indices = page_indices(selected)
    pages = {}
    document_issues = []

    def add(issue, phase='final', warning=False, output_mode=None):
        safe = public_issue(issue, page_indices(selected, bilingual=output_mode != 'chinese'),
                            phase=phase, output_mode=output_mode)
        number = safe.get('source_page')
        if number not in indices:
            document_issues.append(safe)
            return
        page = pages.setdefault(number, {**indices[number], 'missing_targets': [], 'overflow_targets': [],
                               'unplaced_targets': [], 'extraction_failed': False,
                               'unplaced_translations': [], 'issues': [], 'warnings': [],
                               'intermediate_diagnostics': []})
        field = 'intermediate_diagnostics' if safe['phase'] == 'intermediate' else 'warnings' if warning else 'issues'
        if safe not in page[field]:
            page[field].append(safe)

    for info in result.get('pages', []):
        number = info.get('source_page')
        if number not in indices:
            continue
        for field in ('issues', 'warnings', 'intermediate_diagnostics'):
            for issue in info.get(field, []):
                add(dict(issue, source_page=number), 'intermediate' if field == 'intermediate_diagnostics' else 'final',
                    warning=field == 'warnings')
        if any(info.get(key) for key in ('missing_targets', 'overflow_targets', 'unplaced_targets', 'extraction_failed', 'unplaced_translations')):
            # Even legacy results without an issue kind must retain their targets.
            pages.setdefault(number, {**indices[number], 'issues': [], 'warnings': [], 'intermediate_diagnostics': []})
        if number in pages:
            page = pages[number]
            page.update(missing_targets=info.get('missing_targets', []),
                        overflow_targets=info.get('overflow_targets', []),
                        unplaced_targets=info.get('unplaced_targets', info.get('overflow_targets', [])),
                        extraction_failed=info.get('extraction_failed', False),
                        unplaced_translations=[{key: block.get(key) for key in ('id', 'bbox', 'translation')}
                                               for block in info.get('unplaced_translations', [])])
    for issue in result.get('failures', []) + result.get('layout_failures', []):
        add(issue)
    for issue in result.get('intermediate_diagnostics', []):
        add(issue, phase='intermediate')
    for issue in result.get('preparation_warnings', []):
        if isinstance(issue, dict):
            add(issue, warning=True)
    for issue in read_cache(work / 'layout-plan.json').get('warnings', []):
        add(issue, warning=True)
    for mode, qa in result.get('product_qa', {}).items():
        for failure in qa.get('failures', []):
            add(failure, output_mode=mode)
    final_qa = result.get('final_qa', {})
    for failure in final_qa.get('failures', []):
        add(failure)
    for page in pages.values():
        page['unplaced_reasons'] = {key: [i for i in page['issues'] if key in i['region_ids']]
                                   for key in page.get('unplaced_targets', [])}
    return {'page_issues': [pages[number] for number in selected if number in pages],
            'document_issues': document_issues}


def publish_copy(source: Path, destination: Path):
    temp = destination.with_name(destination.name + '.tmp')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def chinese_pdf(source: Path, output: Path, expected_pages: int):
    temp = output.with_suffix('.tmp.pdf')
    with fitz.open(source) as bilingual, fitz.open() as chinese:
        if len(bilingual) != expected_pages * 2:
            raise ValueError('Bilingual page count does not match the selected source pages')
        for n in range(expected_pages):
            chinese.insert_pdf(bilingual, from_page=n * 2 + 1, to_page=n * 2 + 1)
        chinese.save(temp, garbage=4, deflate=True)
    temp.replace(output)


def text_exports(work: Path, selected: list[int], artifacts: Path, modes: list[str]):
    from .best_effort import retained_values
    from .extract import restore

    document = Document.load(work / 'document.json')
    # The worker supplies the exact settings used by this run, including its
    # fingerprint. Candidate recovery must never mix an unrelated cache.
    config = _load_config()
    values = retained_values(work, document, config)
    for mode in modes:
        pages = []
        markdown = []
        for n in selected:
            page = document.pages[n - 1]
            blocks = []
            markdown.append(f'## 第 {n} 页\n')
            for region in page.regions:
                text = restore(region, values[region.id]) if region.id in values else None
                block = {'id': region.id, 'role': region.role, 'bbox': region.bbox,
                         'translation': text, 'status': 'translated' if text is not None else 'missing'}
                if mode == 'bilingual':
                    block['source'] = restore(region, region.source)
                    markdown.append(block['source'] + '\n')
                markdown.append((text if text is not None else '[未取得译文]') + '\n')
                blocks.append(block)
            pages.append({'page': n, 'width': page.width, 'height': page.height, 'blocks': blocks})
        write_json(artifacts / f'{mode}.json', {'schema': 'slidetwin.translation.v1',
                   'language': 'zh-CN', 'mode': mode, 'source_sha256': document.source_sha256, 'pages': pages})
        temp = artifacts / f'{mode}.tmp.md'
        temp.write_text('\n'.join(markdown), encoding='utf8')
        temp.replace(artifacts / f'{mode}.md')


def _load_config():
    config = Settings.load(Path(os.environ.get('SLIDETWIN_CONFIG', '/app/config.toml')))
    if value := os.environ.get('SLIDETWIN_API_KEY_FILE'):
        config.provider.api_key_file = value
    # This API promises Chinese outputs, independent of a CLI user's language.
    config.translation.target_language = 'Simplified Chinese'
    return config


def run_job(job: Path, max_pages: int = 500):
    request = read_cache(job / 'job.json')
    selected = []
    warnings = []
    errors = []
    translation_result = {}
    product_qa = {}
    work = job / 'work'
    artifacts = job / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    source = job / 'input.pdf'
    (artifacts / 'report.json').unlink(missing_ok=True)
    update(job, status='running', stage='extracting', started_at=now(), error=None,
           attempts=request.get('attempts', 0) + 1, result=None, finished_at=None,
           artifacts=collect_artifacts(job))

    def progress(message):
        print(message, flush=True)

    def record_error(stage, exc):
        # Tracebacks remain in the private job volume, never in an API payload.
        traceback.print_exception(exc)
        errors.append({'stage': stage, 'type': type(exc).__name__, 'message': f'{stage} failed; inspect server job logs'})

    try:
        with fitz.open(source) as pdf:
            if pdf.needs_pass or len(pdf) < 1 or len(pdf) > max_pages:
                raise ValueError('PDF must be unencrypted and within the configured page limit')
            selected = parse_pages(request.get('pages'), len(pdf))
        update(job, selected_pages=selected, input_sha256=digest(source.read_bytes()))
        extracted = convert_docling(source, work, selected, log=progress)
        warnings.extend(extracted['warnings'])
        if 'docling' in request['outputs']:
            for ext in ('json', 'md'):
                raw = work / f'docling-document.{ext}'
                if raw.is_file():
                    publish_copy(raw, artifacts / f'docling.{ext}')
            update(job, artifacts=collect_artifacts(job))
        modes = [m for m in ('chinese', 'bilingual') if m in request['outputs']]
        if modes:
            update(job, stage='translating')
            config = _load_config()
            output = work / 'bilingual-output.pdf'
            result = run(source, output, work, config, pages=request.get('pages'), preview=False, log=progress)
            translation_result = result
            if result.get('status') != 'automated_checks_passed':
                warnings.append('translation_or_layout_needs_review')
            update(job, stage='exporting')
            # Each product is isolated: a Markdown or Chinese-PDF export failure
            # cannot discard a successfully produced bilingual PDF or Docling tree.
            for mode in modes:
                try:
                    if mode == 'chinese':
                        chinese_pdf(output, artifacts / 'chinese.pdf', len(selected))
                    else:
                        publish_copy(output, artifacts / 'bilingual.pdf')
                except Exception as exc:
                    record_error(mode + '_pdf', exc)
                try:
                    text_exports(work, selected, artifacts, [mode])
                except Exception as exc:
                    record_error(mode + '_text', exc)
            # Recheck the actual downloadable products, after Chinese-page
            # extraction and final PDF serialization. Intermediate warnings
            # are historical evidence, not a verdict on this product's bytes.
            plan = read_cache(work / 'final-layout-plan.json')
            if isinstance(plan.get('placements'), list) and plan.get('output_sha256') == digest(output.read_bytes()):
                placements = [Placement(**entry) for entry in plan['placements']]
                for mode in modes:
                    product = artifacts / f'{mode}.pdf'
                    if not product.is_file():
                        continue
                    try:
                        qa = verify(source, product, selected, placements, work / f'{mode}-final-qa',
                                    bilingual=mode == 'bilingual', phase='final')
                        translation_result.setdefault('product_qa', {})[mode] = qa
                        product_qa[mode] = {key: qa[key] for key in ('passed', 'phase', 'output_mode',
                                                                 'source_sha256', 'output_sha256')}
                        product_qa[mode]['failures'] = [public_issue(i, page_indices(selected, bilingual=mode == 'bilingual'),
                                                                   output_mode=mode) for i in qa['failures']]
                        if not qa['passed']:
                            warnings.append(f'{mode}_final_checks_need_review')
                    except Exception as exc:
                        record_error(mode + '_verification', exc)
            else:
                warnings.append('final_placement_evidence_unavailable')
    except Exception as exc:
        record_error('processing', exc)
    available = collect_artifacts(job)
    available.pop('report.json', None)
    expected = {mode: [f'{mode}.{ext}' for ext in (('json', 'md') if mode == 'docling' else ('pdf', 'json', 'md'))]
                for mode in request['outputs']}
    groups = {mode: 'ready' if all(n in available for n in names) else 'partial' if any(n in available for n in names) else 'failed'
              for mode, names in expected.items()}
    status = ('completed_with_warnings' if errors or warnings or any(s != 'ready' for s in groups.values()) else 'completed') if available else 'failed'
    diagnostics = public_diagnostics(translation_result, selected, work)
    report = {'status': status, 'selected_pages': selected, 'outputs': groups, 'warnings': list(dict.fromkeys(warnings)), 'errors': errors,
              **diagnostics, 'product_qa': product_qa,
              'visual_review': 'not_performed', 'note': 'Automatic completion does not certify visual or semantic perfection.'}
    write_json(artifacts / 'report.json', report)
    update(job, status=status, stage='finished', finished_at=now(), artifacts=collect_artifacts(job),
           result=report, error=errors[-1] if errors else None)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    root = Path(os.environ.get('SLIDETWIN_DATA_DIR', '/data')).resolve()
    run_job(job_path(root, args.job), int(os.environ.get('SLIDETWIN_MAX_PAGES', '500')))


if __name__ == '__main__':
    main()
