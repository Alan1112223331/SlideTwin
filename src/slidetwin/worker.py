"""Run one persisted job in its own process (MuPDF is not shared across threads)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

import pymupdf as fitz

from .config import Settings
from .docling_export import convert_docling
from .models import Document, digest, read_cache, write_json
from .pipeline import parse_pages, run
from .service_job import collect_artifacts, job_path, now, update


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
    except Exception as exc:
        record_error('processing', exc)
    available = collect_artifacts(job)
    available.pop('report.json', None)
    expected = {mode: [f'{mode}.{ext}' for ext in (('json', 'md') if mode == 'docling' else ('pdf', 'json', 'md'))]
                for mode in request['outputs']}
    groups = {mode: 'ready' if all(n in available for n in names) else 'partial' if any(n in available for n in names) else 'failed'
              for mode, names in expected.items()}
    status = ('completed_with_warnings' if errors or warnings or any(s != 'ready' for s in groups.values()) else 'completed') if available else 'failed'
    report = {'status': status, 'selected_pages': selected, 'outputs': groups, 'warnings': warnings, 'errors': errors,
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
