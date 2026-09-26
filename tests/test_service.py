import asyncio
import json
from pathlib import Path
import time

import pymupdf as fitz
import pytest
pytest.importorskip('fastapi', reason='Install .[server,dev] to run API tests')
from fastapi.testclient import TestClient

from slidetwin.api import JobManager, ServerSettings, create_app
from slidetwin.models import write_json, read_cache, Document, Page, Region, digest
from slidetwin.service_job import collect_artifacts, job_path, update
from slidetwin.worker import chinese_pdf, run_job


TOKEN = 'synthetic-http-test-token-only'
HEADERS = {'Authorization': 'Bearer ' + TOKEN}


def pdf_bytes(pages=2):
    with fitz.open() as pdf:
        for n in range(pages):
            pdf.new_page().insert_text((30, 60), f'Synthetic page {n + 1}')
        return pdf.tobytes()


def wait_done(client, job_id):
    for _ in range(100):
        status = client.get(f'/v1/jobs/{job_id}', headers=HEADERS).json()
        if status['status'] not in {'queued', 'running'}:
            return status
        time.sleep(.01)
    raise AssertionError('Job did not finish')


async def success(job):
    update(job, status='running')
    (job / 'artifacts').mkdir(exist_ok=True)
    write_json(job / 'artifacts/docling.json', {'texts': []})
    update(job, status='completed', artifacts=collect_artifacts(job))


def test_api_upload_status_download_auth_and_delete(tmp_path):
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN), success)) as client:
        assert client.get('/healthz').status_code == 200
        assert client.get('/docs').status_code == 200
        schema = client.get('/openapi.json').json()
        assert schema['paths']['/v1/jobs']['post']['security']
        assert client.post('/v1/jobs').status_code == 401
        response = client.post('/v1/jobs', headers=HEADERS, files={'file': ('../../private.pdf', pdf_bytes())}, data={'outputs': 'docling'})
        assert response.status_code == 202
        job_id = response.json()['id']
        result = wait_done(client, job_id)
        assert result['status'] == 'completed'
        url = result['artifacts']['docling.json']['url']
        assert client.get(url).status_code == 401
        assert client.get(url, headers=HEADERS).json() == {'texts': []}
        assert client.get(f'/v1/jobs/{job_id}/artifacts/worker.log', headers=HEADERS).status_code == 404
        assert client.post(f'/v1/jobs/{job_id}/retry', headers=HEADERS).status_code == 409
        assert client.delete(f'/v1/jobs/{job_id}', headers=HEADERS).status_code == 204
        assert not job_path(tmp_path, job_id).exists()
        assert client.get(f'/v1/jobs/{job_id}', headers=HEADERS).status_code == 404


@pytest.mark.parametrize('filename,content,data,code', [
    ('slide.txt', b'%PDF-', {}, 415), ('slide.pdf', b'not a pdf', {}, 415),
    ('slide.pdf', b'%PDF-', {'outputs': 'unknown'}, 422),
    ('slide.pdf', b'%PDF-', {'pages': '../secret'}, 422),
    ('slide.pdf', b'%PDF-' + b'0' * 1000, {}, 413),
])
def test_invalid_upload_is_rejected_without_leaving_jobs(tmp_path, filename, content, data, code):
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN, max_upload_bytes=100), success)) as client:
        assert client.post('/v1/jobs', headers=HEADERS, files={'file': (filename, content)}, data=data).status_code == code
    assert not list((tmp_path / 'jobs').iterdir())


def test_streamed_body_limit_applies_without_content_length(tmp_path):
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN, max_upload_bytes=10), success)) as client:
        chunks = iter([b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n', b'%PDF-' + b'x' * (2 * 1024 * 1024), b'\r\n--x--'])
        result = client.post('/v1/jobs', headers={**HEADERS, 'Content-Type': 'multipart/form-data; boundary=x'}, content=chunks)
        assert result.status_code == 413


def test_failed_worker_can_retry_and_restart_recovers_queue(tmp_path):
    async def fail(job):
        update(job, status='failed', error={'message': 'synthetic failure'})
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN), fail)) as client:
        job_id = client.post('/v1/jobs', headers=HEADERS, files={'file': ('x.pdf', pdf_bytes())}).json()['id']
        assert wait_done(client, job_id)['status'] == 'failed'
    update(job_path(tmp_path, job_id), status='running')
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN), success)) as client:
        result = wait_done(client, job_id)
        assert result['status'] == 'completed' and result['recovered_after_restart']


def test_queue_limit_and_active_delete(tmp_path):
    async def slow(job):
        await asyncio.sleep(100)
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN, max_pending=1), slow)) as client:
        job_id = client.post('/v1/jobs', headers=HEADERS, files={'file': ('x.pdf', pdf_bytes())}).json()['id']
        assert client.post('/v1/jobs', headers=HEADERS, files={'file': ('x.pdf', pdf_bytes())}).status_code == 429
        assert client.delete(f'/v1/jobs/{job_id}', headers=HEADERS).status_code == 409
    assert read_cache(job_path(tmp_path, job_id) / 'job.json')['status'] == 'queued'


def test_finished_job_frees_worker_without_waiting_for_other_job(tmp_path):
    async def scenario():
        blocker = asyncio.Event()
        started = asyncio.Event()
        completed = []
        ids = ['a' * 32, 'b' * 32, 'c' * 32]
        async def runner(job):
            if job.name == ids[0]:
                started.set()
                await blocker.wait()
            completed.append(job.name)
            update(job, status='completed')
        manager = JobManager(ServerSettings(tmp_path, TOKEN, workers=2), runner)
        await manager.start()
        try:
            for job_id in ids:
                job = job_path(tmp_path, job_id);job.mkdir()
                write_json(job / 'job.json', {'id': job_id, 'status': 'queued'})
                manager.enqueue(job_id)
            await started.wait()
            for _ in range(50):
                if ids[2] in completed:break
                await asyncio.sleep(.01)
            assert completed == ids[1:]
        finally:
            blocker.set()
            await manager.close()
    asyncio.run(scenario())


def test_chinese_pdf_exactly_reuses_translated_pages(tmp_path):
    original = tmp_path / 'bilingual.pdf';original.write_bytes(pdf_bytes(4))
    target = tmp_path / 'chinese.pdf'
    chinese_pdf(original, target, 2)
    with fitz.open(original) as both, fitz.open(target) as cn:
        assert len(cn) == 2
        for n in range(2):
            assert both[n * 2 + 1].get_pixmap().samples == cn[n].get_pixmap().samples
    with pytest.raises(ValueError):chinese_pdf(original, target, 3)


def make_job(tmp_path, outputs):
    job = tmp_path / 'job';job.mkdir()
    (job / 'input.pdf').write_bytes(pdf_bytes())
    write_json(job / 'job.json', {'id': 'a' * 32, 'status': 'queued', 'outputs': outputs, 'pages': None})
    return job


def extraction(source, work, selected, log):
    write_json(work / 'docling-document.json', {'schema_name': 'DoclingDocument', 'texts': [{'text': 'Original English'}]})
    (work / 'docling-document.md').write_text('Original English', encoding='utf8')
    return {'warnings': []}


def test_docling_only_requires_no_translation_config_or_key(tmp_path, monkeypatch):
    import slidetwin.worker as worker
    job = make_job(tmp_path, ['docling'])
    monkeypatch.setattr(worker, 'convert_docling', extraction)
    monkeypatch.setattr(worker, '_load_config', lambda: pytest.fail('Docling-only must not load translation credentials'))
    result = run_job(job)
    assert result['status'] == 'completed'
    assert (job / 'artifacts/docling.json').read_bytes() == (job / 'work/docling-document.json').read_bytes()


def test_model_failure_keeps_original_docling_available(tmp_path, monkeypatch):
    import slidetwin.worker as worker
    job = make_job(tmp_path, ['chinese', 'bilingual', 'docling'])
    monkeypatch.setattr(worker, 'convert_docling', extraction)
    monkeypatch.setattr(worker, '_load_config', lambda: (_ for _ in ()).throw(RuntimeError('private internal details')))
    result = run_job(job)
    assert result['status'] == 'completed_with_warnings'
    assert result['outputs'] == {'chinese': 'failed', 'bilingual': 'failed', 'docling': 'ready'}
    assert 'private internal details' not in json.dumps(result)
    assert 'docling.json' in read_cache(job / 'job.json')['artifacts']


def test_old_error_report_does_not_turn_failed_retry_into_success(tmp_path, monkeypatch):
    import slidetwin.worker as worker
    job = make_job(tmp_path, ['chinese'])
    (job / 'artifacts').mkdir();write_json(job / 'artifacts/report.json', {'status': 'failed'})
    monkeypatch.setattr(worker, 'convert_docling', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('failure')))
    assert run_job(job)['status'] == 'failed'


def test_retry_endpoint_runs_failed_job_again(tmp_path):
    attempts = []
    async def runner(job):
        attempts.append(job.name)
        if len(attempts) == 1:
            update(job, status='failed')
        else:
            await success(job)
    with TestClient(create_app(ServerSettings(tmp_path, TOKEN), runner)) as client:
        job_id = client.post('/v1/jobs', headers=HEADERS, files={'file': ('x.pdf', pdf_bytes())}).json()['id']
        assert wait_done(client, job_id)['status'] == 'failed'
        assert client.post(f'/v1/jobs/{job_id}/retry', headers=HEADERS).status_code == 202
        assert wait_done(client, job_id)['status'] == 'completed'
        assert attempts == [job_id, job_id]


def test_text_export_uses_retained_translations_and_preserves_source_pair(tmp_path, monkeypatch):
    from slidetwin.config import Settings
    import slidetwin.worker as worker
    config = Settings();work = tmp_path / 'work';work.mkdir()
    out = tmp_path / 'artifacts';out.mkdir()
    document = Document('source-hash', [Page(1, 720, 405, [
        Region('a', 1, 'Voltage ⟦P000⟧', [10, 10, 200, 40], protected={'⟦P000⟧': '6'}),
        Region('b', 1, 'Unavailable text', [10, 50, 200, 80]),
    ])])
    document.save(work / 'document.json')
    write_json(work / 'translation-ledger.json', {'source_sha256': 'source-hash', 'config_fingerprint': config.fingerprint(),
                                                'translations': {'a': '电压 ⟦P000⟧'}})
    monkeypatch.setattr(worker, '_load_config', lambda: config)
    worker.text_exports(work, [1], out, ['chinese', 'bilingual'])
    cn = read_cache(out / 'chinese.json')['pages'][0]['blocks']
    both = read_cache(out / 'bilingual.json')['pages'][0]['blocks']
    assert cn[0]['translation'] == '电压 6' and 'source' not in cn[0]
    assert cn[1]['translation'] is None and cn[1]['status'] == 'missing'
    assert both[0]['source'] == 'Voltage 6' and both[0]['translation'] == '电压 6'
    assert '未取得译文' in (out / 'chinese.md').read_text(encoding='utf8')


def test_unplaced_text_is_available_in_api_report_without_internal_paths(tmp_path, monkeypatch):
    from slidetwin.config import Settings
    import slidetwin.worker as worker
    job=make_job(tmp_path,['chinese','bilingual']);config=Settings()
    text='模型返回的完整译文，保留在独立报告中。'
    def translated(source,output,work,*args,**kwargs):
        document=Document('source-hash',[
            Page(1,612,792,[Region('a',1,'Source text',[30,40,100,60])]),
            Page(2,612,792,[]),
        ])
        document.save(work/'document.json')
        write_json(work/'translation-ledger.json',{'source_sha256':'source-hash',
                   'config_fingerprint':config.fingerprint(),'translations':{'a':text}})
        output.write_bytes(pdf_bytes(4))
        return {'status':'completed_with_warnings','previous_output_backup':'/private/internal.pdf',
                'pages':[{'source_page':1,'missing_targets':[],'overflow_targets':['a'],
                          'issues':[{'reason':'/private/trace.log'}],
                          'unplaced_translations':[{'id':'a','bbox':[30,40,100,60],
                                                   'translation':text,'private_path':'/private/debug.json'}]}]}
    monkeypatch.setattr(worker,'convert_docling',extraction)
    monkeypatch.setattr(worker,'_load_config',lambda:config)
    monkeypatch.setattr(worker,'run',translated)
    result=run_job(job)
    assert result['status']=='completed_with_warnings'
    assert result['outputs']=={'chinese':'ready','bilingual':'ready'}
    assert result['page_issues']==[{'source_page':1,'missing_targets':[],'overflow_targets':['a'],
                                  'extraction_failed':False,'unplaced_translations':[
                                      {'id':'a','bbox':[30,40,100,60],'translation':text}]}]
    report=read_cache(job/'artifacts/report.json')
    assert report==result and '/private/' not in json.dumps(report)
    assert read_cache(job/'artifacts/chinese.json')['pages'][0]['blocks'][0]['translation']==text
    assert text in (job/'artifacts/bilingual.md').read_text(encoding='utf8')
