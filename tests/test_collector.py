import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import httpx
import pytest

from slidetwin import collector


def setup_run(tmp_path, **kwargs):
    collector.write_json(tmp_path/'manifest.json', {'created_at': collector.timestamp(), 'samples': [
        {'slug': 'course-01', 'job_id': 'testjob', 'name': 'lecture.pdf', 'pages': 5}]})
    return argparse.Namespace(run_dir=tmp_path, url='http://127.0.0.1:8000', env_file=None,
                              container=kwargs.get('container'), poll=.03, heartbeat=.01, stale_after=.1)


def job(status='completed', artifacts=None):
    return {'id': 'testjob', 'status': status, 'stage': 'complete' if status != 'running' else 'translating',
            'artifacts': artifacts or {}}


def test_stale_running_json_is_not_reported_as_alive(tmp_path):
    args = setup_run(tmp_path)
    at = datetime.now(timezone.utc)
    collector.write_json(tmp_path/'summary.json', {'status': 'running'})
    collector.write_json(tmp_path/'collector.json', {'state': 'running', 'heartbeat_at': (at-timedelta(seconds=90)).isoformat()})
    status = collector.current_status(tmp_path)
    assert status['monitoring']['collector'] == 'unresponsive' and status['job_status_is_stale']
    assert collector.heartbeat_health({'state': 'finished'}) == 'finished'
    assert collector.heartbeat_health({'state': 'running', 'heartbeat_at': 'not-a-time'}) == 'unknown'
    assert collector.main(['status', '--run-dir', str(args.run_dir)]) == 0


def test_artifact_checksum_failure_retains_previous_complete_file(tmp_path):
    target = tmp_path/'chinese.pdf'
    target.write_bytes(b'previous complete')
    with httpx.Client(base_url='http://127.0.0.1:8000', transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b'wrong payload'))) as client:
        with pytest.raises(ValueError, match='checksum'):
            collector.download_artifact(client, 'testjob', 'chinese.pdf', {'sha256': hashlib.sha256(b'correct').hexdigest()}, target)
    assert target.read_bytes() == b'previous complete'
    assert not (tmp_path/'chinese.pdf.part').exists()


def test_resume_repairs_corrupt_download_and_archives_without_submitting(monkeypatch, tmp_path):
    args = setup_run(tmp_path, container='test-container')
    content = b'correct product'
    checksum = hashlib.sha256(content).hexdigest()
    artifacts = {'chinese.pdf': {'sha256': checksum}}
    target = tmp_path/'jobs'/'course-01'/'artifacts'/'chinese.pdf'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'corrupted after previous collection')
    collector.write_json(tmp_path/'collector.json', {'jobs': {'course-01': {'downloads': {'chinese.pdf': checksum}, 'archive_complete': True}}})
    methods = []
    def handler(request):
        methods.append(request.method)
        if request.url.path.endswith('chinese.pdf'):
            return httpx.Response(200, content=content)
        return httpx.Response(200, json=job(artifacts=artifacts))
    copied = []
    def copy(container, source, destination):
        copied.append(source)
        collector.write_json(destination/'job.json', job(artifacts=artifacts))
        return True
    monkeypatch.setattr(collector, 'docker_copy', copy)
    monitor = collector.Collector(args)
    with httpx.Client(base_url=args.url, transport=httpx.MockTransport(handler)) as client:
        assert monitor.collect_once(client)
        assert monitor.collect_once(client)
    assert target.read_bytes() == content and copied == ['/data/jobs/testjob/.']
    assert set(methods) == {'GET'}
    summary = collector.read_json(tmp_path/'summary.json')
    assert summary['finished_and_archived'] == 1 and summary['status'] == 'completed'


def test_poll_error_does_not_mark_job_finished_and_redacts_service_token(tmp_path):
    args = setup_run(tmp_path)
    token = 'fake-test-secret'
    def handler(request):
        raise httpx.ConnectError(token)
    monitor = collector.Collector(args)
    with httpx.Client(base_url=args.url, transport=httpx.MockTransport(handler)) as client:
        assert not monitor.collect_once(client, token)
    assert token not in (tmp_path/'events.jsonl').read_text(encoding='utf8')
    summary = collector.read_json(tmp_path/'summary.json')
    assert summary['finished_and_archived'] == 0 and summary['status'] == 'running'


def test_failed_server_job_is_archived_and_reported_as_failed(tmp_path):
    args = setup_run(tmp_path)
    monitor = collector.Collector(args)
    with httpx.Client(base_url=args.url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=job('failed')))) as client:
        assert monitor.collect_once(client)
    summary = collector.read_json(tmp_path/'summary.json')
    assert summary['status'] == 'failed' and summary['counts'] == {'failed': 1}
    assert summary['jobs'][0]['archive_scope'] == 'published_artifacts_only'


def test_heartbeat_updates_while_main_collection_is_blocked(tmp_path):
    args = setup_run(tmp_path)
    monitor = collector.Collector(args)
    worker = threading.Thread(target=monitor.heartbeat_loop)
    worker.start()
    try:
        time.sleep(.025)
        first = collector.read_json(tmp_path/'collector.json')['heartbeat_at']
        time.sleep(.025)
        assert collector.read_json(tmp_path/'collector.json')['heartbeat_at'] > first
    finally:
        monitor.stop.set()
        worker.join()


def test_service_log_resume_fills_gap_without_follow_process_or_duplicates(monkeypatch, tmp_path):
    args = setup_run(tmp_path, container='test-container')
    before = b'2026-10-06T14:00:00.000000001Z before restart\n'
    after = b'2026-10-06T14:00:00.000000002Z after restart\n'
    (tmp_path/'service.log').write_bytes(before)
    commands = []
    def snapshot(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, before+after)
    monkeypatch.setattr(collector.subprocess, 'run', snapshot)
    monitor = collector.Collector(args)
    monitor.follow_logs()
    monitor.follow_logs()
    assert (tmp_path/'service.log').read_bytes() == before+after
    assert '--follow' not in commands[0] and '--since' in commands[0]


def test_late_interleaved_logs_and_identical_messages_keep_exact_multiplicity(monkeypatch, tmp_path):
    args = setup_run(tmp_path, container='test-container')
    first = b'2026-10-07T00:00:00.000000003Z first stdout\n'
    late = b'2026-10-07T00:00:00.000000002Z identical stderr\n'
    same_time = b'2026-10-07T00:00:00.000000003Z another stdout\n'
    (tmp_path/'service.log').write_bytes(first)
    snapshot = first+late+late+same_time
    monkeypatch.setattr(collector.subprocess, 'run', lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, snapshot))
    monitor = collector.Collector(args)
    monitor.follow_logs()
    monitor.follow_logs()
    assert (tmp_path/'service.log').read_bytes() == snapshot
    # A third legitimate message with identical timestamp and body still
    # increments the multiplicity, and repeating that snapshot changes nothing.
    snapshot += late
    monitor.follow_logs()
    monitor.follow_logs()
    assert (tmp_path/'service.log').read_bytes() == snapshot


def test_overlap_cache_prevents_dense_large_tail_from_repeating_forever(monkeypatch, tmp_path):
    args = setup_run(tmp_path, container='test-container')
    line = b'2026-10-07T00:00:00.000000001Z identical message\n'
    path = tmp_path/'service.log'
    content = line*8000  # Larger than the bounded startup tail.
    path.write_bytes(content)
    original_open = Path.open
    read_sizes = []
    class Reader:
        def __init__(self, wrapped): self.wrapped = wrapped
        def __enter__(self): return self
        def __exit__(self, *args): return self.wrapped.__exit__(*args)
        def seek(self, *args): return self.wrapped.seek(*args)
        def read(self, size=-1):
            read_sizes.append(size)
            assert 0 < size <= 262144
            return self.wrapped.read(size)
    def open_file(self, *args, **kwargs):
        result = original_open(self, *args, **kwargs)
        return Reader(result) if self == path and args and args[0] == 'rb' else result
    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(collector.subprocess, 'run', lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, content))
    monitor = collector.Collector(args)
    monitor.follow_logs()
    size = path.stat().st_size
    # Existing logs without a sidecar may conservatively duplicate part of
    # their first overlap. Once indexed, repeat snapshots never grow the file.
    monitor.follow_logs()
    monitor.follow_logs()
    assert path.stat().st_size == size and read_sizes == [262144]


def test_supervisor_restarts_exited_collector_without_resubmitting(monkeypatch, tmp_path):
    args = setup_run(tmp_path)
    calls = []
    class Child:
        pid = 123
        returncode = 17
        def __init__(self, returncode):
            self.returncode = returncode
        def poll(self):
            return self.returncode
    def launch(args, mode, detached=False):
        calls.append(mode)
        if len(calls) == 2:
            manifest = collector.read_json(tmp_path/'manifest.json')
            identity = collector.manifest_identity(manifest)
            collector.write_json(tmp_path/'summary.json', {'documents': 1, 'finished_and_archived': 1,
                'status': 'completed', 'manifest_identity': identity})
            collector.write_json(tmp_path/'collector.json', {'pid': 123, 'state': 'finished',
                'started_at': collector.timestamp(), 'manifest_identity': identity, 'session_id': args.session_id})
        return Child(0 if len(calls) == 2 else 17)
    monkeypatch.setattr(collector, 'launch_child', launch)
    monkeypatch.setattr(collector.time, 'sleep', lambda seconds: None)
    collector.supervise(args)
    state = collector.read_json(tmp_path/'supervisor.json')
    assert calls == ['run', 'run'] and state['restarts'] == 1 and state['state'] == 'finished'


@pytest.mark.parametrize('change', ['job_id', 'attempts', 'finished_at'])
def test_old_archive_is_not_reused_for_new_job_or_attempt(monkeypatch, tmp_path, change):
    args = setup_run(tmp_path, container='test-container')
    old = {**job(), 'attempts': 1, 'finished_at': '2026-10-07T00:00:00Z'}
    current = dict(old)
    job_id = 'testjob'
    if change == 'job_id':
        job_id = 'newjob'
        current['id'] = job_id
        manifest = collector.read_json(tmp_path/'manifest.json')
        manifest['samples'][0]['job_id'] = job_id
        collector.write_json(tmp_path/'manifest.json', manifest)
    elif change == 'attempts':
        current['attempts'] = 2
    else:
        current['finished_at'] = '2026-10-07T01:00:00Z'
    target = tmp_path/'jobs'/'course-01'/'server-job'
    collector.write_json(target/'job.json', old)
    (target/'only-old-attempt.log').write_text('old attempt', encoding='utf8')
    collector.write_json(tmp_path/'collector.json', {'jobs': {'course-01': {
        'job_id': 'testjob', 'downloads': {}, 'archive_complete': True,
        'archive_scope': 'full_docker_job', 'archive_identity': collector.archive_identity('testjob', old)}}})
    copied = []
    def copy(container, source, destination):
        copied.append(source)
        collector.write_json(destination/'job.json', current)
        return True
    monkeypatch.setattr(collector, 'docker_copy', copy)
    monitor = collector.Collector(args)
    with httpx.Client(base_url=args.url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=current))) as client:
        assert monitor.collect_once(client)
        assert monitor.collect_once(client)
    assert copied == [f'/data/jobs/{job_id}/.']
    assert collector.read_json(target/'job.json') == current
    assert not (target/'only-old-attempt.log').exists()
    assert list(target.parent.glob('server-job-previous-*/only-old-attempt.log'))


def test_inconsistent_docker_archive_does_not_destroy_previous_or_finish(monkeypatch, tmp_path):
    args = setup_run(tmp_path, container='test-container')
    current = {**job(), 'attempts': 2, 'finished_at': '2026-10-07T00:00:00Z'}
    old = {**current, 'attempts': 1}
    target = tmp_path/'jobs'/'course-01'/'server-job'
    collector.write_json(target/'job.json', old)
    def copy(container, source, destination):
        collector.write_json(destination/'job.json', old)
        return True
    monkeypatch.setattr(collector, 'docker_copy', copy)
    monitor = collector.Collector(args)
    with httpx.Client(base_url=args.url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=current))) as client:
        assert not monitor.collect_once(client)
    assert collector.read_json(target/'job.json') == old
    assert collector.read_json(tmp_path/'summary.json')['finished_and_archived'] == 0


@pytest.mark.parametrize('mismatch', ['exit', 'session_id', 'started_at', 'manifest', 'not_finished'])
def test_previous_summary_cannot_certify_current_child(tmp_path, mismatch):
    setup_run(tmp_path)
    manifest = collector.read_json(tmp_path/'manifest.json')
    identity = collector.manifest_identity(manifest)
    launched_at = '2026-10-07T00:00:00.000+00:00'
    state = {'pid': 123, 'state': 'finished', 'started_at': launched_at,
             'manifest_identity': identity, 'session_id': 'current-launch'}
    class Child:
        pid = 123
        returncode = 0
    child = Child()
    if mismatch == 'exit': child.returncode = 17
    elif mismatch == 'session_id': state['session_id'] = 'old-launch'
    elif mismatch == 'started_at': state['started_at'] = '2026-10-06T00:00:00.000+00:00'
    elif mismatch == 'manifest': state['manifest_identity'] = 'another-manifest'
    else: state['state'] = 'running'
    collector.write_json(tmp_path/'collector.json', state)
    collector.write_json(tmp_path/'summary.json', {'documents': 1, 'finished_and_archived': 1,
        'status': 'completed', 'manifest_identity': identity})
    assert not collector.child_finished(tmp_path, child, launched_at, manifest, 'current-launch')


def test_status_marks_summary_from_another_manifest_stale(tmp_path):
    setup_run(tmp_path)
    collector.write_json(tmp_path/'summary.json', {'documents': 1, 'finished_and_archived': 1,
        'status': 'completed', 'manifest_identity': 'other-jobs'})
    result = collector.current_status(tmp_path)
    assert result['status'] == 'stale' and result['job_status_is_stale']
    assert result['finished_and_archived'] == 0 and result['previous_status'] == 'completed'


def test_launch_session_supports_windows_venv_child_pid(tmp_path):
    setup_run(tmp_path)
    manifest = collector.read_json(tmp_path/'manifest.json')
    identity = collector.manifest_identity(manifest)
    started = collector.timestamp()
    collector.write_json(tmp_path/'collector.json', {'pid': 456, 'state': 'finished',
        'started_at': started, 'manifest_identity': identity, 'session_id': 'this-launch'})
    collector.write_json(tmp_path/'summary.json', {'documents': 1, 'finished_and_archived': 1,
        'status': 'completed', 'manifest_identity': identity})
    child = argparse.Namespace(pid=123, returncode=0)
    assert collector.child_finished(tmp_path, child, started, manifest, 'this-launch')


def test_atomic_state_access_retries_transient_windows_sharing_error(monkeypatch, tmp_path):
    path = tmp_path/'state.json'
    original_replace, original_read = Path.replace, Path.read_text
    calls = {'replace': 0, 'read': 0}
    def replace(self, target):
        calls['replace'] += 1
        if calls['replace'] == 1:
            raise PermissionError('temporary sharing violation')
        return original_replace(self, target)
    def read(self, *args, **kwargs):
        calls['read'] += 1
        if calls['read'] == 1:
            raise PermissionError('temporary sharing violation')
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(collector.time, 'sleep', lambda seconds: None)
    collector.write_json(path, {'state': 'complete'})
    assert collector.read_json(path) == {'state': 'complete'}
    assert calls == {'replace': 2, 'read': 2}


@pytest.mark.parametrize('bad', ['../escape', '..', 'folder/file', 'folder\\file', 'C:escape'])
def test_manifest_rejects_traversal_before_network_or_background_launch(tmp_path, bad):
    setup_run(tmp_path)
    collector.write_json(tmp_path/'manifest.json', {'samples': [{'slug': bad, 'job_id': 'testjob'}]})
    with pytest.raises(ValueError):
        collector.manifest_samples(tmp_path)


def test_detached_background_can_finish_from_local_mock_api(tmp_path, monkeypatch):
    """Real subprocesses, localhost only: no Docker/models/provider credentials."""
    setup_run(tmp_path)
    content = b'local test artifact'
    metadata = {'chinese.pdf': {'sha256': hashlib.sha256(content).hexdigest()}}
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = content if self.path.endswith('/chinese.pdf') else json.dumps(job(artifacts=metadata)).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    env_file = tmp_path/'test.env'
    env_file.write_text('SLIDETWIN_API_TOKEN=local-collector-test\n', encoding='utf8')
    monkeypatch.delenv('SLIDETWIN_API_TOKEN', raising=False)
    try:
        result = subprocess.run([sys.executable, '-m', 'slidetwin.collector', 'start', '--run-dir', str(tmp_path),
                                 '--url', f'http://127.0.0.1:{server.server_port}', '--env-file', str(env_file),
                                 '--poll', '.03', '--heartbeat', '.05', '--stale-after', '.5'],
                                capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr.decode(errors='replace')
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            state = collector.read_json(tmp_path/'supervisor.json', {})
            if state.get('state') == 'finished':
                break
            time.sleep(.05)
        assert state.get('state') == 'finished'
        assert collector.current_status(tmp_path)['status'] == 'completed'
        assert (tmp_path/'jobs'/'course-01'/'artifacts'/'chinese.pdf').read_bytes() == content
        assert all(path.startswith('/v1/jobs/testjob') for path in requests)
    finally:
        server.shutdown()
        server.server_close()
