"""Resumable local job collection with independent heartbeats and supervision.

The collector never submits or retries server jobs. It only observes them,
downloads published artifacts, and archives existing Docker job directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
from filelock import FileLock, Timeout as LockTimeout
import httpx


TERMINAL = {'completed', 'completed_with_warnings', 'failed'}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def read_json(path, default=None):
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            if default is not None:
                return default
            raise
        except PermissionError:
            # Windows may momentarily deny a reader during atomic replacement.
            if attempt == 3:
                raise
            time.sleep(.01 * 2**attempt)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Different writer threads/processes must not reuse a temporary filename.
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for attempt in range(4):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.01 * 2**attempt)


def valid_component(value):
    if not isinstance(value, str) or value in {'', '.', '..'} or any(c in value for c in '/\\:'):
        raise ValueError('Unsafe local job or artifact component')
    return value


def manifest_samples(root):
    manifest = read_json(root/'manifest.json')
    samples = manifest.get('samples', [])
    if not samples:
        raise ValueError('manifest.json must contain a nonempty samples array')
    seen = set()
    for sample in samples:
        valid_component(sample['slug'])
        valid_component(sample['job_id'])
        if sample['slug'] in seen:
            raise ValueError('Duplicate sample slug in manifest')
        seen.add(sample['slug'])
    return manifest, samples


def manifest_identity(manifest):
    """Bind persisted completion evidence to the actual requested server jobs."""
    jobs = sorted((item['slug'], item['job_id']) for item in manifest['samples'])
    return hashlib.sha256(json.dumps(jobs, ensure_ascii=False).encode('utf-8')).hexdigest()


def archive_identity(job_id, job):
    return {'job_id': job_id, 'attempts': job.get('attempts'), 'finished_at': job.get('finished_at'),
            'status': job['status'], 'artifacts': {
                name: metadata.get('sha256') for name, metadata in job.get('artifacts', {}).items()}}


def heartbeat_health(state, stale_after=30, at=None):
    if state.get('state') == 'finished':
        return 'finished'
    if state.get('state') in {'stopped', 'failed'}:
        return state['state']
    try:
        beat = datetime.fromisoformat(state['heartbeat_at'])
        if beat.tzinfo is None:
            raise ValueError('Heartbeat requires timezone')
        age = ((at or datetime.now(timezone.utc))-beat).total_seconds()
    except (KeyError, TypeError, ValueError):
        return 'unknown'
    return 'unresponsive' if age > stale_after else 'alive'


def current_status(root, stale_after=30):
    """Compute freshness at read time: old running JSON is never proof of life."""
    summary = read_json(root/'summary.json', {})
    collector = read_json(root/'collector.json', {})
    supervisor = read_json(root/'supervisor.json', {})
    manifest = read_json(root/'manifest.json', {})
    if summary and manifest.get('samples') and summary.get('manifest_identity') != manifest_identity(manifest):
        summary['previous_status'] = summary.get('status')
        summary['status'] = 'stale'
        summary['previous_finished_and_archived'] = summary.get('finished_and_archived')
        summary['finished_and_archived'] = 0
        summary['job_status_is_stale'] = True
    summary['monitoring'] = {
        'collector': heartbeat_health(collector, stale_after),
        'heartbeat_at': collector.get('heartbeat_at'),
        'supervisor': heartbeat_health(supervisor, stale_after),
        'supervisor_heartbeat_at': supervisor.get('heartbeat_at'),
        'last_error': collector.get('last_error'),
        'assessed_at': timestamp(),
    }
    if summary.get('status') == 'running' and summary['monitoring']['collector'] != 'alive':
        summary['job_status_is_stale'] = True
    return summary


def make_summary(manifest, rows, archived):
    counts = dict(Counter(row['status'] for row in rows))
    size = len(manifest['samples'])
    # Failure stays distinguishable from ordinary layout warnings.
    status = 'running'
    if archived == size:
        status = 'failed' if counts.get('failed') else (
            'completed' if counts.get('completed') == size else 'completed_with_warnings')
    return {'status': status, 'created_at': manifest.get('created_at'), 'updated_at': timestamp(),
            'manifest_identity': manifest_identity(manifest),
            'documents': size, 'total_pages': sum(s.get('pages', 0) for s in manifest['samples']),
            'finished_and_archived': archived, 'counts': counts, 'jobs': rows}


def event(root, kind, **details):
    obj = {'time': timestamp(), 'kind': kind, **details}
    with (root/'events.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps(obj, ensure_ascii=False)+'\n')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def download_artifact(client, job_id, name, metadata, target):
    valid_component(name)
    expected = metadata['sha256']
    if target.is_file() and digest(target) == expected:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(name+'.part')
    checksum = hashlib.sha256()
    try:
        with client.stream('GET', f'/v1/jobs/{job_id}/artifacts/{name}') as response:
            response.raise_for_status()
            with temp.open('wb') as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
                    checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError('Downloaded artifact checksum mismatch')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return True


def docker_copy(container, source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(['docker', 'cp', f'{container}:{source}', str(destination)],
                                capture_output=True, timeout=180,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def last_log_time(path, fallback=None):
    """Use actual captured log timestamps to fill gaps after process restarts."""
    if path.is_file():
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size-32768))
            for line in reversed(source.read().decode('utf-8', errors='replace').splitlines()):
                value = line.split(' ', 1)[0]
                try:
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return (dt-timedelta(seconds=1)).isoformat()
                except ValueError:
                    pass
    if fallback:
        try:
            return (datetime.fromisoformat(fallback)-timedelta(minutes=1)).isoformat()
        except ValueError:
            pass
    return '24h'


def log_order(value):
    """Compare Docker UTC timestamps without losing their nanosecond suffix."""
    normalized = value.removesuffix('Z')
    base, _, fraction = normalized.partition('.')
    datetime.fromisoformat(base)  # Reject ordinary stderr lines.
    return base, (fraction+'000000000')[:9]


def captured_log_cursor(path):
    if path.is_file():
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size-32768))
            for line in reversed(source.read().decode('utf-8', errors='replace').splitlines()):
                value = line.split(' ', 1)[0]
                try:
                    log_order(value)
                    return value
                except ValueError:
                    pass
    return None


def log_record(line):
    value = line.split(b' ', 1)[0].decode('ascii')
    log_order(value)
    return value, hashlib.sha256(line).hexdigest()


def captured_log_overlap(path, cache_path, fallback=None, tail_bytes=262144):
    """Bound startup reads; subsequently retain exact overlap multiplicities.

    The sidecar holds only a one-second window of timestamp+full-line hashes,
    never scans a large service.log, and is invalidated after external edits.
    """
    stat = path.stat() if path.is_file() else None
    size, modified = (stat.st_size, stat.st_mtime_ns) if stat else (0, None)
    cache = read_json(cache_path, {})
    records, latest = {}, None
    if cache.get('log_size') == size and cache.get('log_mtime_ns') == modified:
        latest = cache.get('latest_timestamp')
        records = {key: dict(value) for key, value in cache.get('records', {}).items()}
    elif size:
        with path.open('rb') as source:
            start = max(0, size-tail_bytes)
            source.seek(start)
            data = source.read(tail_bytes)
        if start:
            data = data.partition(b'\n')[2]  # Drop a possibly partial first line.
        for line in data.splitlines(keepends=True):
            try:
                value, key = log_record(line)
            except (ValueError, UnicodeError):
                continue
            record = records.setdefault(key, {'timestamp': value, 'count': 0})
            record['count'] += 1
            if latest is None or log_order(value) > log_order(latest):
                latest = value
    if latest is None:
        return last_log_time(path, fallback), records, latest
    since = datetime.fromisoformat(latest.replace('Z', '+00:00'))-timedelta(seconds=1)
    records = {key: value for key, value in records.items()
               if datetime.fromisoformat(value['timestamp'].replace('Z', '+00:00')) >= since}
    return since.isoformat(), records, latest


def connection(args):
    env = dotenv_values(args.env_file) if args.env_file else {}
    token = os.environ.get('SLIDETWIN_API_TOKEN') or env.get('SLIDETWIN_API_TOKEN')
    if not token:
        raise ValueError('Set SLIDETWIN_API_TOKEN in the environment or --env-file')
    url = args.url or f"http://127.0.0.1:{env.get('SLIDETWIN_PORT') or '8000'}"
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError('Service URL cannot contain credentials, query or fragment')
    if parsed.scheme == 'http' and parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Remote services require HTTPS')
    return httpx.Client(base_url=url.rstrip('/'), headers={'Authorization': 'Bearer '+token}, timeout=120), token


class Collector:
    def __init__(self, args):
        self.args = args
        self.root = args.run_dir.resolve()
        self.manifest, self.samples = manifest_samples(self.root)
        self.path = self.root/'collector.json'
        self.state = read_json(self.path, {'jobs': {}})
        self.state.update(pid=os.getpid(), state='running', started_at=timestamp(),
                          manifest_identity=manifest_identity(self.manifest),
                          session_id=getattr(args, 'session_id', None) or uuid4().hex)
        self.state.setdefault('jobs', {})
        for item in self.samples:
            entry = self.state['jobs'].get(item['slug'], {})
            if entry.get('job_id') != item['job_id']:
                entry = {'downloads': {}, 'archive_complete': False}
            entry['job_id'] = item['job_id']
            self.state['jobs'][item['slug']] = entry
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.log_error = None
        self.previous = {}

    def save_heartbeat(self):
        with self.lock:
            self.state['heartbeat_at'] = timestamp()
            write_json(self.path, self.state)

    def heartbeat_loop(self):
        while not self.stop.wait(self.args.heartbeat):
            self.save_heartbeat()

    def follow_logs(self):
        if not self.args.container:
            return
        log_path = self.root/'service.log'
        try:
            cache_path = self.root/'service-log-overlap.json'
            since, records, latest = captured_log_overlap(log_path, cache_path, self.manifest.get('created_at'))
            remaining = {key: value['count'] for key, value in records.items()}
            # A bounded snapshot has no long-lived `docker logs --follow` child
            # to orphan if the collector is killed. The next poll fills gaps.
            result = subprocess.run(['docker', 'logs', '--timestamps', '--since', since, self.args.container],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode:
                raise RuntimeError('Docker log snapshot unavailable')
            lines = []
            for line in result.stdout.splitlines(keepends=True):
                try:
                    value, key = log_record(line)
                except (ValueError, UnicodeError):
                    continue
                if remaining.get(key, 0):
                    remaining[key] -= 1
                else:
                    lines.append(line)
                    record = records.setdefault(key, {'timestamp': value, 'count': 0})
                    record['count'] += 1
                if latest is None or log_order(value) > log_order(latest):
                    latest = value
            if lines:
                with log_path.open('ab') as output:
                    output.writelines(lines)
            if latest is not None:
                cutoff = datetime.fromisoformat(latest.replace('Z', '+00:00'))-timedelta(seconds=1)
                records = {key: value for key, value in records.items()
                           if datetime.fromisoformat(value['timestamp'].replace('Z', '+00:00')) >= cutoff}
            saved = log_path.stat() if log_path.is_file() else None
            write_json(cache_path, {'log_size': saved.st_size if saved else 0,
                                   'log_mtime_ns': saved.st_mtime_ns if saved else None,
                                   'latest_timestamp': latest, 'records': records})
            self.log_error = None
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            message = type(exc).__name__
            if self.log_error != message:
                event(self.root, 'service_log_unavailable', error_kind=message)
                self.log_error = message

    def collect_once(self, client, token='', copy_logs=False):
        rows, archived = [], 0
        for item in self.samples:
            slug, job_id = item['slug'], item['job_id']
            directory = self.root/'jobs'/slug
            with self.lock:
                entry = copy.deepcopy(self.state['jobs'][slug])
            try:
                response = client.get('/v1/jobs/'+job_id)
                response.raise_for_status()
                job = response.json()
                if job.get('id', job_id) != job_id:
                    raise ValueError('Service returned a different job identity')
                write_json(directory/'status.json', job)
                progress = (job['status'], job.get('stage'), len(job.get('artifacts', {})))
                if self.previous.get(slug) != progress:
                    event(self.root, 'progress', slug=slug, job_id=job_id, status=progress[0], stage=progress[1], artifacts=progress[2])
                    self.previous[slug] = progress
                entry['downloads'] = {}
                for name, metadata in job.get('artifacts', {}).items():
                    if download_artifact(client, job_id, name, metadata, directory/'artifacts'/valid_component(name)):
                        event(self.root, 'artifact_saved', slug=slug, artifact=name)
                    entry['downloads'][name] = metadata['sha256']
                identity = archive_identity(job_id, job)
                scope = 'full_docker_job' if self.args.container else 'published_artifacts_only'
                archive_valid = entry.get('archive_identity') == identity and entry.get('archive_scope') == scope
                if archive_valid and self.args.container:
                    saved = read_json(directory/'server-job'/'job.json', {})
                    archive_valid = saved.get('id') == job_id and archive_identity(job_id, saved) == identity
                entry['archive_complete'] = bool(entry.get('archive_complete') and archive_valid)
                if copy_logs and self.args.container and job['status'] == 'running':
                    docker_copy(self.args.container, f'/data/jobs/{job_id}/worker.log', directory/'logs'/'worker.log')
                    for filename in ('request-timings.jsonl', 'translation-events.json', 'run.json'):
                        docker_copy(self.args.container, f'/data/jobs/{job_id}/work/{filename}', directory/'logs'/filename)
                if job['status'] in TERMINAL and not entry['archive_complete']:
                    if self.args.container:
                        target = directory/'server-job'
                        archive_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:20]
                        temp = directory/('.server-job.'+archive_key+'.part')
                        # Reuse the same attempt's partial directory across
                        # transient copy failures instead of growing one per poll.
                        temp.mkdir(exist_ok=True)
                        copied = docker_copy(self.args.container, f'/data/jobs/{job_id}/.', temp)
                        saved = read_json(temp/'job.json', {}) if copied else {}
                        entry['archive_complete'] = (saved.get('id') == job_id and
                            archive_identity(job_id, saved) == identity) if saved else False
                        if entry['archive_complete']:
                            previous = None
                            if target.exists():
                                previous = directory/('server-job-previous-'+uuid4().hex)
                                target.rename(previous)
                            try:
                                temp.rename(target)
                            except OSError:
                                if previous is not None and not target.exists():
                                    previous.rename(target)
                                raise
                    else:
                        # Remote API exposes artifacts, not private worker/checkpoint logs.
                        entry['archive_complete'] = True
                    if entry['archive_complete']:
                        entry.update(archive_identity=identity, archive_scope=scope)
                        event(self.root, 'server_job_archived', slug=slug, scope=scope)
                entry.update(status=job['status'], stage=job.get('stage'), last_successful_poll=timestamp())
                entry.pop('last_error', None)
                if job['status'] in TERMINAL and entry['archive_complete']:
                    archived += 1
                rows.append({'slug': slug, 'name': item.get('name', slug), 'pages': item.get('pages'), 'job_id': job_id,
                             'status': job['status'], 'stage': job.get('stage'), 'artifacts': list(entry['downloads']),
                             'logs_archived': entry['archive_complete'], 'archive_scope': entry.get('archive_scope', 'full_docker_job')})
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                if token:
                    message = message.replace(token, '[REDACTED]')
                if entry.get('last_error') != message:
                    event(self.root, 'collector_error', slug=slug, error=message)
                entry['last_error'] = message
                rows.append({'slug': slug, 'job_id': job_id, 'status': entry.get('status', 'unreachable'),
                             'collector_error': message, 'last_successful_poll': entry.get('last_successful_poll')})
            with self.lock:
                self.state['jobs'][slug] = entry
        summary = make_summary(self.manifest, rows, archived)
        summary['monitoring'] = {'collector': 'alive', 'heartbeat_file': 'collector.json',
                                 'heartbeat_at': timestamp(),
                                 'stale_after_seconds': self.args.stale_after,
                                 'status_command': 'python scripts/collect_jobs.py status --run-dir <this-directory>'}
        write_json(self.root/'summary.json', summary)
        self.save_heartbeat()
        return archived == len(self.samples)

    def run(self):
        finished = False
        self.save_heartbeat()
        heart = threading.Thread(target=self.heartbeat_loop, daemon=True)
        heart.start()
        event(self.root, 'collector_started', pid=os.getpid())
        next_logs = 0
        try:
            with connection(self.args)[0] as client:
                token = client.headers['Authorization'].removeprefix('Bearer ')
                while True:
                    self.follow_logs()
                    copy_logs = time.monotonic() >= next_logs
                    if copy_logs:
                        next_logs = time.monotonic()+30
                    if self.collect_once(client, token, copy_logs):
                        finished = True
                        event(self.root, 'all_jobs_finished')
                        break
                    time.sleep(self.args.poll)
        finally:
            self.stop.set()
            heart.join(timeout=self.args.heartbeat+1)
            with self.lock:
                self.state.update(state='finished' if finished else 'stopped', stopped_at=timestamp())
            self.save_heartbeat()


def child_arguments(args, mode):
    result = [sys.executable, '-u', '-m', 'slidetwin.collector', mode, '--run-dir', str(args.run_dir.resolve()),
              '--poll', str(args.poll), '--heartbeat', str(args.heartbeat), '--stale-after', str(args.stale_after)]
    if mode == 'run' and getattr(args, 'session_id', None):
        result += ['--session-id', args.session_id]
    for key in ('url', 'env_file', 'container'):
        value = getattr(args, key)
        if value:
            result += ['--'+key.replace('_', '-'), str(Path(value).resolve()) if key == 'env_file' else value]
    return result


def launch_child(args, mode, detached=False):
    root = args.run_dir.resolve()
    prefix = 'supervisor' if mode == 'supervise' else 'collector'
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' and detached else (
        subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    with (root/(prefix+'.stdout.log')).open('ab') as stdout, (root/(prefix+'.stderr.log')).open('ab') as stderr:
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        return subprocess.Popen(child_arguments(args, mode), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                creationflags=flags, start_new_session=detached and os.name != 'nt', close_fds=True, env=environment)


def child_finished(root, child, launched_at, manifest, session_id):
    """Historical summary files cannot certify a failed or unrelated child."""
    if child.returncode != 0:
        return False
    state = read_json(root/'collector.json', {})
    summary = read_json(root/'summary.json', {})
    identity = manifest_identity(manifest)
    # Windows venv executables may launch a second Python process, so the
    # Popen PID is not necessarily the collector PID. A fresh launch nonce
    # binds evidence to this child across that launcher and PID reuse.
    return (state.get('session_id') == session_id and state.get('state') == 'finished' and
            state.get('started_at', '') >= launched_at and
            state.get('manifest_identity') == identity == summary.get('manifest_identity') and
            summary.get('documents') == len(manifest['samples']) == summary.get('finished_and_archived'))


def supervise(args):
    root = args.run_dir.resolve()
    with FileLock(root/'.supervisor.lock', timeout=0):
        state = {'pid': os.getpid(), 'state': 'running', 'started_at': timestamp(), 'restarts': 0}
        child = None
        launched_at = None
        try:
            while True:
                if child is None or child.poll() is not None:
                    if child is not None:
                        state['last_exit_code'] = child.returncode
                        manifest, _ = manifest_samples(root)
                        if child_finished(root, child, launched_at, manifest, args.session_id):
                            state['state'] = 'finished'
                            break
                        state['restarts'] += 1
                        event(root, 'collector_restarting', exit_code=child.returncode)
                    launched_at = timestamp()
                    args.session_id = uuid4().hex
                    child = launch_child(args, 'run')
                    state['collector_pid'] = child.pid
                state['heartbeat_at'] = timestamp()
                write_json(root/'supervisor.json', state)
                # Persist an explicitly fresh assessment in a separate file;
                # status always recalculates it if both processes have died.
                write_json(root/'health.json', current_status(root, args.stale_after)['monitoring'])
                time.sleep(args.heartbeat)
        finally:
            state.setdefault('state', 'stopped')
            if state['state'] != 'finished':
                state['state'] = 'stopped'
            state['heartbeat_at'] = timestamp()
            write_json(root/'supervisor.json', state)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'run', 'supervise', 'status'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--url')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--container', help='Docker container to archive private job logs; omit for remote artifacts only')
    parser.add_argument('--session-id', help=argparse.SUPPRESS)
    parser.add_argument('--poll', type=float, default=10)
    parser.add_argument('--heartbeat', type=float, default=5)
    parser.add_argument('--stale-after', type=float, default=30)
    args = parser.parse_args(argv)
    if args.poll <= 0 or args.heartbeat <= 0 or args.stale_after <= args.heartbeat:
        parser.error('Positive poll/heartbeat required; stale-after must exceed heartbeat')
    manifest_samples(args.run_dir.resolve())
    if args.mode == 'status':
        print(json.dumps(current_status(args.run_dir.resolve(), args.stale_after), ensure_ascii=False, indent=2))
        return 0
    if args.mode == 'start':
        # Fail fast on bad connection configuration without exposing credentials.
        client, _ = connection(args)
        client.close()
        try:
            with FileLock(args.run_dir/'.supervisor.lock', timeout=0):
                pass
        except LockTimeout:
            print('Collector supervisor already running; use status.')
            return 0
        child = launch_child(args, 'supervise', detached=True)
        print(f'Background supervisor started: PID {child.pid}; {args.run_dir.resolve()}')
        return 0
    try:
        if args.mode == 'supervise':
            supervise(args)
        else:
            with FileLock(args.run_dir/'.collector.lock', timeout=0):
                Collector(args).run()
    except LockTimeout:
        print('A collector for this run is already active.', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
