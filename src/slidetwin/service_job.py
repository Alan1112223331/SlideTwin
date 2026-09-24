"""Durable job metadata and a deliberately small downloadable artifact surface."""
from datetime import datetime, timezone
from pathlib import Path
import re

from .models import read_cache, write_json, digest

ARTIFACTS = {
    'chinese.pdf': 'application/pdf', 'chinese.json': 'application/json', 'chinese.md': 'text/markdown',
    'bilingual.pdf': 'application/pdf', 'bilingual.json': 'application/json', 'bilingual.md': 'text/markdown',
    'docling.json': 'application/json', 'docling.md': 'text/markdown', 'report.json': 'application/json',
}
OUTPUTS = {'chinese', 'bilingual', 'docling'}
ACTIVE = {'queued', 'running'}


def now():
    return datetime.now(timezone.utc).isoformat()


def job_path(root: Path, job_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('Invalid job ID')
    path = root / 'jobs' / job_id
    if path.is_symlink() or path.resolve().parent != (root / 'jobs').resolve():
        raise ValueError('Invalid job directory')
    return path


def update(job: Path, **changes):
    status = read_cache(job / 'job.json')
    status.update(changes, updated_at=now())
    write_json(job / 'job.json', status)
    return status


def collect_artifacts(job: Path):
    return {name: {'media_type': media_type, 'bytes': path.stat().st_size,
                   'sha256': digest(path.read_bytes())}
            for name, media_type in ARTIFACTS.items()
            if (path := job / 'artifacts' / name).is_file()}
