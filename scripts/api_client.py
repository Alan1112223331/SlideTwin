"""Upload a PDF, wait for its asynchronous job, and download published artifacts."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--outputs', default='chinese,bilingual,docling')
    parser.add_argument('--pages')
    parser.add_argument('--out', type=Path, default=Path('output/api-result'))
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get('SLIDETWIN_API_TOKEN')
    if not token:
        parser.error('Set SLIDETWIN_API_TOKEN in your environment or local .env')
    with httpx.Client(base_url=args.url.rstrip('/'), headers={'Authorization': 'Bearer ' + token}, timeout=120) as client:
        with args.source.open('rb') as source:
            data = {'outputs': args.outputs}
            if args.pages:data['pages'] = args.pages
            response = client.post('/v1/jobs', files={'file': (args.source.name, source, 'application/pdf')}, data=data)
        response.raise_for_status()
        job = response.json()
        print('Job:', job['id'], flush=True)
        deadline = time.monotonic() + args.timeout
        previous = None
        while job['status'] in {'queued', 'running'}:
            progress = (job['status'], job['stage'])
            if progress != previous:
                print(*progress, flush=True)
                previous = progress
            if time.monotonic() > deadline:
                raise TimeoutError('Client wait expired; server job continues. Query /v1/jobs/' + job['id'])
            time.sleep(2)
            response = client.get('/v1/jobs/' + job['id']);response.raise_for_status();job = response.json()
        args.out.mkdir(parents=True, exist_ok=True)
        for name, metadata in job.get('artifacts', {}).items():
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Unsafe artifact filename')
            response = client.get(f'/v1/jobs/{job["id"]}/artifacts/{name}');response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != metadata['sha256']:
                raise ValueError('Artifact checksum mismatch')
            (args.out / name).write_bytes(response.content)
        print(job['status'], args.out.resolve())
        if job['status'] != 'completed':
            print('Inspect report.json for incomplete products or warnings.')
        return 2 if job['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
