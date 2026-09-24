"""Single-owner, persistent asynchronous HTTP service for SlideTwin."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from filelock import FileLock, Timeout

from . import __version__
from .models import read_cache, write_json
from .service_job import ACTIVE, ARTIFACTS, OUTPUTS, job_path, now, update


@dataclass
class ServerSettings:
    root: Path
    token: str
    workers: int = 1
    max_pending: int = 16
    max_upload_bytes: int = 100 * 1024 * 1024
    job_timeout: int = 7200

    @classmethod
    def environment(cls):
        token = os.environ.get('SLIDETWIN_API_TOKEN', '')
        if path := os.environ.get('SLIDETWIN_API_TOKEN_FILE'):
            token = Path(path).read_text(encoding='utf-8-sig').strip()
        return cls(Path(os.environ.get('SLIDETWIN_DATA_DIR', '/data')).resolve(), token,
                   int(os.environ.get('SLIDETWIN_JOB_WORKERS', '1')),
                   int(os.environ.get('SLIDETWIN_MAX_PENDING', '16')),
                   int(os.environ.get('SLIDETWIN_MAX_UPLOAD_MB', '100')) * 1024 * 1024,
                   int(os.environ.get('SLIDETWIN_JOB_TIMEOUT', '7200')))

    def validate(self):
        if len(self.token) < 16:
            raise ValueError('Set SLIDETWIN_API_TOKEN (at least 16 characters) before starting the server')
        if min(self.workers, self.max_pending, self.max_upload_bytes, self.job_timeout) < 1:
            raise ValueError('Server limits must be positive')
        if self.workers > self.max_pending:
            raise ValueError('Job workers cannot exceed the pending-job limit')


class BodyTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, 'Request body exceeds upload limit')


class AccessAndSizeMiddleware:
    """Authenticate before multipart parsing; bound streamed as well as sized bodies."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        if scope['path'] not in {'/healthz', '/docs', '/openapi.json', '/redoc', '/docs/oauth2-redirect'}:
            expected = ('Bearer ' + self.settings.token).encode()
            if not hmac.compare_digest(headers.get(b'authorization', b''), expected):
                return await JSONResponse({'detail': 'Bearer token required'}, 401)(scope, receive, send)
        limit = self.settings.max_upload_bytes + 1024 * 1024  # multipart envelope
        try:
            if int(headers.get(b'content-length', b'0')) > limit:
                raise BodyTooLarge()
        except (ValueError, BodyTooLarge):
            return await JSONResponse({'detail': 'Request body exceeds upload limit'}, 413)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > limit:
                raise BodyTooLarge()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except BodyTooLarge:
            await JSONResponse({'detail': 'Request body exceeds upload limit'}, 413)(scope, receive, send)


class JobManager:
    def __init__(self, settings, runner=None):
        self.settings = settings
        self.runner = runner or self.subprocess_job
        self.queue = asyncio.Queue()
        self.pending = set()
        self.tasks = []
        self.uploading = 0
        self.lock = FileLock(settings.root / 'service.lock', timeout=0)

    async def start(self):
        (self.settings.root / 'jobs').mkdir(parents=True, exist_ok=True)
        try:
            self.lock.acquire()
        except Timeout:
            raise RuntimeError('One API process must own each data directory; use one Uvicorn worker') from None
        for path in sorted((self.settings.root / 'jobs').glob('*/job.json')):
            status = read_cache(path)
            if status.get('status') in ACTIVE:
                try:
                    job = job_path(self.settings.root, path.parent.name)
                except ValueError:
                    continue
                update(job, status='queued', stage='queued', recovered_after_restart=True)
                self.enqueue(job.name)
        self.tasks = [asyncio.create_task(self.consume()) for _ in range(self.settings.workers)]

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.lock.release()

    def enqueue(self, job_id):
        self.pending.add(job_id)
        self.queue.put_nowait(job_id)

    async def consume(self):
        while True:
            job_id = await self.queue.get()
            job = job_path(self.settings.root, job_id)
            try:
                await self.runner(job)
                if read_cache(job / 'job.json').get('status') in ACTIVE:
                    update(job, status='failed', stage='finished', error={'type': 'WorkerExited', 'message': 'Worker exited without completing metadata; retry is available'})
            except asyncio.CancelledError:
                if read_cache(job / 'job.json').get('status') in ACTIVE:
                    update(job, status='queued', stage='queued')
                raise
            except Exception as exc:
                update(job, status='failed', stage='finished', error={'type': type(exc).__name__, 'message': 'Worker failed or exceeded its time limit; completed artifacts are retained'})
            finally:
                self.pending.discard(job_id)
                self.queue.task_done()

    async def subprocess_job(self, job):
        env = os.environ.copy()
        env['SLIDETWIN_DATA_DIR'] = str(self.settings.root)
        with (job / 'worker.log').open('ab') as log:
            process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'slidetwin.worker', '--job', job.name,
                                                           env=env, stdout=log, stderr=log)
            try:
                await asyncio.wait_for(process.wait(), self.settings.job_timeout)
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()


def create_app(settings=None, runner=None):
    settings = settings or ServerSettings.environment()
    settings.validate()
    manager = JobManager(settings, runner)

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        yield
        await manager.close()

    app = FastAPI(title='SlideTwin API', version=__version__, lifespan=lifespan,
                  description='Asynchronous PDF translation and original Docling extraction. Bearer authentication required.',
                  license_info={'name': 'AGPL-3.0-only',
                                'url': 'https://github.com/Alan1112223331/SlideTwin/blob/main/LICENSE'},
                  contact={'name': 'SlideTwin source code',
                           'url': 'https://github.com/Alan1112223331/SlideTwin'})
    app.add_middleware(AccessAndSizeMiddleware, settings=settings)
    app.state.manager = manager
    router = APIRouter(prefix='/v1', dependencies=[Depends(HTTPBearer())])

    @app.exception_handler(BodyTooLarge)
    async def body_too_large(request, exc):
        return JSONResponse({'detail': 'Request body exceeds upload limit'}, status_code=413)

    def get_job(job_id):
        try:
            path = job_path(settings.root, job_id)
        except ValueError:
            raise HTTPException(404, 'Job not found') from None
        status = read_cache(path / 'job.json')
        if not status:
            raise HTTPException(404, 'Job not found')
        return path, status

    def response(status):
        result = dict(status)
        result['status_url'] = f'/v1/jobs/{status["id"]}'
        result['artifacts'] = {name: {**meta, 'url': f'/v1/jobs/{status["id"]}/artifacts/{name}'}
                               for name, meta in status.get('artifacts', {}).items() if name in ARTIFACTS}
        return result

    @app.get('/healthz')
    async def health():
        return {'status': 'ok', 'version': __version__}

    @router.post('/jobs', status_code=202)
    async def submit(file: UploadFile = File(...), outputs: str = Form('chinese,bilingual,docling'), pages: str | None = Form(None)):
        modes = list(dict.fromkeys(p.strip() for p in outputs.split(',')))
        if not modes or set(modes) - OUTPUTS:
            raise HTTPException(422, 'outputs must contain chinese, bilingual and/or docling')
        if pages and (len(pages) > 2000 or any(c not in '0123456789,- ' for c in pages)):
            raise HTTPException(422, 'Use page numbers such as 1,3-5')
        if Path(file.filename or '').suffix.lower() != '.pdf':
            raise HTTPException(415, 'Only PDF files are supported')
        if len(manager.pending) + manager.uploading >= settings.max_pending:
            raise HTTPException(429, 'Job queue is full; try again later')
        manager.uploading += 1
        job_id = uuid.uuid4().hex
        job = job_path(settings.root, job_id)
        job.mkdir()
        try:
            size = 0
            header = b''
            with (job / 'input.pdf').open('wb') as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, 'PDF exceeds upload limit')
                    if not header:
                        header = chunk[:1024]
                    target.write(chunk)
            if b'%PDF-' not in header or size == 0:
                raise HTTPException(415, 'File is not a PDF')
            status = {'id': job_id, 'status': 'queued', 'stage': 'queued', 'outputs': modes,
                      'pages': pages, 'created_at': now(), 'updated_at': now(), 'artifacts': {}, 'attempts': 0}
            write_json(job / 'job.json', status)
            manager.enqueue(job_id)
            return response(status)
        except BaseException:
            shutil.rmtree(job)
            raise
        finally:
            manager.uploading -= 1
            await file.close()

    @router.get('/jobs/{job_id}')
    async def status(job_id: str):
        _, data = get_job(job_id)
        return response(data)

    @router.get('/jobs/{job_id}/artifacts/{name}')
    async def download(job_id: str, name: str):
        job, data = get_job(job_id)
        if name not in ARTIFACTS or name not in data.get('artifacts', {}):
            raise HTTPException(404, 'Artifact is not available')
        path = job / 'artifacts' / name
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, 'Artifact is not available')
        return FileResponse(path, media_type=ARTIFACTS[name], filename=name)

    @router.post('/jobs/{job_id}/retry', status_code=202)
    async def retry(job_id: str):
        job, data = get_job(job_id)
        if job_id in manager.pending or data['status'] in ACTIVE:
            raise HTTPException(409, 'Job is already active')
        if data['status'] == 'completed':
            raise HTTPException(409, 'Job already completed successfully')
        if len(manager.pending) + manager.uploading >= settings.max_pending:
            raise HTTPException(429, 'Job queue is full')
        data = update(job, status='queued', stage='queued', error=None)
        manager.enqueue(job_id)
        return response(data)

    @router.delete('/jobs/{job_id}', status_code=204)
    async def delete(job_id: str):
        job, data = get_job(job_id)
        if job_id in manager.pending or data['status'] in ACTIVE:
            raise HTTPException(409, 'Cannot delete an active job')
        shutil.rmtree(job)

    app.include_router(router)
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=int(os.environ.get('PORT', '8000')), workers=1)


if __name__ == '__main__':
    main()
