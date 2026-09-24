FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/Alan1112223331/SlideTwin" \
    org.opencontainers.image.licenses="AGPL-3.0-only" \
    org.opencontainers.image.version="0.4.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/home/slidetwin XDG_CACHE_HOME=/models HF_HOME=/models/huggingface \
    SLIDETWIN_DATA_DIR=/data SLIDETWIN_CONFIG=/app/config.toml \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 libgomp1 fonts-noto-cjk poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 slidetwin \
    && useradd --uid 10001 --gid 10001 --create-home slidetwin \
    && mkdir -p /app /data /models \
    && chown -R slidetwin:slidetwin /app /data /models /home/slidetwin

WORKDIR /app
# CPU wheels avoid bundling CUDA libraries on CPU deployments.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
COPY pyproject.toml ./
COPY requirements-docker.txt ./
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; open('/tmp/requirements.txt','w').write('\n'.join(p['dependencies']+p['optional-dependencies']['server']))" \
    && pip install -c requirements-docker.txt -r /tmp/requirements.txt
COPY src/slidetwin/ ./src/slidetwin/
COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
RUN pip install --no-deps .
COPY docker-fonts.py /tmp/docker-fonts.py
RUN pip install fonttools==4.66.0 && python /tmp/docker-fonts.py \
    && pip uninstall -y fonttools && rm /tmp/docker-fonts.py
# RapidOCR defaults to a package-local download directory. Redirect it to the
# writable, persistent model volume instead of making site-packages writable.
RUN python -c "import pathlib,rapidocr,shutil; p=pathlib.Path(rapidocr.__file__).parent/'models'; d=pathlib.Path('/models/rapidocr'); d.mkdir(exist_ok=True); shutil.copytree(p,d,dirs_exist_ok=True) if p.exists() else None; shutil.rmtree(p) if p.exists() else None; p.symlink_to(d,target_is_directory=True)" \
    && chown -R slidetwin:slidetwin /models
COPY config.example.toml ./config.toml
COPY docker-entrypoint.sh /usr/local/bin/slidetwin-entrypoint
RUN chmod +x /usr/local/bin/slidetwin-entrypoint
USER slidetwin
EXPOSE 8000
VOLUME ["/data", "/models"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
CMD ["slidetwin-server"]
ENTRYPOINT ["slidetwin-entrypoint"]
