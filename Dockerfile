# syntax=docker/dockerfile:1
#
# One image for the API, the index builder and the eval runner.
#
#   TORCH_VARIANT=cpu    (default) runs anywhere, incl. Apple Silicon (arm64)
#   TORCH_VARIANT=cu130  NVIDIA GPU; needs the NVIDIA Container Toolkit on the
#                        host (see docker-compose.gpu.yml)
#
# torch ships its CUDA runtime inside the wheel, so no CUDA base image is
# needed — only the host's NVIDIA driver.

FROM python:3.12-slim-bookworm

ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.13.0
ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    # Inside the container the API must listen on all interfaces; the
    # compose file publishes it on the host's 127.0.0.1 only.
    AGENTIC_RAG__SERVING__HOST=0.0.0.0

RUN groupadd --gid "${GID}" app \
 && useradd --uid "${UID}" --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first so code changes don't invalidate this (large) layer.
COPY requirements-core.txt requirements-ml.txt ./
RUN pip install "torch==${TORCH_VERSION}+${TORCH_VARIANT}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
 && pip install -r requirements-core.txt -r requirements-ml.txt

COPY --chown=app:app . .

# Mount points for named volumes. Creating them here, owned by `app`, makes
# Docker initialise the volumes with that ownership.
RUN mkdir -p data/scifact indexes results "${HF_HOME}" \
 && chown -R app:app data indexes results /home/app/.cache

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "serving.main"]
