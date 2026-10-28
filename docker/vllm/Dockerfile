# syntax=docker/dockerfile:1.7
# vLLM 0.31.0 (CPU) — rootless runtime image with hash-locked Python dependencies.
# Build:  docker build -f docker/vllm/Dockerfile -t vllm-cpu-guard:0.31.0 .
# Arch:   native (linux/arm64 on Apple Silicon, linux/amd64 on x86); one lock file covers both.

ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------- builder
FROM ${UBUNTU_IMAGE} AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv venv /opt/venv --python /usr/bin/python3.12
COPY docker/vllm/requirements.txt /tmp/requirements.txt
# --require-hashes: every wheel (incl. transitive deps and the vLLM wheel itself) must match its sha256.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync --python /opt/venv/bin/python --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        /tmp/requirements.txt

# ---------------------------------------------------------------------------- runtime
FROM ${UBUNTU_IMAGE} AS runtime
# python3.12      : interpreter for the venv
# libtcmalloc     : allocator recommended by vLLM CPU docs (LD_PRELOAD in the entrypoint)
# libnuma1        : linked by vLLM CPU kernels
# g++ / python-dev: torch.compile (inductor) builds C++ kernels at runtime on CPU
# openssl         : inductor checksums precompiled headers with the openssl CLI
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.12 libtcmalloc-minimal4t64 libnuma1 g++ libpython3.12-dev openssl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp/home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY src/vllm_guard /opt/app/vllm_guard
COPY docker/vllm/logging.json /etc/vllm/logging.json

# Writable paths are provided by volumes (emptyDir / PVC); the root filesystem stays read-only.
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/opt/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    VLLM_CACHE_ROOT=/var/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/var/cache/vllm/inductor \
    VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging.json \
    VLLM_CONFIG_FILE=/etc/vllm/config.yaml \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm_guard.entrypoint"]
