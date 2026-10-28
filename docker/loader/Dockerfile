# syntax=docker/dockerfile:1.7
# Lightweight rootless image for the model-fetch initContainer (Bonus 2).
# distroless: no shell, no package manager, only the Python runtime (~70 MB in total).
# Build:  docker build -f docker/loader/Dockerfile -t model-loader:1 .

ARG BUILDER_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
ARG RUNTIME_IMAGE=gcr.io/distroless/python3-debian12:nonroot@sha256:7d1042ce588ab97019fe95c24ffca7bc5a82ccdac572511d5e09bda4435c89c5
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0

FROM ${UV_IMAGE} AS uv

# Same CPython minor (3.11) and glibc family (Debian 12) as the distroless runtime.
FROM ${BUILDER_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
COPY docker/loader/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PYTHON_DOWNLOADS=never uv pip install --python /usr/local/bin/python3.11 --require-hashes \
        --no-compile --target /deps -r /tmp/requirements.txt

FROM ${RUNTIME_IMAGE}
COPY --from=builder /deps /opt/loader/deps
COPY src/loader/fetch_model.py /opt/loader/fetch_model.py
# The lock doubles as the allowlist of models/revisions; changing it requires rebuilding this image.
COPY docker/loader/models.lock.json /opt/loader/models.lock.json
ENV PYTHONPATH=/opt/loader/deps \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    LOCK_PATH=/opt/loader/models.lock.json \
    MODELS_DIR=/models
USER 10001:10001
ENTRYPOINT ["/usr/bin/python3", "/opt/loader/fetch_model.py"]
