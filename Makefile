# Developer entry points. All commands run from the repository root.
VLLM_IMAGE   ?= vllm-cpu-guard:0.31.0
LOADER_IMAGE ?= model-loader:1
UV_IMAGE     ?= ghcr.io/astral-sh/uv:0.13.6-python3.12-trixie-slim
UV_RUN        = docker run --rm -v "$(CURDIR)":/w -v uvcache:/uvcache -w /w -e UV_CACHE_DIR=/uvcache $(UV_IMAGE)

.PHONY: lock build test smoke smoke-stock scan

## lock: re-resolve and hash-pin all Python dependencies (both images, both CPU architectures)
lock:
	$(UV_RUN) uv pip compile docker/vllm/requirements.in --universal --python-version 3.12 --generate-hashes \
		--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
		-o docker/vllm/requirements.txt
	$(UV_RUN) uv pip compile docker/loader/requirements.in --universal --python-version 3.13 --generate-hashes \
		-o docker/loader/requirements.txt

## build: build both images for the native architecture
build:
	docker build -f docker/vllm/Dockerfile -t $(VLLM_IMAGE) .
	docker build -f docker/loader/Dockerfile -t $(LOADER_IMAGE) .

## test: unit tests (guard middleware, log redaction, entrypoint policy, loader verification)
test:
	uv run --python 3.12 --no-project --with pytest --with anyio --with httpx --with pyyaml python -m pytest -q

## smoke: plain-Docker validation of the guarded image (restricted flags, offline, auth, canary)
smoke:
	scripts/docker-smoke.sh

## smoke-stock: the same checks against stock `vllm serve --api-key` (demonstrates the /invocations bypass)
smoke-stock:
	MODE=stock scripts/docker-smoke.sh

## scan: Dockerfile lint + HIGH/CRITICAL CVEs and secrets in both images + k8s misconfigurations
scan:
	mkdir -p results/scan
	hadolint docker/vllm/Dockerfile docker/loader/Dockerfile | tee results/scan/hadolint.txt || true
	trivy image --quiet --scanners vuln,secret --severity HIGH,CRITICAL $(VLLM_IMAGE) | tee results/scan/trivy-vllm.txt
	trivy image --quiet --scanners vuln,secret --severity HIGH,CRITICAL $(LOADER_IMAGE) | tee results/scan/trivy-loader.txt
	trivy config --quiet --severity HIGH,CRITICAL k8s/ | tee results/scan/trivy-k8s.txt
