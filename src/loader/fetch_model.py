"""Fetch and verify model weights for vLLM (runs as a Kubernetes initContainer).

  first start      : policy checks -> download into a temp dir -> verify -> atomic rename -> marker
  every next start : marker found -> full sha256 re-check against the baked lock -> exit (NO network)

Verification layers
  provenance  MODEL_ID must be listed in the lock baked into this image (= allowlist, typosquatting),
              MODEL_REVISION must be a full commit sha listed for that model (immutable content)
  scanners    Hugging Face securityRepoStatus (malware/pickle scans) must report no issues
  formats     only safetensors weights + json/txt/tokenizer files; pickle/python never downloaded;
              config.json with `auto_map` (custom code / trust_remote_code) is rejected
  integrity   each file vs Hugging Face metadata (LFS sha256, git blob sha1 for small files)
              AND vs the sha256 recorded in models.lock.json (two independent sources)
  structure   safetensors header: bounded size, allowed dtypes, contiguous in-bounds offsets,
              parameter count equal to the lock

Environment
  MODEL_ID, MODEL_REVISION   what to fetch (from the ConfigMap)
  MODELS_DIR                 PVC mount (default /models)
  LOCK_PATH                  baked lock (default /opt/loader/models.lock.json)
  HF_TOKEN_PATH              optional token FILE mounted from the Secret (read by huggingface_hub)
  FETCH_MODE                 ensure (default) | lock  (lock: print a lock entry, used once by maintainers)
  REPAIR                     1 = delete a copy that fails re-verification and download again

Exit codes: 0 ok, 2 access denied (token), 3 integrity failure, 4 policy violation, 5 other error.
"""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_ACCESS, EXIT_INTEGRITY, EXIT_POLICY, EXIT_OTHER = 0, 2, 3, 4, 5

ALLOWED_PATTERNS = ("*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                    "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja", "chat_template.json",
                    "preprocessor_config.json", "recipe.yaml")
DENIED_PATTERNS = ("*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.ckpt", "*.py", "*.h5", "*.msgpack",
                   "*.gguf", "*.onnx", "*.npy", "*.npz", "*.joblib")
SAFETENSORS_DTYPES = {"BOOL", "U8", "I8", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64", "I64", "U64",
                      "F8_E4M3", "F8_E5M2"}
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
CHUNK = 8 * 1024 * 1024
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class FetchError(Exception):
    def __init__(self, code: int, message: str, **fields: Any) -> None:
        super().__init__(message)
        self.code, self.fields = code, fields


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}, ensure_ascii=False), flush=True)


# --------------------------------------------------------------------------- hashing / formats
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def safetensors_params(path: Path) -> int:
    """Validate the safetensors header and return the number of tensor elements."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise FetchError(EXIT_INTEGRITY, f"{path.name}: truncated safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > MAX_SAFETENSORS_HEADER or 8 + header_len > size:
            raise FetchError(EXIT_INTEGRITY, f"{path.name}: invalid safetensors header length {header_len}")
        header = json.loads(fh.read(header_len))
    if not isinstance(header, dict):
        raise FetchError(EXIT_INTEGRITY, f"{path.name}: safetensors header is not an object")
    data_len = size - 8 - header_len
    spans, params = [], 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPES:
            raise FetchError(EXIT_INTEGRITY, f"{path.name}: tensor {name} has unexpected dtype {dtype}")
        if not (isinstance(offsets, list) and len(offsets) == 2 and 0 <= offsets[0] <= offsets[1] <= data_len):
            raise FetchError(EXIT_INTEGRITY, f"{path.name}: tensor {name} has out-of-bounds offsets")
        count = 1
        for dim in shape or []:
            count *= int(dim)
        params += count
        spans.append((offsets[0], offsets[1]))
    spans.sort()
    cursor = 0
    for start, end in spans:
        if start != cursor:
            raise FetchError(EXIT_INTEGRITY, f"{path.name}: tensor data is not contiguous (gap/overlap at {start})")
        cursor = end
    if cursor != data_len:
        raise FetchError(EXIT_INTEGRITY, f"{path.name}: {data_len - cursor} trailing bytes after tensor data")
    return params


def is_allowed(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    if any(fnmatch.fnmatch(base, p) for p in DENIED_PATTERNS):
        return False
    return any(fnmatch.fnmatch(base, p) for p in ALLOWED_PATTERNS)


# --------------------------------------------------------------------------- lock / layout
def load_lock(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("models", {})
    except FileNotFoundError:
        return {}


def target_dir(models_dir: Path, model_id: str, revision: str) -> Path:
    return models_dir / model_id.replace("/", "--") / revision


def verify_against_lock(target: Path, entry: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    total = 0
    expected = entry["files"]
    present = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() and p.name != ".complete"}
    if present != set(expected):
        raise FetchError(EXIT_INTEGRITY, "file set differs from lock",
                         missing=sorted(set(expected) - present), unexpected=sorted(present - set(expected)))
    for name, meta in sorted(expected.items()):
        path = target / name
        if path.stat().st_size != meta["size"]:
            raise FetchError(EXIT_INTEGRITY, f"{name}: size differs from lock")
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise FetchError(EXIT_INTEGRITY, f"{name}: sha256 differs from lock", got=digest, want=meta["sha256"])
        total += meta["size"]
    return {"verify_ms": round((time.perf_counter() - started) * 1000), "verified_bytes": total}


# --------------------------------------------------------------------------- hugging face
def hf_access_error(exc: Exception) -> FetchError | None:
    from huggingface_hub import errors as hf_errors

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, hf_errors.GatedRepoError):
        return FetchError(EXIT_ACCESS, "gated model: token missing/invalid or the account has not been granted "
                                       "access (accept the license on huggingface.co)")
    if isinstance(exc, hf_errors.RepositoryNotFoundError) or status in (401, 403):
        return FetchError(EXIT_ACCESS, "repository not found or private: a valid HF token with read access is "
                                       "required (Secret hf-token)")
    if isinstance(exc, hf_errors.RevisionNotFoundError):
        return FetchError(EXIT_POLICY, "pinned revision does not exist in the repository")
    return None


def remote_manifest(model_id: str, revision: str, allow_unscanned: bool) -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True, securityStatus=True)
    except Exception as exc:  # noqa: BLE001 - mapped to exit codes below
        raise hf_access_error(exc) or FetchError(EXIT_OTHER, f"model_info failed: {type(exc).__name__}") from exc

    if getattr(info, "sha", revision) != revision:
        raise FetchError(EXIT_POLICY, "hub returned a different revision than pinned", got=info.sha)
    status = getattr(info, "security_repo_status", None) or {}
    issues = status.get("filesWithIssues") or []
    if issues:
        raise FetchError(EXIT_POLICY, "Hugging Face security scanners flagged files", files=issues)
    if not status.get("scansDone") and not allow_unscanned:
        raise FetchError(EXIT_POLICY, "repository has no completed security scan (set allow_unscanned in the "
                                      "lock only for repositories you control)")

    manifest: dict[str, dict[str, Any]] = {}
    skipped = []
    for sib in info.siblings or []:
        if not is_allowed(sib.rfilename):
            skipped.append(sib.rfilename)
            continue
        lfs = getattr(sib, "lfs", None)
        manifest[sib.rfilename] = {
            "size": sib.size,
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
            "blob_sha1": getattr(sib, "blob_id", None),
        }
    log("remote_manifest", files=sorted(manifest), skipped=sorted(skipped),
        scans_done=bool(status.get("scansDone")))
    if not any(n.endswith(".safetensors") for n in manifest):
        raise FetchError(EXIT_POLICY, "no safetensors weights in the repository (pickle-only repos are refused)")
    return manifest


def download(model_id: str, revision: str, manifest: dict[str, dict[str, Any]], tmp: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name, meta in sorted(manifest.items()):
        started = time.perf_counter()
        try:
            hf_hub_download(model_id, name, revision=revision, local_dir=tmp)
        except Exception as exc:  # noqa: BLE001
            raise hf_access_error(exc) or FetchError(EXIT_OTHER, f"download failed: {name}: {type(exc).__name__}") from exc
        path = tmp / name
        size = path.stat().st_size
        if meta["size"] is not None and size != meta["size"]:
            raise FetchError(EXIT_INTEGRITY, f"{name}: size differs from hub metadata")
        if meta["lfs_sha256"]:
            if sha256_file(path) != meta["lfs_sha256"]:
                raise FetchError(EXIT_INTEGRITY, f"{name}: sha256 differs from hub LFS metadata")
        elif meta["blob_sha1"] and git_blob_sha1(path) != meta["blob_sha1"]:
            raise FetchError(EXIT_INTEGRITY, f"{name}: git blob sha1 differs from hub metadata")
        log("downloaded", file=name, bytes=size, ms=round((time.perf_counter() - started) * 1000))
    shutil.rmtree(tmp / ".cache", ignore_errors=True)  # huggingface_hub local_dir bookkeeping


def check_structure(tmp: Path) -> int:
    config = tmp / "config.json"
    if config.exists():
        cfg = json.loads(config.read_text(encoding="utf-8"))
        if "auto_map" in cfg:
            raise FetchError(EXIT_POLICY, "config.json declares auto_map (custom modelling code); refused")
    return sum(safetensors_params(p) for p in sorted(tmp.rglob("*.safetensors")))


def lock_entry(tmp: Path, params: int) -> dict[str, Any]:
    files = {p.relative_to(tmp).as_posix(): {"sha256": sha256_file(p), "size": p.stat().st_size}
             for p in sorted(tmp.rglob("*")) if p.is_file()}
    return {"files": files, "params": params}


def write_marker(target: Path, model_id: str, revision: str, params: int) -> None:
    marker = target / ".complete"
    tmp = target / ".complete.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump({"model": model_id, "revision": revision, "params": params, "ts": time.time()}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, marker)


# --------------------------------------------------------------------------- main flow
def run() -> dict[str, Any]:
    model_id = os.environ.get("MODEL_ID", "").strip()
    revision = os.environ.get("MODEL_REVISION", "").strip()
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    mode = os.environ.get("FETCH_MODE", "ensure")
    lock = load_lock(os.environ.get("LOCK_PATH", "/opt/loader/models.lock.json"))

    if not model_id or not REVISION_RE.match(revision):
        raise FetchError(EXIT_POLICY, "MODEL_ID and a full 40-hex MODEL_REVISION (commit sha) are required")
    entry = lock.get(model_id, {}).get(revision)
    if mode == "ensure" and entry is None:
        raise FetchError(EXIT_POLICY, "model/revision is not in the allowlist (models.lock.json baked into "
                                      "the loader image)", model=model_id, revision=revision)

    target = target_dir(models_dir, model_id, revision)
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / ".fetch.lock").open("w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)

        if mode == "ensure" and (target / ".complete").exists():
            try:
                result = verify_against_lock(target, entry)
                log("verified", source="pvc", network=False, **result)
                return {"status": "cached", **result}
            except FetchError:
                if os.environ.get("REPAIR") != "1":
                    raise
                log("repair", reason="re-verification failed, deleting local copy")
                shutil.rmtree(target)

        started = time.perf_counter()
        allow_unscanned = bool((entry or {}).get("allow_unscanned", False)) or (
            mode == "lock" and os.environ.get("ALLOW_UNSCANNED") == "1")
        manifest = remote_manifest(model_id, revision, allow_unscanned)
        tmp = models_dir / ".tmp" / f"{target.parent.name}-{revision}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            download(model_id, revision, manifest, tmp)
            params = check_structure(tmp)
            computed = lock_entry(tmp, params)
            if mode == "lock":
                if allow_unscanned:
                    computed["allow_unscanned"] = True
                # Trust-on-first-use bootstrap for maintainers: files were verified against hub metadata;
                # the printed entry is reviewed and committed to models.lock.json.
                log("lock_entry", entry={model_id: {revision: computed}})
                entry = computed
            if computed["files"] != entry["files"]:
                raise FetchError(EXIT_INTEGRITY, "downloaded files differ from models.lock.json")
            if computed["params"] != entry.get("params"):
                raise FetchError(EXIT_INTEGRITY, "parameter count differs from models.lock.json",
                                 got=computed["params"], want=entry.get("params"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)  # atomic on the same filesystem
            write_marker(target, model_id, revision, params)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        total = sum(m["size"] for m in entry["files"].values())
        result = {"download_ms": round((time.perf_counter() - started) * 1000), "bytes": total, "params": params}
        log("downloaded_all", **result)
        return {"status": "downloaded", **result}


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    started = time.perf_counter()
    try:
        result = run()
    except FetchError as exc:
        log("failed", code=exc.code, error=str(exc), **exc.fields)
        return exc.code
    except Exception as exc:  # noqa: BLE001
        log("failed", code=EXIT_OTHER, error=f"{type(exc).__name__}: {exc}")
        return EXIT_OTHER
    log("done", total_ms=round((time.perf_counter() - started) * 1000), **result)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
