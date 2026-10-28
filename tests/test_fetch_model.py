import json
import struct

import pytest

import fetch_model as fm


def write_safetensors(path, tensors, trailing=b""):
    header, offset, blobs = {}, 0, b""
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        blobs += b"\0" * nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blobs + trailing)


def test_valid_safetensors_param_count(tmp_path):
    p = tmp_path / "model.safetensors"
    write_safetensors(p, {"a": ("BF16", [2, 3], 12), "b": ("F32", [4], 16)})
    assert fm.safetensors_params(p) == 10


@pytest.mark.parametrize("tensors,trailing", [
    ({"a": ("PICKLE", [1], 4)}, b""),   # unknown dtype
    ({"a": ("F32", [1], 4)}, b"junk"),  # trailing bytes
])
def test_invalid_safetensors(tmp_path, tensors, trailing):
    p = tmp_path / "bad.safetensors"
    write_safetensors(p, tensors, trailing)
    with pytest.raises(fm.FetchError):
        fm.safetensors_params(p)


def test_huge_header_rejected(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(fm.FetchError):
        fm.safetensors_params(p)


@pytest.mark.parametrize("name,ok", [
    ("model.safetensors", True), ("model-00001-of-00002.safetensors", True), ("config.json", True),
    ("tokenizer.json", True), ("tokenizer.model", True), ("pytorch_model.bin", False),
    ("modeling_qwen.py", False), ("model.pt", False), ("README.md", False), ("x.gguf", False),
])
def test_file_policy(name, ok):
    assert fm.is_allowed(name) is ok


def test_auto_map_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x.Y"}}))
    with pytest.raises(fm.FetchError) as err:
        fm.check_structure(tmp_path)
    assert err.value.code == fm.EXIT_POLICY


def test_verify_against_lock_detects_tampering(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    entry = fm.lock_entry(tmp_path, 0)
    assert fm.verify_against_lock(tmp_path, entry)["verified_bytes"] == 2
    (tmp_path / "config.json").write_text("{ }")
    with pytest.raises(fm.FetchError) as err:
        fm.verify_against_lock(tmp_path, entry)
    assert err.value.code == fm.EXIT_INTEGRITY


def test_unlisted_model_is_policy_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ID", "Qwenn/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("LOCK_PATH", str(tmp_path / "missing.json"))
    assert fm.main() == fm.EXIT_POLICY


def test_git_blob_sha1_matches_git(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert fm.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object`
