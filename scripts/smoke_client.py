"""Black-box checks against a running vLLM+guard endpoint (used by docker-smoke.sh and in k8s).

Usage: BASE_URL=http://vllm:8000 API_KEY=... CANARY=... python3 smoke_client.py
Prints one JSON line per check and exits non-zero on the first failed expectation.
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ["API_KEY"]
CANARY = os.environ.get("CANARY", "CANARY-unset")
MODEL = os.environ.get("MODEL_NAME", "qwen2.5-1.5b-instruct")
EXPECT_GUARD = os.environ.get("EXPECT_GUARD", "1") == "1"
AUTH = {"Authorization": f"Bearer {KEY}"}
failures = 0


def report(check: str, ok: bool, **fields) -> None:
    global failures
    failures += 0 if ok else 1
    print(json.dumps({"check": check, "ok": ok, **fields}, ensure_ascii=False), flush=True)


def wait_ready(timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    report("ready", False)
    sys.exit(1)


def chat(content: str, auth: dict | None = AUTH, path: str = "/v1/chat/completions", **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 32,
            "temperature": 0, **extra}
    return requests.post(f"{BASE}{path}", json=body, headers=auth or {}, timeout=600)


def main() -> int:
    t0 = time.time()
    wait_ready()
    report("ready", True, waited_s=round(time.time() - t0, 1))

    r = requests.get(f"{BASE}/health", timeout=10)
    report("health_public", r.status_code == 200, status=r.status_code)

    # Every route the server exposes, called WITHOUT a token.
    spec = requests.get(f"{BASE}/openapi.json", headers=AUTH, timeout=10)
    paths = sorted(spec.json()["paths"]) if spec.ok else []
    paths += ["/metrics", "/openapi.json", "/docs"]
    open_paths = []
    for p in sorted(set(paths)):
        if "{" in p:
            p = p.replace("{", "").replace("}", "")
        status = requests.get(f"{BASE}{p}", timeout=30).status_code
        if status != 401 and p != "/health":
            open_paths.append({"path": p, "status": status})
    report("no_token_everything_401", not open_paths if EXPECT_GUARD else True,
           routes_checked=len(set(paths)), open_without_token=open_paths)

    r = chat("Reply with one word: hello", auth=None, path="/invocations")
    report("invocations_without_token", (r.status_code == 401) if EXPECT_GUARD else True,
           status=r.status_code, note="200 here = documented vLLM --api-key bypass")
    if not EXPECT_GUARD:
        return 0

    r = chat("Reply with one word: hello", auth=None)
    report("chat_without_token", r.status_code == 401, status=r.status_code)

    r = chat("What is the capital of France? Answer in one word.")
    answer = r.json()["choices"][0]["message"]["content"] if r.ok else None
    report("chat_with_token", r.ok and bool(answer), status=r.status_code, answer=answer,
           usage=r.json().get("usage") if r.ok else None, request_id=r.headers.get("x-request-id"))

    r = chat("Say hi", max_tokens=100_000)
    report("max_tokens_limit", r.status_code == 400, status=r.status_code)

    # Canary request: secret marker in prompt, stop string and JSON schema; logs must never contain it.
    schema = {"type": "object", "title": CANARY, "properties": {"word": {"type": "string"}}, "required": ["word"]}
    r = chat(f"Repeat this code exactly: {CANARY}. Then write a short poem about the sea.", max_tokens=48,
             stop=[f"STOP-{CANARY}"],
             response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": schema}})
    canary_answer = r.json()["choices"][0]["message"]["content"] if r.ok else ""
    report("canary_request", r.ok, status=r.status_code, request_id=r.headers.get("x-request-id"))
    with open(os.environ.get("ANSWER_FILE", "/tmp/canary_answer.txt"), "w", encoding="utf-8") as fh:
        fh.write(canary_answer)

    # Streaming: time to the first chunk that actually carries generated text.
    started = time.perf_counter()
    first_token, chunks, text = None, 0, ""
    with requests.post(f"{BASE}/v1/chat/completions", headers=AUTH, stream=True, timeout=600,
                       json={"model": MODEL, "stream": True, "max_tokens": 32, "temperature": 0,
                             "messages": [{"role": "user", "content": "Count from 1 to 10."}]}) as resp:
        status = resp.status_code
        for line in resp.iter_lines():
            if not line.startswith(b"data: {"):
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content") or ""
            if delta:
                chunks += 1
                text += delta
                if first_token is None:
                    first_token = time.perf_counter()
    report("streaming", status == 200 and chunks > 0, status=status, content_chunks=chunks,
           ttft_ms=round((first_token - started) * 1000) if first_token else None,
           total_ms=round((time.perf_counter() - started) * 1000), text=text[:60])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
