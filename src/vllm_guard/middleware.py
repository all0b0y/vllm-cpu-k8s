"""ASGI middleware: deny-by-default authentication, generation limits and a content-free access log.

Why: vLLM's built-in `--api-key` check only guards the /v1, /v2, /inference and /cohere prefixes
(see GUARDED_PREFIX in vllm/entrypoints/serve/middleware/authenticate.py). Its own docstring warns
that /invocations exposes the same inference capabilities unauthenticated; /metrics, /tokenize,
/openapi.json, ... are open too. This middleware requires the Bearer token on EVERY path except an
explicit allowlist (default: /health, used by kubelet probes).

Loaded by vLLM through the official flag:  vllm serve --middleware vllm_guard.middleware.GuardMiddleware
Configuration (environment):
  VLLM_API_KEY          the same secret vLLM uses (required)
  GUARD_PUBLIC_PATHS    comma-separated exact paths that need no token      (default "/health")
  GUARD_MAX_TOKENS      upper bound for max_tokens / max_completion_tokens   (default 2048)
  GUARD_MAX_N           upper bound for `n` (parallel samples per request)   (default 4)
  GUARD_MAX_BODY_BYTES  request body limit                                   (default 8 MiB)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

access_log = logging.getLogger("vllm_guard.access")

GENERATION_PATHS = {
    "/v1/chat/completions": ("max_completion_tokens", "max_tokens"),
    "/v1/completions": ("max_tokens",),
    "/v1/responses": ("max_output_tokens",),
    "/invocations": ("max_completion_tokens", "max_tokens"),
}
TRUSTED_PROXIES = {"127.0.0.1", "::1"}
JSON_SNIFF_LIMIT = 1 << 20  # parse `usage` from non-streaming JSON responses up to 1 MiB


def _hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


class GuardMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        tokens = [t.strip() for t in os.environ.get("VLLM_API_KEY", "").split(",") if t.strip()]
        if not tokens:
            raise RuntimeError("vllm_guard: VLLM_API_KEY is empty; refusing to serve without authentication")
        self._token_hashes = [_hash(t) for t in tokens]
        self.public_paths = {p.strip() for p in os.environ.get("GUARD_PUBLIC_PATHS", "/health").split(",") if p.strip()}
        self.max_tokens = _env_int("GUARD_MAX_TOKENS", 2048)
        self.max_n = _env_int("GUARD_MAX_N", 4)
        self.max_body = _env_int("GUARD_MAX_BODY_BYTES", 8 << 20)

    # ------------------------------------------------------------------ auth
    def _auth_status(self, headers: dict[bytes, bytes]) -> str:
        value = headers.get(b"authorization")
        if not value:
            return "missing"
        scheme, _, token = value.decode("latin-1").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return "invalid"
        presented = _hash(token.strip())
        match = False
        for expected in self._token_hashes:  # no early exit: constant work per request
            match |= hmac.compare_digest(presented, expected)
        return "ok" if match else "invalid"

    # ------------------------------------------------------------------ ASGI
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope["path"].removeprefix(scope.get("root_path", "") or "")
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        public = path in self.public_paths
        auth = "public" if public else self._auth_status(headers)

        if scope["type"] == "websocket":
            if auth not in ("ok", "public"):
                self._log(scope, path, headers, {"status": 403, "auth": auth, "websocket": True}, time.perf_counter())
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        info: dict[str, Any] = {"auth": auth, "status": None}

        if auth not in ("ok", "public"):
            info["status"] = 401
            await _send_json(send, 401, {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                             extra_headers=[(b"www-authenticate", b"Bearer")])
            self._log(scope, path, headers, info, started)
            return

        request_id = headers.get(b"x-request-id", b"").decode("latin-1") or uuid.uuid4().hex
        info["request_id"] = request_id
        if b"x-request-id" not in headers:
            scope = dict(scope)
            scope["headers"] = list(scope.get("headers", [])) + [(b"x-request-id", request_id.encode())]

        # Generation endpoints: read the (bounded) body, enforce limits, replay it downstream.
        if scope.get("method") == "POST" and path in GENERATION_PATHS:
            body, too_large = await _read_body(receive, self.max_body)
            info["req_bytes"] = len(body)
            if too_large:
                info["status"] = 413
                await _send_json(send, 413, {"error": {"message": f"Request body exceeds {self.max_body} bytes",
                                                       "type": "invalid_request_error"}})
                self._log(scope, path, headers, info, started)
                return
            body, error = self._apply_limits(path, body, info)
            if error is not None:
                info["status"] = 400
                await _send_json(send, 400, {"error": {"message": error, "type": "invalid_request_error"}})
                self._log(scope, path, headers, info, started)
                return
            scope = _with_content_length(scope, len(body))
            receive = _replay(body, receive)

        state: dict[str, Any] = {"first_byte": None, "bytes": 0, "sse": False, "json": False,
                                 "buf": [], "buf_len": 0, "usage": None, "events": 0}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                info["status"] = message["status"]
                resp_headers = list(message.get("headers", []))
                ctype = next((v for k, v in resp_headers if k.lower() == b"content-type"), b"")
                state["sse"] = ctype.startswith(b"text/event-stream")
                state["json"] = ctype.startswith(b"application/json")
                if not any(k.lower() == b"x-request-id" for k, _ in resp_headers):
                    resp_headers.append((b"x-request-id", request_id.encode()))
                message = dict(message, headers=resp_headers)
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    if state["first_byte"] is None:
                        state["first_byte"] = time.perf_counter()
                    state["bytes"] += len(chunk)
                    if state["sse"]:
                        _sniff_sse(chunk, state)
                    elif state["json"] and state["buf_len"] < JSON_SNIFF_LIMIT:
                        state["buf"].append(chunk)
                        state["buf_len"] += len(chunk)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            info["status"] = info["status"] or 500
            raise
        finally:
            if state["json"] and state["buf"]:
                state["usage"] = _usage_from_json(b"".join(state["buf"]))
            usage = state["usage"] or {}
            info.update(
                resp_bytes=state["bytes"],
                ttfb_ms=round((state["first_byte"] - started) * 1000, 1) if state["first_byte"] else None,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                sse_events=state["events"] if state["sse"] else None,
            )
            if not (public and info["status"] == 200):  # do not flood the log with probe hits
                self._log(scope, path, headers, info, started)

    # --------------------------------------------------------------- limits
    def _apply_limits(self, path: str, body: bytes, info: dict[str, Any]) -> tuple[bytes, str | None]:
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body, None  # let vLLM produce its own 400
        if not isinstance(payload, dict):
            return body, None
        info["stream"] = bool(payload.get("stream", False))

        n = payload.get("n")
        if isinstance(n, int) and n > self.max_n:
            return body, f"n={n} exceeds the limit of {self.max_n}"

        fields = GENERATION_PATHS[path]
        given = {f: payload[f] for f in fields if payload.get(f) is not None}
        for field, value in given.items():
            if not isinstance(value, int) or isinstance(value, bool):
                return body, f"{field} must be an integer"
            if value > self.max_tokens:
                return body, f"{field}={value} exceeds the limit of {self.max_tokens}"
        if not given:
            # Without an explicit value vLLM would allow up to max_model_len - prompt tokens.
            payload[fields[-1]] = self.max_tokens
            info["max_tokens_injected"] = True
            info["max_tokens"] = self.max_tokens
            return json.dumps(payload).encode("utf-8"), None
        info["max_tokens"] = max(given.values())
        return body, None

    # --------------------------------------------------------------- logging
    def _log(self, scope: Scope, path: str, headers: dict[bytes, bytes], info: dict[str, Any], started: float) -> None:
        client = (scope.get("client") or (None, None))[0]
        if client in TRUSTED_PROXIES and headers.get(b"x-real-ip"):
            client = headers[b"x-real-ip"].decode("latin-1")
        event = {"event": "request", "method": scope.get("method", "WS"), "path": path, "client": client,
                 "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        event.update({k: v for k, v in info.items() if v is not None})
        access_log.info("request", extra={"event": event})


# ---------------------------------------------------------------------- helpers
async def _read_body(receive: Receive, limit: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    more = True
    while more:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            return b"", True
        chunks.append(chunk)
        more = message.get("more_body", False)
    return b"".join(chunks), False


def _replay(body: bytes, receive: Receive) -> Receive:
    sent = False

    async def replay() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _with_content_length(scope: Scope, length: int) -> Scope:
    headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
    headers.append((b"content-length", str(length).encode()))
    return dict(scope, headers=headers)


async def _send_json(send: Send, status: int, payload: dict[str, Any],
                     extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += extra_headers or []
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _sniff_sse(chunk: bytes, state: dict[str, Any]) -> None:
    for line in chunk.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        state["events"] += 1
        if b'"usage"' in line:
            usage = _usage_from_json(line[5:].strip())
            if usage:
                state["usage"] = usage


def _usage_from_json(raw: bytes) -> dict[str, Any] | None:
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    usage = doc.get("usage") if isinstance(doc, dict) else None
    if not isinstance(usage, dict):
        return None
    return {k: usage[k] for k in ("prompt_tokens", "completion_tokens") if isinstance(usage.get(k), int)}
