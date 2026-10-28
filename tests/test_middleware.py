import json
import logging

import httpx
import pytest

from vllm_guard import middleware as mw

TOKEN = "t" * 64


async def echo_app(scope, receive, send):
    """Downstream stand-in for vLLM: echoes the request body and a usage block."""
    if scope["type"] != "http":
        return
    body = b""
    more = True
    while more:
        msg = await receive()
        body += msg.get("body", b"")
        more = msg.get("more_body", False)
    if scope["path"] == "/stream":
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b'data: {"choices":[]}\n\n', "more_body": True})
        await send({"type": "http.response.body",
                    "body": b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n'})
        return
    payload = {"received": json.loads(body) if body else None,
               "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
    raw = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": raw})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", TOKEN)
    monkeypatch.setenv("GUARD_MAX_TOKENS", "256")
    app = mw.GuardMiddleware(echo_app)
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 1234))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/invocations", "/metrics", "/tokenize", "/openapi.json", "/v1/models",
                                  "/version", "/load", "/docs", "/health/", "/HEALTH"])
async def test_everything_but_health_requires_token(client, path):
    async with client:
        r = await client.get(path)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.anyio
async def test_health_is_public(client):
    async with client:
        r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_wrong_token_and_wrong_scheme(client):
    async with client:
        assert (await client.get("/metrics", headers={"Authorization": "Bearer nope"})).status_code == 401
        assert (await client.get("/metrics", headers={"Authorization": f"Basic {TOKEN}"})).status_code == 401
        assert (await client.get("/metrics", headers=AUTH)).status_code == 200


@pytest.mark.anyio
async def test_invocations_bypass_closed(client):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    async with client:
        assert (await client.post("/invocations", json=body)).status_code == 401
        assert (await client.post("/invocations", json=body, headers=AUTH)).status_code == 200


@pytest.mark.anyio
async def test_max_tokens_limit_and_injection(client):
    async with client:
        too_many = await client.post("/v1/chat/completions", headers=AUTH,
                                     json={"messages": [], "max_tokens": 10_000})
        assert too_many.status_code == 400
        injected = await client.post("/v1/chat/completions", headers=AUTH, json={"messages": []})
        assert injected.status_code == 200
        assert injected.json()["received"]["max_tokens"] == 256
        many_n = await client.post("/v1/completions", headers=AUTH, json={"prompt": "x", "max_tokens": 5, "n": 99})
        assert many_n.status_code == 400


@pytest.mark.anyio
async def test_body_limit(client, monkeypatch):
    monkeypatch.setenv("GUARD_MAX_BODY_BYTES", "100")
    app = mw.GuardMiddleware(echo_app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/completions", headers=AUTH, content=b"x" * 500)
    assert r.status_code == 413


@pytest.mark.anyio
async def test_access_log_is_content_free(client, caplog):
    secret = "CANARY-7f3c-prompt-text"
    caplog.set_level(logging.INFO, logger="vllm_guard.access")
    async with client:
        r = await client.post("/v1/chat/completions", headers=AUTH,
                              json={"messages": [{"role": "user", "content": secret}], "max_tokens": 8,
                                    "stop": [secret]})
    assert r.status_code == 200 and r.headers["x-request-id"]
    events = [rec.event for rec in caplog.records if hasattr(rec, "event")]
    assert len(events) == 1
    event = events[0]
    assert event["status"] == 200 and event["auth"] == "ok"
    assert event["prompt_tokens"] == 11 and event["completion_tokens"] == 7
    dumped = json.dumps(event) + " ".join(rec.getMessage() for rec in caplog.records)
    assert secret not in dumped and TOKEN not in dumped


@pytest.mark.anyio
async def test_streaming_usage_is_sniffed(client, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="vllm_guard.access")
    async with client:
        r = await client.get("/stream", headers=AUTH)
    assert r.status_code == 200
    event = [rec.event for rec in caplog.records if hasattr(rec, "event")][0]
    assert event["prompt_tokens"] == 5 and event["completion_tokens"] == 2 and event["sse_events"] == 3


def test_empty_api_key_refuses(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "")
    with pytest.raises(RuntimeError):
        mw.GuardMiddleware(echo_app)


@pytest.fixture
def anyio_backend():
    return "asyncio"
