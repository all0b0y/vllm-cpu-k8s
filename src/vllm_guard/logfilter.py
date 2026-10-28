"""Logging: one JSON object per line, and a filter that keeps request *content* out of the logs.

Threat model: the ConfigMap may be edited (that is its purpose), so privacy must not depend on
vLLM flags being off. The filter is attached to every handler (see docker/vllm/logging.json),
so it sees records from all loggers in all vLLM processes (API server and engine core).

Rules
  1. Records that exist only to carry content are dropped:
     prompt / prompt_token_ids / JSON body (RequestLogger, DEBUG), generated output
     (RequestLogger.log_outputs), raw response bodies (log_response middleware).
  2. "Received request ... params: SamplingParams(...)" is kept as the *fact* of a request, but the
     params object is replaced by an allowlist summary: stop strings, logit_bias, structured-output
     schemas etc. are user content too.
  3. Pydantic validation errors may echo the user's input (`input_value=...`): redacted.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from typing import Any

REQUEST_LOGGER = "vllm.entrypoints.serve.utils.request_logger"
RESPONSE_BODY_LOGGER = "vllm.entrypoints.serve.middleware.log_response"
# On an engine crash vLLM dumps the scheduler state, incl. full SamplingParams (stop strings, schemas...).
CRASH_DUMP_LOGGER = "vllm.logging_utils.dump_input"
_CRASH_DUMP_SAFE_PREFIX = "Dumping input data for V1 LLM engine"  # engine config only, no request data

# Message templates (record.msg, i.e. before %-formatting) that carry content.
_CONTENT_TEMPLATES = (
    "Request %s details: prompt",
    "Request %s JSON body",
    "Generated response %s%s details",
    "Generated response %s%s: output",
)
_RECEIVED_TEMPLATE = "Received request %s: params:"

SAFE_PARAM_FIELDS = ("n", "max_tokens", "temperature", "top_p", "top_k", "seed")

_INPUT_VALUE_RE = re.compile(r"input_value=(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+?)(?=,\s*input_type=|\]|$)",
                             re.DOTALL)
REDACTED = "<redacted>"


def summarize_params(params: Any) -> dict[str, Any]:
    """Allowlisted, content-free view of SamplingParams / PoolingParams / BeamSearchParams."""
    if params is None:
        return {}
    out: dict[str, Any] = {"type": type(params).__name__}
    for field in SAFE_PARAM_FIELDS:
        value = getattr(params, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[field] = value
    # Presence flags only (never the values): these fields contain user-supplied text/schemas.
    for field in ("stop", "structured_outputs", "logit_bias", "bad_words", "allowed_token_ids"):
        value = getattr(params, field, None)
        if value:
            out[f"has_{field}"] = True
    return out


def redact_text(text: str) -> str:
    return _INPUT_VALUE_RE.sub(f"input_value={REDACTED}", text)


class RedactionFilter(logging.Filter):
    """Drop or rewrite records that could contain prompts or generated text."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (logging API name)
        if record.name == RESPONSE_BODY_LOGGER:
            return False

        msg = record.msg if isinstance(record.msg, str) else str(record.msg)

        if record.name == CRASH_DUMP_LOGGER:
            return msg.startswith(_CRASH_DUMP_SAFE_PREFIX)

        if record.name == REQUEST_LOGGER or msg.startswith(_CONTENT_TEMPLATES + (_RECEIVED_TEMPLATE,)):
            if msg.startswith(_CONTENT_TEMPLATES):
                return False
            if msg.startswith(_RECEIVED_TEMPLATE):
                args = record.args if isinstance(record.args, tuple) else ()
                request_id = args[0] if args else None
                params = args[1] if len(args) > 1 else None
                record.msg = "Received request %s: params: %s"
                record.args = (request_id, json.dumps(summarize_params(params), sort_keys=True))
                return True
            # Unknown message from the request logger: fail closed.
            return False

        # Generic safety net for every other logger.
        if record.args:
            try:
                rendered = record.getMessage()
            except Exception:  # malformed record: keep the template only
                rendered = msg
            record.msg = redact_text(rendered)
            record.args = None
        else:
            record.msg = redact_text(msg)

        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            if type(exc).__name__ == "ValidationError":
                # pydantic errors render the offending input; keep only location + type.
                summary = _summarize_validation_error(exc)
                record.msg = f"{record.msg} [validation error: {summary}]"
                record.exc_info = None
                record.exc_text = None
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True


def _summarize_validation_error(exc: BaseException) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            items = errors()
            if isinstance(items, list):
                return json.dumps([{"loc": e.get("loc"), "type": e.get("type")} for e in items if isinstance(e, dict)],
                                  default=str)
        except Exception:
            pass
    return type(exc).__name__


class JsonFormatter(logging.Formatter):
    """Single-line JSON; adds structured fields passed via `extra={"event": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _dt.datetime.fromtimestamp(record.created, _dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "proc": getattr(record, "vllm_process_name", None) or record.processName,
            "msg": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        elif record.exc_text:
            payload["exc"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False, default=str)
