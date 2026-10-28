import json
import logging

from vllm_guard.logfilter import REQUEST_LOGGER, RESPONSE_BODY_LOGGER, JsonFormatter, RedactionFilter

CANARY_TEXT = "CANARY-prompt-1234"


class FakeSamplingParams:
    def __init__(self):
        self.n = 1
        self.max_tokens = 64
        self.temperature = 0.0
        self.top_p = 1.0
        self.stop = [CANARY_TEXT]
        self.structured_outputs = {"json": {"title": CANARY_TEXT}}

    def __repr__(self):
        return f"SamplingParams(stop={self.stop!r}, structured_outputs={self.structured_outputs!r})"


def make(name, msg, *args, level=logging.INFO, exc_info=None):
    return logging.LogRecord(name, level, __file__, 1, msg, args, exc_info)


def render(record):
    return JsonFormatter().format(record)


def test_content_records_are_dropped():
    f = RedactionFilter()
    assert not f.filter(make(REQUEST_LOGGER, "Request %s details: prompt: %r, prompt_token_ids: %s, "
                                             "prompt_embeds shape: %s.", "r1", CANARY_TEXT, [1], None, level=10))
    assert not f.filter(make(REQUEST_LOGGER, "Request %s JSON body: %s", "r1", CANARY_TEXT, level=10))
    assert not f.filter(make(REQUEST_LOGGER, "Generated response %s%s: output: %r, finish_reason: %s",
                             "r1", "", CANARY_TEXT, "stop"))
    assert not f.filter(make(RESPONSE_BODY_LOGGER, "response_body={%s}", CANARY_TEXT))
    assert not f.filter(make(REQUEST_LOGGER, "some future message %s", CANARY_TEXT))  # fail closed


def test_crash_dump_keeps_only_engine_config():
    f = RedactionFilter()
    dump = "vllm.logging_utils.dump_input"
    assert f.filter(make(dump, "Dumping input data for V1 LLM engine (v0.31.0) with config: model=x", level=40))
    assert not f.filter(make(dump, "Dumping scheduler output for model execution: %s",
                             f"SamplingParams(stop=['{CANARY_TEXT}'])", level=40))
    assert not f.filter(make(dump, "Dumping scheduler stats: %s", "x", level=40))


def test_received_request_keeps_fact_without_content():
    f = RedactionFilter()
    rec = make(REQUEST_LOGGER, "Received request %s: params: %s, lora_request: %s.", "req-42",
               FakeSamplingParams(), None)
    assert f.filter(rec)
    line = render(rec)
    msg = json.loads(line)["msg"]
    assert "req-42" in msg and '"max_tokens": 64' in msg
    assert CANARY_TEXT not in line
    assert "has_stop" in msg and "has_structured_outputs" in msg


def test_pydantic_input_value_is_redacted():
    f = RedactionFilter()
    rec = make("vllm.entrypoints.openai", "1 validation error: [type=string_type, input_value='%s', "
                                          "input_type=str]" % CANARY_TEXT)
    assert f.filter(rec)
    assert CANARY_TEXT not in render(rec)


def test_json_formatter_is_single_line_json():
    rec = make("vllm_guard.access", "request")
    rec.event = {"event": "request", "status": 200}
    line = render(rec)
    assert "\n" not in line
    doc = json.loads(line)
    assert doc["status"] == 200 and doc["logger"] == "vllm_guard.access"
