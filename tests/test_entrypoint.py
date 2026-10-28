from vllm_guard.entrypoint import BAKED_LOGGING_CONFIG, parse_cli, validate

GOOD_ENV = {
    "VLLM_LOGGING_CONFIG_PATH": BAKED_LOGGING_CONFIG,
    "HF_HUB_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_API_KEY": "k" * 64,
}
GOOD_CONFIG = {"model": "/models/x", "kv-cache-memory-bytes": 2147483648, "enable-log-requests": True}


def test_good_config_passes():
    assert validate(GOOD_CONFIG, {}, GOOD_ENV) == []


def test_content_logging_and_remote_code_are_refused():
    cfg = dict(GOOD_CONFIG, enable_log_outputs=True, **{"trust-remote-code": "true"})
    problems = validate(cfg, {}, GOOD_ENV)
    assert any("enable-log-outputs" in p for p in problems)
    assert any("trust-remote-code" in p for p in problems)


def test_cli_args_are_checked_too():
    problems = validate(GOOD_CONFIG, parse_cli(["--enable-log-outputs", "--api-key", "x"]), GOOD_ENV)
    assert len(problems) == 2


def test_env_guards():
    env = dict(GOOD_ENV, VLLM_DEBUG_LOG_API_SERVER_RESPONSE="1", VLLM_LOGGING_LEVEL="debug",
               VLLM_SERVER_DEV_MODE="1", HF_HUB_OFFLINE="0", VLLM_API_KEY="short")
    assert len(validate(GOOD_CONFIG, {}, env)) == 5


def test_kv_cache_size_must_be_explicit():
    cfg = {k: v for k, v in GOOD_CONFIG.items() if k != "kv-cache-memory-bytes"}
    assert any("kv-cache-memory-bytes" in p for p in validate(cfg, {}, GOOD_ENV))
    assert validate(cfg, {}, dict(GOOD_ENV, VLLM_CPU_KVCACHE_SPACE="2")) == []


def test_parse_cli_forms():
    assert parse_cli(["--a", "1", "--b=2", "--c", "--no-d"]) == {"a": "1", "b": "2", "c": True, "d": False}
