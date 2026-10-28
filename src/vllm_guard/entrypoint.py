"""Fail-closed container entrypoint: validate the (ConfigMap-provided) configuration, then exec vLLM.

The ConfigMap is meant to be edited without rebuilding the image, so it is also the easiest place
to accidentally switch on content logging or remote code. Instead of silently overriding such
settings, the entrypoint refuses to start and explains what to remove (exit code 78, EX_CONFIG).

Final command:
  vllm serve --config $VLLM_CONFIG_FILE --middleware vllm_guard.middleware.GuardMiddleware \
             --disable-uvicorn-access-log  [extra args]
"""

from __future__ import annotations

import glob
import json
import os
import platform
import sys
from typing import Any

import yaml

BAKED_LOGGING_CONFIG = "/etc/vllm/logging.json"
MIDDLEWARE = "vllm_guard.middleware.GuardMiddleware"
EX_CONFIG = 78
MIN_API_KEY_LEN = 32
TRUE = {"1", "true", "yes", "on"}

# vLLM option -> reason. Options are compared after normalising "_" to "-".
FORBIDDEN_TRUE = {
    "enable-log-outputs": "logs generated text (privacy requirement: answers must not reach stdout/stderr)",
    "trust-remote-code": "executes Python code shipped with the model repository",
}
FORBIDDEN_PRESENT = {
    "api-key": "secrets must not live in the ConfigMap or the command line (visible in ps/kubectl); "
               "use the VLLM_API_KEY env var from the Secret",
    "middleware": "the guard middleware is injected by the entrypoint and must not be replaced",
    "log-config-file": "would replace the redacting logging configuration",
}
FORBIDDEN_VALUES = {
    "uvicorn-log-level": {"debug", "trace"},
}


def _norm(key: str) -> str:
    return key.strip().lstrip("-").replace("_", "-").lower()


def parse_cli(args: list[str]) -> dict[str, Any]:
    """Very small parser for `--flag`, `--flag value`, `--flag=value`, `--no-flag`."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg.split("=", 1)
                out[_norm(key)] = value
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[_norm(arg)] = args[i + 1]
                i += 1
            else:
                key = _norm(arg)
                if key.startswith("no-"):
                    out[key[3:]] = False
                else:
                    out[key] = True
        i += 1
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE


def validate(config: dict[str, Any], cli: dict[str, Any], env: dict[str, str]) -> list[str]:
    problems: list[str] = []
    merged = {_norm(k): v for k, v in config.items()}
    merged.update(cli)

    for key, reason in FORBIDDEN_TRUE.items():
        if key in merged and _truthy(merged[key]):
            problems.append(f"option '{key}' is forbidden: {reason}")
    for key, reason in FORBIDDEN_PRESENT.items():
        if key in merged:
            problems.append(f"option '{key}' is forbidden: {reason}")
    for key, bad in FORBIDDEN_VALUES.items():
        if key in merged and str(merged[key]).lower() in bad:
            problems.append(f"option '{key}={merged[key]}' is forbidden: debug/trace output can contain request data")
    if "kv-cache-memory-bytes" not in merged and not env.get("VLLM_CPU_KVCACHE_SPACE"):
        problems.append("set 'kv-cache-memory-bytes' explicitly: on CPU vLLM sizes the KV cache from the node's "
                        "total memory, not from the pod limit, and the pod would be OOM-killed")

    if _truthy(env.get("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "")):
        problems.append("env VLLM_DEBUG_LOG_API_SERVER_RESPONSE is forbidden: logs full response bodies")
    if env.get("VLLM_LOGGING_LEVEL", "INFO").upper() == "DEBUG":
        problems.append("env VLLM_LOGGING_LEVEL=DEBUG is forbidden: the request logger prints prompts at DEBUG")
    if env.get("VLLM_CONFIGURE_LOGGING", "1") == "0":
        problems.append("env VLLM_CONFIGURE_LOGGING=0 is forbidden: it bypasses the redacting logging config")
    if env.get("VLLM_LOGGING_CONFIG_PATH") != BAKED_LOGGING_CONFIG:
        problems.append(f"env VLLM_LOGGING_CONFIG_PATH must be {BAKED_LOGGING_CONFIG} (baked into the image)")
    if _truthy(env.get("VLLM_SERVER_DEV_MODE", "")):
        problems.append("env VLLM_SERVER_DEV_MODE is forbidden: exposes /collective_rpc, /sleep, /reset_* ...")
    if env.get("HF_HUB_OFFLINE") != "1":
        problems.append("env HF_HUB_OFFLINE must be 1: the serving container never downloads (initContainer does)")
    if env.get("VLLM_NO_USAGE_STATS") != "1":
        problems.append("env VLLM_NO_USAGE_STATS must be 1: no outbound telemetry from the serving container")
    api_key = env.get("VLLM_API_KEY", "")
    if len(api_key) < MIN_API_KEY_LEN:
        problems.append(f"env VLLM_API_KEY must be set and at least {MIN_API_KEY_LEN} characters "
                        "(generate with: openssl rand -hex 32)")
    return problems


def preload_libraries() -> list[str]:
    arch = platform.machine()
    candidates = [f"/usr/lib/{arch}-linux-gnu/libtcmalloc_minimal.so.4"]
    if arch == "x86_64":
        candidates += sorted(glob.glob(f"{sys.prefix}/lib/libiomp5.so"))
    return [c for c in candidates if os.path.exists(c)]


def main(argv: list[str]) -> int:
    config_path = os.environ.get("VLLM_CONFIG_FILE", "/etc/vllm/config.yaml")
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        _emit("guard_refused", problems=[f"config file not found: {config_path}"])
        return EX_CONFIG
    if not isinstance(config, dict):
        _emit("guard_refused", problems=[f"{config_path} must be a YAML mapping of vLLM options"])
        return EX_CONFIG

    problems = validate(config, parse_cli(argv), dict(os.environ))
    if problems:
        _emit("guard_refused", config=config_path, problems=problems)
        return EX_CONFIG

    env = dict(os.environ)
    preload = preload_libraries()
    if preload:
        env["LD_PRELOAD"] = ":".join(preload + [p for p in env.get("LD_PRELOAD", "").split(":") if p])
    cmd = ["vllm", "serve", "--config", config_path, "--middleware", MIDDLEWARE,
           "--disable-uvicorn-access-log", *argv]
    _emit("guard_start", config=config_path, preload=preload,
          options=sorted(_norm(k) for k in config), extra_args=len(argv))
    os.execvpe(cmd[0], cmd, env)
    return 0  # unreachable


def _emit(event: str, **fields: Any) -> None:
    stream = sys.stderr if event == "guard_refused" else sys.stdout
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), file=stream, flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
