"""vllm_guard: privacy-preserving logging, access control and a fail-closed entrypoint for vLLM.

Plugged into vLLM only through official extension points:
  * VLLM_LOGGING_CONFIG_PATH  -> vllm_guard.logfilter (JSON formatter + redaction filter)
  * vllm serve --middleware   -> vllm_guard.middleware.GuardMiddleware
  * container ENTRYPOINT      -> vllm_guard.entrypoint (config validation, then exec `vllm serve`)
"""
