"""vLLM engine-state sampler (runs inside the benchmark client pod, role=client).

Scrapes /metrics through the edge proxy with the API key every INTERVAL seconds and keeps gauges, counters and
histogram _sum/_count (no buckets): running/waiting requests, KV-cache usage, prefix-cache queries/hits,
prompt/generation token counters. JSON lines on stdout.
Usage: python3 vllm_metrics.py BASE_URL INTERVAL_SECONDS   (API key from $OPENAI_API_KEY)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

LINE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+?)(_bucket)?(\{[^}]*\})?\s+([0-9.eE+-]+|NaN)$")


def scrape(url: str, key: str) -> dict[str, float]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if not m or m.group(2):  # skip comments and histogram buckets
            continue
        name, value = m.group(1), m.group(4)
        if name.endswith("_created"):
            continue
        out[name] = out.get(name, 0.0) + (float(value) if value != "NaN" else 0.0)  # sum over label sets
    return out


def main() -> None:
    base, interval = sys.argv[1].rstrip("/"), float(sys.argv[2])
    key = os.environ["OPENAI_API_KEY"]
    while True:
        started = time.time()
        try:
            rec = {"ts": round(started, 3), **scrape(base + "/metrics", key)}
        except Exception as exc:  # noqa: BLE001 - keep sampling through restarts
            rec = {"ts": round(started, 3), "error": type(exc).__name__}
        print(json.dumps(rec), flush=True)
        time.sleep(max(0.0, interval - (time.time() - started)))


if __name__ == "__main__":
    main()
