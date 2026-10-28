#!/usr/bin/env python3
"""Benchmark orchestrator: runs `vllm bench serve` scenarios from an in-cluster client pod (role=client) while
three samplers record resource usage on one time axis:

  node   cgroup v2 counters of the vLLM container / nginx / client (CPU time, throttling, anon + page cache)
  host   host CPU / memory, Docker-VM process, GPU utilisation (control metric: must stay at background level)
  vllm   engine state from /metrics: running/waiting requests, KV-cache usage, prefix-cache hits, token counters

Every scenario repetition is a *phase* (start/end timestamps in phases.jsonl) so the analysis can cut the sampler
timelines per phase. Results: results/raw/<run-id>/ (local only), summaries via scripts/analyze.py.

Usage:
  scripts/bench.py --tier smoke                       # 1-minute check of the whole pipeline
  scripts/bench.py --tier main --repeats 3            # short/long prompts at several concurrencies
  scripts/bench.py --tier cache                       # prefix cache: shared prefix vs unique prompts
  scripts/bench.py --list                             # show scenarios
Env: NS (llm), MODEL_NAME, MODEL_PATH (tokenizer inside the PVC), IMAGE (client image), LABEL (free text tag).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = os.environ.get("NS", "llm")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5-1.5b-instruct")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen--Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
IMAGE = os.environ.get("IMAGE", "vllm-cpu-guard:0.31.0")
BASE_URL = f"http://vllm.{NS}.svc:8000"
CLIENT = "bench-client"

# name -> (dataset args, concurrency, prompts per concurrency slot)
# Case A = many short requests (128 -> 128), case B = few long-context requests (4096/8192 -> 128).
SCENARIOS: dict[str, list[dict]] = {
    "smoke": [
        {"name": "short_c1", "in": 128, "out": 32, "conc": 1, "n": 3},
    ],
    "main": [
        {"name": "short_c1", "in": 128, "out": 128, "conc": 1, "n": 4},
        {"name": "short_c4", "in": 128, "out": 128, "conc": 4, "n": 12},
        {"name": "short_c16", "in": 128, "out": 128, "conc": 16, "n": 32},
        {"name": "long_c1", "in": 4096, "out": 128, "conc": 1, "n": 3},
        {"name": "long_c2", "in": 4096, "out": 128, "conc": 2, "n": 4},
        {"name": "long_c4", "in": 4096, "out": 128, "conc": 4, "n": 8},
        {"name": "xlong_c1", "in": 8192, "out": 128, "conc": 1, "n": 2},
    ],
    # Prefill-only and decode-only workloads isolate the cost of each phase (CPU-seconds per 1000 tokens).
    "phases": [
        {"name": "prefill_only", "in": 4096, "out": 1, "conc": 1, "n": 6},
        {"name": "decode_only", "in": 16, "out": 256, "conc": 1, "n": 3},
        {"name": "decode_c8", "in": 16, "out": 256, "conc": 8, "n": 16},
    ],
    # Same total prompt length; with a shared 3584-token prefix the prefix cache serves most of the prefill.
    "cache": [
        {"name": "unique_4096", "in": 4096, "out": 64, "conc": 1, "n": 6},
        {"name": "shared_prefix_3584+512", "prefix": 3584, "in": 512, "out": 64, "conc": 1, "n": 6},
    ],
}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def kubectl(*args: str, **kw) -> str:
    return sh(["kubectl", "-n", NS, *args], **kw).stdout


def now() -> float:
    return round(time.time(), 3)


def api_secret() -> str:
    return kubectl("get", "secret", "-l", "app.kubernetes.io/component=api-key",
                   "-o", "jsonpath={.items[0].metadata.name}")


def ensure_client() -> None:
    cm = sh(["kubectl", "-n", NS, "create", "configmap", "bench-samplers",
             f"--from-file={ROOT}/bench/samplers/vllm_metrics.py", "--dry-run=client", "-o", "yaml"]).stdout
    sh(["kubectl", "apply", "-f", "-"], input=cm)
    pod = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": CLIENT, "namespace": NS, "labels": {"role": "client", "app": "bench-client"}},
        "spec": {
            "restartPolicy": "Never", "automountServiceAccountToken": False, "enableServiceLinks": False,
            "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001,
                                "seccompProfile": {"type": "RuntimeDefault"}},
            "containers": [{
                "name": "client", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                "command": ["sleep", "infinity"],
                "env": [{"name": "OPENAI_API_KEY",
                         "valueFrom": {"secretKeyRef": {"name": api_secret(), "key": "api-key"}}},
                        {"name": "HF_HUB_OFFLINE", "value": "1"}],
                "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]}},
                # Burstable QoS -> shared CPU pool (cores not given to vLLM); may burst to 2 cores there,
                # so the load generator never competes for vLLM's exclusive cores and is not the bottleneck.
                "resources": {"requests": {"cpu": "500m", "memory": "1Gi"},
                              "limits": {"cpu": "2", "memory": "1536Mi"}},
                "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True},
                                 {"name": "samplers", "mountPath": "/samplers", "readOnly": True},
                                 {"name": "tmp", "mountPath": "/tmp"},
                                 {"name": "cache", "mountPath": "/var/cache/vllm"}],
            }],
            "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": "models", "readOnly": True}},
                        {"name": "samplers", "configMap": {"name": "bench-samplers"}},
                        {"name": "tmp", "emptyDir": {}}, {"name": "cache", "emptyDir": {}}],
        },
    }
    existing = subprocess.run(["kubectl", "-n", NS, "get", "pod", CLIENT, "-o", "jsonpath={.status.phase}"],
                              text=True, capture_output=True).stdout
    if existing != "Running":
        subprocess.run(["kubectl", "-n", NS, "delete", "pod", CLIENT, "--ignore-not-found", "--wait"],
                       capture_output=True)
        sh(["kubectl", "apply", "-f", "-"], input=json.dumps(pod))
        kubectl("wait", "--for=condition=Ready", f"pod/{CLIENT}", "--timeout=180s")


def vllm_pod() -> dict:
    """The serving vLLM pod: Running, all containers ready, not being deleted (a Recreate rollout briefly has an
    old Terminating pod and a new Pending one)."""
    for pod in json.loads(kubectl("get", "pod", "-l", "app=vllm", "-o", "json"))["items"]:
        sts = pod["status"].get("containerStatuses", [])
        if (pod["status"].get("phase") == "Running" and "deletionTimestamp" not in pod["metadata"]
                and sts and all(c.get("ready") for c in sts)):
            return pod
    raise RuntimeError("no ready vLLM pod")


def wait_vllm_ready(timeout: str = "40m") -> None:
    """Block until the deployment is rolled out; on failure print why (pending/scheduling/OOM) and stop."""
    print("waiting for deploy/vllm to be ready...", flush=True)
    r = subprocess.run(["kubectl", "-n", NS, "rollout", "status", "deploy/vllm", f"--timeout={timeout}"],
                       text=True, capture_output=True, check=False)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        for cmd in (["get", "pods", "-o", "wide"], ["describe", "pod", "-l", "app=vllm"]):
            out = subprocess.run(["kubectl", "-n", NS, *cmd], text=True, capture_output=True, check=False).stdout
            print("\n".join(out.splitlines()[-40:]))
        sys.exit("vLLM pod is not ready -- see the pod status/events above")


def cgroup_targets() -> list[str]:
    pod = vllm_pod()
    base = f"/sys/fs/cgroup/kubepods/pod{pod['metadata']['uid']}"  # Guaranteed QoS pod
    targets = [f"vllm_pod={base}"]
    for st in pod["status"]["containerStatuses"]:
        cid = st["containerID"].split("://", 1)[1]
        targets.append(f"{st['name']}={base}/{cid}")
    client = json.loads(kubectl("get", "pod", CLIENT, "-o", "json"))
    qos = client["status"]["qosClass"].lower()
    cbase = f"/sys/fs/cgroup/kubepods/{'' if qos == 'guaranteed' else qos + '/'}pod{client['metadata']['uid']}"
    targets.append(f"bench_client={cbase}")
    return targets


class Samplers:
    def __init__(self, outdir: Path, interval: float) -> None:
        self.procs: list[subprocess.Popen] = []
        self.outdir, self.interval = outdir, interval

    def start(self) -> None:
        sh(["minikube", "cp", str(ROOT / "bench/samplers/node_cgroup.sh"), "/tmp/node_cgroup.sh"])
        targets = " ".join(shlex.quote(t) for t in cgroup_targets())
        self._spawn("node.tsv", ["minikube", "ssh", "--",
                                 f"sudo bash /tmp/node_cgroup.sh {self.interval} {targets}"])
        self._spawn("host.jsonl", [sys.executable, str(ROOT / "bench/samplers/host_sampler.py"), str(self.interval)])
        self._spawn("vllm.jsonl", ["kubectl", "-n", NS, "exec", CLIENT, "--", "/opt/venv/bin/python", "-u",
                                   "/samplers/vllm_metrics.py", BASE_URL, str(self.interval)])

    def _spawn(self, name: str, cmd: list[str]) -> None:
        fh = open(self.outdir / name, "w")  # noqa: SIM115 - closed with the process
        self.procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL, start_new_session=True))

    def stop(self) -> None:
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        subprocess.run(["minikube", "ssh", "--", "sudo pkill -f node_cgroup.sh"], capture_output=True)
        subprocess.run(["kubectl", "-n", NS, "exec", CLIENT, "--", "pkill", "-f", "vllm_metrics.py"],
                       capture_output=True)


def bench_cmd(sc: dict, tag: str, seed: int) -> list[str]:
    cmd = ["/opt/venv/bin/vllm", "bench", "serve", "--backend", "vllm", "--endpoint", "/v1/completions",
           "--base-url", BASE_URL, "--model", MODEL_NAME, "--tokenizer", MODEL_PATH,
           "--num-prompts", str(sc["n"]), "--max-concurrency", str(sc["conc"]), "--request-rate", "inf",
           "--ignore-eos", "--temperature", "0", "--seed", str(seed), "--num-warmups", "0",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--save-result", "--save-detailed", "--result-dir", "/tmp/bench", "--result-filename", f"{tag}.json",
           "--label", tag, "--disable-tqdm", "--ready-check-timeout-sec", "60"]
    if "prefix" in sc:
        cmd += ["--dataset-name", "prefix_repetition", "--prefix-repetition-prefix-len", str(sc["prefix"]),
                "--prefix-repetition-suffix-len", str(sc["in"]), "--prefix-repetition-num-prefixes", "1",
                "--prefix-repetition-output-len", str(sc["out"])]
    else:
        cmd += ["--dataset-name", "random", "--random-input-len", str(sc["in"]),
                "--random-output-len", str(sc["out"]), "--random-range-ratio", "0"]
    return cmd


def phase_seed(rep: int, idx: int) -> int:
    """Unique per (repeat, scenario): random prompts with the same seed are identical across scenarios, and the
    prefix cache would then serve the prefill of the next scenario (seen: 81% hit rate, TTFT 66 s -> 4.6 s)."""
    return rep * 1000 + idx + 1


def warmup_cmd(rep: int, idx: int) -> list[str]:
    """Tiny request with its own seed: warms the HTTP/engine path without caching any prompt of the phase.
    (`--num-warmups` in vllm bench re-sends the FIRST measured prompt, which then hits the prefix cache.)"""
    return ["/opt/venv/bin/vllm", "bench", "serve", "--backend", "vllm", "--endpoint", "/v1/completions",
            "--base-url", BASE_URL, "--model", MODEL_NAME, "--tokenizer", MODEL_PATH, "--dataset-name", "random",
            "--random-input-len", "16", "--random-output-len", "4", "--random-range-ratio", "0", "--num-prompts", "1",
            "--seed", str(900000 + phase_seed(rep, idx)), "--num-warmups", "0", "--disable-tqdm",
            "--ready-check-timeout-sec", "60"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="smoke", choices=sorted(SCENARIOS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="scenario names to run (default: whole tier)")
    ap.add_argument("--interval", type=float, default=1.0, help="sampler interval, seconds")
    ap.add_argument("--idle", type=float, default=15.0, help="idle baseline before the first phase, seconds")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for tier, scs in SCENARIOS.items():
            print(tier, ", ".join(s["name"] for s in scs))
        return 0

    machine = os.environ.get("MACHINE", "mac-m3" if platform.system() == "Darwin" else "win-i5")
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{machine}-{args.tier}"
    out = ROOT / "results" / "raw" / run_id
    wait_vllm_ready()
    out.mkdir(parents=True)
    ensure_client()
    meta = {"run_id": run_id, "machine": machine, "tier": args.tier, "model": MODEL_NAME,
            "label": os.environ.get("LABEL", ""), "repeats": args.repeats, "started": now(),
            "host_platform": platform.platform()}
    pod = vllm_pod()
    cfg_name = next(v["configMap"]["name"] for v in pod["spec"]["volumes"] if v["name"] == "vllm-config")
    meta["vllm_config"] = json.loads(kubectl("get", "cm", cfg_name, "-o", "json"))["data"]["config.yaml"]
    meta["vllm_resources"] = pod["spec"]["containers"][0]["resources"]
    meta["vllm_env"] = {e["name"]: e.get("value") for e in pod["spec"]["containers"][0].get("env", [])
                        if "value" in e}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    phases = open(out / "phases.jsonl", "w")  # noqa: SIM115
    samplers = Samplers(out, args.interval)
    samplers.start()
    print(f"run {run_id}: samplers started, idle baseline {args.idle:.0f}s", flush=True)
    try:
        t0 = now()
        time.sleep(args.idle)
        phases.write(json.dumps({"phase": "idle", "start": t0, "end": now()}) + "\n")
        scenarios = [s for s in SCENARIOS[args.tier] if not args.only or s["name"] in args.only]
        for rep in range(1, args.repeats + 1):
            for idx, sc in enumerate(scenarios):
                tag = f"{sc['name']}_r{rep}"
                subprocess.run(["kubectl", "-n", NS, "exec", CLIENT, "--", *warmup_cmd(rep, idx)],
                               text=True, capture_output=True)
                start = now()
                proc = subprocess.run(["kubectl", "-n", NS, "exec", CLIENT, "--",
                                       *bench_cmd(sc, tag, seed=phase_seed(rep, idx))],
                                      text=True, capture_output=True)
                end = now()
                (out / f"{tag}.log").write_text(proc.stdout + proc.stderr)
                ok = proc.returncode == 0
                if ok:
                    (out / f"{tag}.json").write_text(kubectl("exec", CLIENT, "--", "cat", f"/tmp/bench/{tag}.json"))
                phases.write(json.dumps({"phase": tag, "scenario": sc["name"], "repeat": rep, "start": start,
                                         "end": end, "ok": ok, **sc}) + "\n")
                phases.flush()
                summary = ""
                if ok:
                    r = json.loads((out / f"{tag}.json").read_text())
                    summary = (f"TTFT p50 {r['median_ttft_ms']:.0f} ms | E2E p50 {r['median_e2el_ms']:.0f} ms | "
                               f"out {r['output_throughput']:.1f} tok/s | total {r['total_token_throughput']:.1f} tok/s")
                print(f"[{tag}] {'ok' if ok else 'FAILED'} {end - start:.0f}s {summary}", flush=True)
                time.sleep(3)  # cool-down gap between phases on the timeline
    finally:
        samplers.stop()
        phases.close()
    print(f"done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
