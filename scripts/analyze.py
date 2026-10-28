#!/usr/bin/env python3
"""Turn raw benchmark runs (results/raw/<run>/) into per-phase tables and timeline plots.

For every phase (scenario repetition) the sampler timelines are cut to [start, end] and joined with the
`vllm bench serve` result:
  latency     TTFT / TPOT / ITL / E2E (p50, p90, p99)                  -- from vllm bench serve
  throughput  output tok/s, total tok/s, requests/s                     -- from vllm bench serve
  CPU         vLLM container cores used (mean, p95), CFS throttling     -- cgroup cpu.stat deltas
  memory      vLLM anon (weights + KV + activations) mean/peak, page cache
  engine      KV-cache usage peak, max running / waiting requests, prefix-cache hit rate in the phase
  host        host CPU busy %, Docker-VM CPU %, GPU utilisation (minus the idle baseline)
  own metric  CPU-seconds per 1000 tokens (vLLM container CPU time / processed tokens)

Usage: scripts/analyze.py results/raw/<run> [more runs...] [--plots] [--out results/summary]
Plots need matplotlib:  uv run --with matplotlib python scripts/analyze.py ... --plots
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_node(path: Path) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = {}
    if not path.exists():
        return series
    with path.open() as fh:
        for row in csv.DictReader((line.replace("\r", "") for line in fh), delimiter="\t"):
            if row.get("ts") in (None, "ts"):
                continue
            try:
                rec = {k: float(v) for k, v in row.items() if k != "label"}
            except (TypeError, ValueError):
                continue
            series.setdefault(row["label"], []).append(rec)
    return series


def window(rows: list[dict], start: float, end: float) -> list[dict]:
    return [r for r in rows if start <= r["ts"] <= end]


def rate_series(rows: list[dict], key: str, scale: float) -> list[float]:
    """Per-interval rate of a cumulative counter (e.g. cores = d(cpu_usec)/dt/1e6)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        dt = b["ts"] - a["ts"]
        if dt > 0:
            out.append((b[key] - a[key]) / dt / scale)
    return out


def delta(rows: list[dict], key: str) -> float:
    return rows[-1][key] - rows[0][key] if len(rows) >= 2 else 0.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def analyze_run(run: Path) -> list[dict]:
    meta = json.loads((run / "meta.json").read_text())
    phases = load_jsonl(run / "phases.jsonl")
    node = load_node(run / "node.tsv")
    host = load_jsonl(run / "host.jsonl")
    eng = [r for r in load_jsonl(run / "vllm.jsonl") if "error" not in r]
    idle = next((p for p in phases if p["phase"] == "idle"), None)
    base = window(host, idle["start"], idle["end"]) if idle else []
    gpu_idle = mean([r["gpu_util_pct"] for r in base if "gpu_util_pct" in r])
    cpu_idle = mean([r["host_cpu_busy_pct"] for r in base if "host_cpu_busy_pct" in r])

    rows = []
    for ph in phases:
        rec: dict = {"run": run.name, "machine": meta["machine"], "phase": ph["phase"],
                     "scenario": ph.get("scenario", ph["phase"]), "repeat": ph.get("repeat", 0),
                     "duration_s": round(ph["end"] - ph["start"], 1)}
        res_file = run / f"{ph['phase']}.json"
        if res_file.exists():
            r = json.loads(res_file.read_text())
            rec.update(input_len=ph.get("in"), output_len=ph.get("out"), prefix_len=ph.get("prefix", 0),
                       concurrency=ph.get("conc"), requests=r.get("completed"), failed=r.get("failed", 0))
            for m in ("ttft", "tpot", "itl", "e2el"):
                for q in ("median", "p90", "p99"):
                    key = f"{q}_{m}_ms" if q == "median" else f"{q}_{m}_ms"
                    if key in r:
                        rec[f"{m}_{'p50' if q == 'median' else q}_ms"] = round(r[key], 1)
            rec.update(out_tok_s=round(r["output_throughput"], 2), total_tok_s=round(r["total_token_throughput"], 2),
                       req_s=round(r["request_throughput"], 4))
            rec.update(input_tokens=r.get("total_input_tokens"), output_tokens=r.get("total_output_tokens"),
                       bench_duration_s=round(r.get("duration", 0), 1))
        start, end = ph["start"], ph["end"]
        v = window(node.get("vllm", []), start, end)
        if len(v) >= 2:
            cores = rate_series(v, "cpu_usec", 1e6)
            cpu_s = delta(v, "cpu_usec") / 1e6
            rec.update(vllm_cores_mean=round(mean(cores), 2), vllm_cores_p95=round(pct(cores, 95), 2),
                       vllm_cpu_s=round(cpu_s, 1), vllm_throttled_ms=round(delta(v, "throttled_usec") / 1e3, 1),
                       vllm_sys_share=round(delta(v, "system_usec") / max(1.0, delta(v, "cpu_usec")), 3),
                       vllm_anon_gib_mean=round(mean([x["anon"] for x in v]) / 2**30, 2),
                       vllm_anon_gib_peak=round(max(x["anon"] for x in v) / 2**30, 2),
                       vllm_pagecache_gib=round(mean([x["file"] for x in v]) / 2**30, 2))
            rec["_cpu_s"] = cpu_s
        for label, key in (("edge-proxy", "proxy_cores_mean"), ("bench_client", "client_cores_mean")):
            w = window(node.get(label, []), start, end)
            if len(w) >= 2:
                rec[key] = round(mean(rate_series(w, "cpu_usec", 1e6)), 3)
        n = window(node.get("node_procstat", []), start, end)
        if len(n) >= 2:
            rec["vm_cores_mean"] = round(mean(rate_series(n, "cpu_usec", 1e6)), 2)
        h = window(host, start, end)
        if h:
            rec["host_cpu_busy_pct"] = round(mean([x["host_cpu_busy_pct"] for x in h if "host_cpu_busy_pct" in x]), 1)
            rec["host_cpu_over_idle_pct"] = round(rec["host_cpu_busy_pct"] - cpu_idle, 1)
            gpu = [x["gpu_util_pct"] for x in h if "gpu_util_pct" in x]
            if gpu:
                rec["gpu_util_pct"] = round(mean(gpu), 1)
                rec["gpu_over_idle_pct"] = round(mean(gpu) - gpu_idle, 1)
        e = window(eng, start, end)
        if e:
            rec["kv_usage_peak_pct"] = round(100 * max(x.get("vllm:kv_cache_usage_perc", 0) for x in e), 1)
            rec["running_max"] = int(max(x.get("vllm:num_requests_running", 0) for x in e))
            rec["waiting_max"] = int(max(x.get("vllm:num_requests_waiting", 0) for x in e))
            q = delta(e, "vllm:prefix_cache_queries_total") if len(e) >= 2 else 0
            if q > 0:
                rec["prefix_hit_rate_pct"] = round(100 * delta(e, "vllm:prefix_cache_hits_total") / q, 1)
            rec["preemptions"] = int(delta(e, "vllm:num_preemptions_total")) if len(e) >= 2 else 0
            # Own metric: CPU-seconds of the vLLM container per 1000 tokens the ENGINE processed in the same
            # window (engine counters include the warm-up request, so numerator and denominator match).
            if len(e) >= 2 and "_cpu_s" in rec:
                p_tok = delta(e, "vllm:prompt_tokens_total")
                g_tok = delta(e, "vllm:generation_tokens_total")
                rec.update(engine_prompt_tokens=int(p_tok), engine_generated_tokens=int(g_tok))
                if p_tok + g_tok > 0:
                    rec["cpu_s_per_1k_tok"] = round(rec["_cpu_s"] / ((p_tok + g_tok) / 1000), 2)
                if g_tok > 0:
                    rec["cpu_s_per_1k_out_tok"] = round(rec["_cpu_s"] / (g_tok / 1000), 2)
        rec.pop("_cpu_s", None)
        rows.append(rec)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Median over repeats per (machine, scenario)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["phase"] != "idle" and "out_tok_s" in r:
            groups.setdefault((r["machine"], r["scenario"]), []).append(r)
    out = []
    for (machine, scenario), rs in groups.items():
        agg: dict = {"machine": machine, "scenario": scenario, "repeats": len(rs)}
        for k in rs[0]:
            vals = [x[k] for x in rs if isinstance(x.get(k), (int, float))]
            if vals and k not in ("repeat",):
                agg[k] = round(statistics.median(vals), 2)
        if len(rs) > 1 and "out_tok_s" in agg:
            agg["out_tok_s_spread_pct"] = round(100 * (max(x["out_tok_s"] for x in rs) - min(
                x["out_tok_s"] for x in rs)) / agg["out_tok_s"], 1)
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


MD_COLS = [("scenario", "Сценарий"), ("input_len", "Вход, ток"), ("output_len", "Выход, ток"),
           ("concurrency", "Конк."), ("ttft_p50_ms", "TTFT p50, мс"), ("ttft_p99_ms", "TTFT p99, мс"),
           ("tpot_p50_ms", "TPOT p50, мс"), ("e2el_p50_ms", "Latency p50, мс"), ("out_tok_s", "Выход, ток/с"),
           ("total_tok_s", "Всего, ток/с"), ("req_s", "Запр/с"), ("vllm_cores_mean", "CPU, ядер"),
           ("vllm_anon_gib_peak", "RAM пик, GiB"), ("cpu_s_per_1k_tok", "CPU-с / 1k ток"),
           ("gpu_over_idle_pct", "GPU над фоном, п.п.")]


def write_md(path: Path, agg: list[dict]) -> None:
    lines = []
    for machine in sorted({a["machine"] for a in agg}):
        lines += [f"### {machine}", "", "| " + " | ".join(h for _, h in MD_COLS) + " |",
                  "|" + "---|" * len(MD_COLS)]
        for a in [x for x in agg if x["machine"] == machine]:
            lines.append("| " + " | ".join(str(a.get(k, "")) for k, _ in MD_COLS) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def plot_run(run: Path, outdir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = load_jsonl(run / "phases.jsonl")
    node = load_node(run / "node.tsv")
    host = load_jsonl(run / "host.jsonl")
    eng = [r for r in load_jsonl(run / "vllm.jsonl") if "error" not in r]
    t0 = phases[0]["start"]
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    v = node.get("vllm", [])
    if len(v) > 1:
        ax[0].plot([b["ts"] - t0 for b in v[1:]], rate_series(v, "cpu_usec", 1e6), label="vLLM container")
    n = node.get("node_procstat", [])
    if len(n) > 1:
        ax[0].plot([b["ts"] - t0 for b in n[1:]], rate_series(n, "cpu_usec", 1e6), label="whole VM", alpha=0.5)
    ax[0].set_ylabel("CPU, cores")
    if v:
        ax[1].plot([x["ts"] - t0 for x in v], [x["anon"] / 2**30 for x in v], label="vLLM anon")
        ax[1].plot([x["ts"] - t0 for x in v], [x["file"] / 2**30 for x in v], label="page cache")
    ax[1].set_ylabel("RAM, GiB")
    if eng:
        ax[2].plot([x["ts"] - t0 for x in eng], [100 * x.get("vllm:kv_cache_usage_perc", 0) for x in eng],
                   label="KV cache used, %")
        ax[2].plot([x["ts"] - t0 for x in eng], [x.get("vllm:num_requests_running", 0) for x in eng],
                   label="running requests")
        ax[2].plot([x["ts"] - t0 for x in eng], [x.get("vllm:num_requests_waiting", 0) for x in eng],
                   label="waiting requests")
    ax[2].set_ylabel("engine")
    hx = [x for x in host if "host_cpu_busy_pct" in x]
    ax[3].plot([x["ts"] - t0 for x in hx], [x["host_cpu_busy_pct"] for x in hx], label="host CPU busy, %")
    gx = [x for x in host if "gpu_util_pct" in x]
    if gx:
        ax[3].plot([x["ts"] - t0 for x in gx], [x["gpu_util_pct"] for x in gx], label="GPU util, % (control)")
    ax[3].set_ylabel("host, %")
    ax[3].set_xlabel("time, s")
    for a in ax:
        for i, ph in enumerate(phases):
            a.axvspan(ph["start"] - t0, ph["end"] - t0, color=f"C{i % 10}", alpha=0.07)
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    for ph in phases:
        ax[0].text((ph["start"] + ph["end"]) / 2 - t0, ax[0].get_ylim()[1] * 0.92, ph["phase"], ha="center",
                   fontsize=6, rotation=90)
    fig.suptitle(run.name)
    fig.tight_layout()
    out = outdir / f"{run.name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "summary")
    ap.add_argument("--name", default=None, help="output file prefix (default: last run name)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for run in args.runs for r in analyze_run(run)]
    agg = aggregate(rows)
    name = args.name or args.runs[-1].name
    write_csv(args.out / f"{name}-phases.csv", rows)
    write_csv(args.out / f"{name}-summary.csv", agg)
    write_md(args.out / f"{name}-summary.md", agg)
    print((args.out / f"{name}-summary.md").read_text())
    if args.plots:
        for run in args.runs:
            print("plot:", plot_run(run, args.out))


if __name__ == "__main__":
    main()
