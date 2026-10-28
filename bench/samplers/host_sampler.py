"""Host sampler (runs on the machine that hosts the Docker/WSL VM). JSON lines on stdout, one per interval.

macOS : total CPU busy % and memory (top), CPU % of the Virtualization.framework VM process (= Docker Desktop VM),
        GPU utilisation % (ioreg IOAccelerator) -- GPU is a control metric only: inference must not touch it.
Linux/WSL: CPU busy % and memory from /proc (inside WSL this is the WSL VM that runs Docker), GPU via nvidia-smi.
        Under WSL also the WINDOWS host CPU % / free memory: one long-lived powershell.exe loop reads
        locale-independent WMI classes (perf counter names are localised, WMI class names are not).
Usage: python3 host_sampler.py INTERVAL_SECONDS
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time


def run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def mac_sample() -> dict:
    out: dict = {}
    top = run(["top", "-l", "2", "-n", "0", "-s", "1"])  # first sample is since boot -> use the last one
    cpu = re.findall(r"CPU usage: ([\d.]+)% user, ([\d.]+)% sys, ([\d.]+)% idle", top)
    if cpu:
        user, sys_, idle = map(float, cpu[-1])
        out.update(host_cpu_busy_pct=round(100 - idle, 1), host_cpu_user_pct=user, host_cpu_sys_pct=sys_)
    mem = re.findall(r"PhysMem: (\d+)([MG]) used .*?, (\d+)([MG]) unused", top)
    if mem:
        u, uu, f, fu = mem[-1]
        to_gb = lambda v, unit: float(v) / (1024 if unit == "M" else 1)  # noqa: E731
        out.update(host_mem_used_gb=round(to_gb(u, uu), 2), host_mem_free_gb=round(to_gb(f, fu), 2))
    for line in run(["ps", "-A", "-o", "%cpu=,rss=,comm="]).splitlines():
        if "Virtualization.VirtualMachine" in line:
            parts = line.split(None, 2)
            out.update(vm_cpu_pct=float(parts[0]), vm_rss_gb=round(int(parts[1]) / 1048576, 2))
    io = run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
    gpu = re.search(r'"Device Utilization %"=(\d+)', io)
    if gpu:
        out["gpu_util_pct"] = int(gpu.group(1))
    return out


_prev: tuple[int, int] | None = None


def linux_sample() -> dict:
    global _prev
    out: dict = {}
    with open("/proc/stat") as fh:
        v = list(map(int, fh.readline().split()[1:]))
    idle, total = v[3] + v[4], sum(v[:8])
    if _prev:
        dt, di = total - _prev[1], idle - _prev[0]
        out["host_cpu_busy_pct"] = round(100 * (1 - di / dt), 1) if dt else 0.0
    _prev = (idle, total)
    mem = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, val = line.split(":")
            mem[k] = int(val.split()[0])
    out.update(host_mem_used_gb=round((mem["MemTotal"] - mem["MemAvailable"]) / 1048576, 2),
               host_mem_free_gb=round(mem["MemAvailable"] / 1048576, 2))
    wsl_smi = "/usr/lib/wsl/lib/nvidia-smi"
    smi = shutil.which("nvidia-smi") or (wsl_smi if os.path.exists(wsl_smi) else None)
    if smi:
        q = run([smi, "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"])
        if q.strip():
            util, mem_used, power = [x.strip() for x in q.splitlines()[0].split(",")]
            out.update(gpu_util_pct=float(util), gpu_mem_used_mb=float(mem_used),
                       gpu_power_w=float(power) if power not in ("[N/A]", "") else None)
    return out


_win: dict = {}

PS_LOOP = r"""
while ($true) {
  $c = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
  $m = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes
  Write-Output "$c $m"
  Start-Sleep -Seconds INTERVAL
}
"""


def start_windows_reader(interval: float) -> None:
    ps = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not os.path.exists(ps) and not shutil.which("powershell.exe"):
        return
    proc = subprocess.Popen([ps, "-NoProfile", "-NonInteractive", "-Command",
                             PS_LOOP.replace("INTERVAL", str(max(1, int(interval))))],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            parts = line.split()
            if len(parts) == 2:
                try:
                    _win.update(win_cpu_busy_pct=float(parts[0]), win_mem_free_gb=round(float(parts[1]) / 1024, 2))
                except ValueError:
                    pass

    threading.Thread(target=reader, daemon=True).start()


def main() -> None:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    sample = mac_sample if platform.system() == "Darwin" else linux_sample
    if "microsoft" in platform.release().lower():
        start_windows_reader(interval)
    while True:
        started = time.time()
        rec = {"ts": round(started, 3), **sample(), **_win}
        print(json.dumps(rec), flush=True)
        time.sleep(max(0.0, interval - (time.time() - started)))


if __name__ == "__main__":
    main()
