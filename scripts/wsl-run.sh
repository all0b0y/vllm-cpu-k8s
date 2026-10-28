#!/usr/bin/env bash
# One-shot run on the Windows machine (inside WSL2 Ubuntu, Docker Desktop with WSL integration).
# Does everything the Mac run did and packs the raw results into one archive to bring back:
#   preflight -> build -> secrets -> cluster -> deploy -> tests -> bench -> pack
#
#   bash scripts/wsl-run.sh                 # all steps
#   bash scripts/wsl-run.sh bench pack      # only the given steps (e.g. after a fix)
#
# Everything is logged to results/wsl-run-<time>.log. Steps are idempotent: re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
MINIKUBE_VERSION=v1.39.0
KUBECTL_VERSION=v1.37.1
OVERLAY=k8s/overlays/win-i5
STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(preflight build secrets cluster deploy tests bench pack)
mkdir -p results
LOG="results/wsl-run-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n========== %s  [%s]\n' "$1" "$(date +%T)"; }
die() { printf 'STOP: %s\n' "$*"; exit 1; }

install_bin() {  # name url sha256-url
  local name=$1 url=$2 sumurl=$3
  command -v "$name" >/dev/null && return 0
  mkdir -p "$HOME/.local/bin"
  echo "installing $name from $url"
  curl -fsSLo "/tmp/$name" "$url"
  echo "$(curl -fsSL "$sumurl" | awk '{print $1}')  /tmp/$name" | sha256sum -c -
  install -m 0755 "/tmp/$name" "$HOME/.local/bin/$name"
}

preflight() {
  step preflight
  grep -qi microsoft /proc/version || echo "note: not running under WSL"
  for c in git make curl openssl python3 awk; do command -v $c >/dev/null || die "missing $c (sudo apt-get install -y $c)"; done
  docker info >/dev/null 2>&1 || die "docker not reachable: Docker Desktop -> Settings -> Resources -> WSL integration -> enable Ubuntu-24.04"
  install_bin minikube "https://github.com/kubernetes/minikube/releases/download/$MINIKUBE_VERSION/minikube-linux-amd64" \
    "https://github.com/kubernetes/minikube/releases/download/$MINIKUBE_VERSION/minikube-linux-amd64.sha256"
  install_bin kubectl "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl" \
    "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl.sha256"
  local ncpu mem
  ncpu=$(docker info --format '{{.NCPU}}'); mem=$(docker info --format '{{.MemTotal}}')
  echo "docker: $ncpu CPUs, $((mem / 1024 / 1024)) MiB"
  (( ncpu >= 16 )) || die "Docker sees $ncpu CPUs, need 16: C:\\Users\\<you>\\.wslconfig -> processors=16, then 'wsl --shutdown'"
  (( mem >= 10900000000 )) || die "Docker sees $((mem / 1024 / 1024)) MiB, need ~11 GB: .wslconfig -> memory=11GB, then 'wsl --shutdown'"
  echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
  echo "ISA flags: $(grep -m1 -o -w -E 'avx2|avx512f|avx512_bf16|avx_vnni|amx_bf16' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  free -g | head -2
  df -h . | tail -1
  minikube version --short; kubectl version --client | head -1
}

build() {
  step build
  make build
  docker images | grep -E 'vllm-cpu-guard|model-loader'
}

secrets() {
  step secrets
  if [[ -s k8s/base/secrets/api-key.txt && -e k8s/base/secrets/hf-token.txt ]]; then
    bash scripts/make-secrets.sh --check
  else
    bash scripts/make-secrets.sh --no-hf   # public Qwen only; run without --no-hf for gated/private models
  fi
}

cluster() {
  step cluster
  if minikube status >/dev/null 2>&1; then
    echo "minikube already running"
  else
    CPUS=16 MEMORY=10g RESERVED=0-1 bash scripts/minikube-up.sh
  fi
  minikube image load vllm-cpu-guard:0.31.0
  minikube image load model-loader:1
  kubectl get node -o jsonpath='{.items[0].status.allocatable}'; echo
}

deploy() {
  step deploy
  kubectl apply -k "$OVERLAY"
  sleep 10
  echo "first start downloads ~3.1 GB into the PVC; watching the initContainer..."
  kubectl -n llm rollout status deploy/vllm --timeout=40m
  kubectl -n llm logs deploy/vllm -c model-fetch | tail -2
  minikube ssh -- sudo cat /var/lib/kubelet/cpu_manager_state | tr -d '\r'; echo
}

tests() {
  step tests
  OVERLAY=$OVERLAY OUT=results/k8s-win bash scripts/k8s-test.sh all
  kubectl apply -k "$OVERLAY" >/dev/null   # back to the normal overlay (egress allowed, same pod spec)
  kubectl -n llm delete networkpolicy allow-model-fetch-egress --ignore-not-found   # keep it offline for bench
  kubectl -n llm rollout status deploy/vllm --timeout=20m
}

bench() {
  step bench
  LABEL="baseline qwen2.5-1.5b bf16, 10 cores (9 OMP), KV 1.5GiB, nginx" MACHINE=win-i5 \
    python3 -u scripts/bench.py --tier main --repeats 3 --idle 20
  MACHINE=win-i5 python3 -u scripts/bench.py --tier phases --repeats 2 --idle 10
  MACHINE=win-i5 python3 -u scripts/bench.py --tier cache --repeats 2 --idle 10
}

pack() {
  step pack
  local out
  out="results/win-i5-$(date +%Y%m%d-%H%M).tar.gz"
  tar czf "$out" results/raw/*win-i5* results/k8s-win results/wsl-run-*.log 2>/dev/null || true
  ls -la "$out"
  echo "copy to Windows: cp $out /mnt/c/Users/\$(cmd.exe /c echo %USERNAME% 2>/dev/null | tr -d '\r')/Desktop/"
}

for s in "${STEPS[@]}"; do "$s"; done
step "finished: ${STEPS[*]}"
