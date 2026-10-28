#!/usr/bin/env bash
# Start the local cluster with the settings the deployment relies on:
#   * Calico CNI           -> NetworkPolicy is actually enforced (the default CNI ignores it)
#   * CPU Manager "static" -> Guaranteed pods with integer CPUs get exclusive cores (no CFS throttling)
#   * reserved CPUs        -> system pods, Calico, nginx and the benchmark client share these cores
# Usage: CPUS=8  MEMORY=10g RESERVED=0   scripts/minikube-up.sh   (Mac M3 defaults)
#        CPUS=16 MEMORY=10g RESERVED=0-1 scripts/minikube-up.sh   (Windows i5-13400F)
# Exclusive cores available to pods = CPUS - |RESERVED| - system pod requests (~1 CPU).
set -euo pipefail
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-10g}
RESERVED=${RESERVED:-0}
K8S_VERSION=${K8S_VERSION:-v1.37.0}

minikube start \
  --driver=docker \
  --kubernetes-version="$K8S_VERSION" \
  --cpus="$CPUS" --memory="$MEMORY" \
  --cni=calico \
  --extra-config=kubelet.cpu-manager-policy=static \
  --extra-config=kubelet.reserved-cpus="$RESERVED"

kubectl -n kube-system rollout status daemonset/calico-node --timeout=300s
kubectl get nodes -o wide
