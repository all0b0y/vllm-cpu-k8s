#!/usr/bin/env bash
# Sample memory of every container of the vLLM pod from the node's cgroup v2 files.
# Usage: scripts/cgmem.sh SECONDS INTERVAL > file.tsv
set -euo pipefail
DUR=${1:-120}; INT=${2:-2}
UID_=$(kubectl -n llm get pod -l app=vllm -o jsonpath='{.items[0].metadata.uid}')
minikube ssh -- "end=\$((\$(date +%s)+$DUR)); P=/sys/fs/cgroup/kubepods/pod$UID_;
echo -e 'ts\tcg\tcur_mib\tpeak_mib\tanon_mib\tfile_mib\tshmem_mib';
while [ \$(date +%s) -lt \$end ]; do
  for d in \$P \$P/*/; do [ -f \$d/memory.current ] || continue;
    echo -e \"\$(date +%s)\t\$(basename \$d | cut -c1-12)\t\$((\$(cat \$d/memory.current)/1048576))\t\$((\$(cat \$d/memory.peak)/1048576))\t\$(awk '/^anon /{a=\$2}/^file /{f=\$2}/^shmem /{s=\$2}END{printf \"%d\t%d\t%d\", a/1048576, f/1048576, s/1048576}' \$d/memory.stat)\";
  done; sleep $INT; done" | tr -d '\r'
