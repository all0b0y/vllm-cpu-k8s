#!/usr/bin/env bash
# Runs ON the minikube node (root). Samples cgroup v2 counters of the given cgroups every INTERVAL seconds.
# Usage: node_cgroup.sh INTERVAL label=/sys/fs/cgroup/... [label=...]
# Output TSV: ts label cpu_usec user_usec system_usec throttled_usec mem_current anon file
#   cpu_*      cumulative CPU time (usage_usec), deltas give cores used
#   throttled  cumulative CFS throttling (should stay ~0 for exclusive cores)
#   mem_*      bytes; anon = process memory (weights, KV cache, activations), file = page cache
set -u
INT=$1; shift
printf 'ts\tlabel\tcpu_usec\tuser_usec\tsystem_usec\tthrottled_usec\tmem_current\tanon\tfile\n'
while :; do
  ts=$(date +%s.%N)
  for kv in "$@"; do
    label=${kv%%=*}; d=${kv#*=}
    [ -f "$d/cpu.stat" ] || continue
    awk -v ts="$ts" -v label="$label" -v cur="$(cat "$d/memory.current")" '
      FILENAME ~ /cpu.stat$/ { c[$1] = $2 }
      FILENAME ~ /memory.stat$/ { m[$1] = $2 }
      END { printf "%s\t%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n", ts, label, c["usage_usec"], c["user_usec"],
            c["system_usec"], c["throttled_usec"], cur, m["anon"], m["file"] }' "$d/cpu.stat" "$d/memory.stat"
  done
  # whole node (all pods) for context
  awk -v ts="$ts" '/^cpu /{ printf "%s\tnode_procstat\t%.0f\t%.0f\t%.0f\t0\t0\t0\t0\n", ts, ($2+$3+$4+$6+$7+$8)*10000, ($2+$3)*10000, $4*10000 }' /proc/stat
  sleep "$INT"
done
