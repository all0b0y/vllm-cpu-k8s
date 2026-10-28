#!/usr/bin/env bash
# Phase-1 validation in plain Docker, before Kubernetes:
#   * the same restrictions as the k8s `restricted` profile (non-root, read-only root, no capabilities)
#   * serving is offline: the vLLM container sits on an --internal network (no route to the internet)
#   * every route except /health requires the Bearer token; /invocations bypass is closed
#   * container logs contain the fact of each request but never the prompt/answer (canary check)
# Usage: scripts/docker-smoke.sh            (guarded image)
#        MODE=stock scripts/docker-smoke.sh (plain `vllm serve --api-key`, shows the bypass)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-vllm-cpu-guard:0.31.0}
MODE=${MODE:-guard}
NET=llm-internal
NAME=vllm-smoke-$MODE
OUT=${OUT:-results/smoke}
mkdir -p "$OUT"
KEY=$(openssl rand -hex 32)
CANARY="CANARY-$(openssl rand -hex 8)"

docker network inspect "$NET" >/dev/null 2>&1 || docker network create --internal "$NET" >/dev/null
docker rm -f "$NAME" >/dev/null 2>&1 || true

RESTRICTED=(--read-only --cap-drop ALL --security-opt no-new-privileges --user 10001:10001
            --tmpfs /tmp:uid=10001,gid=10001,size=1g --tmpfs /var/cache/vllm:exec,uid=10001,gid=10001,size=2g
            --shm-size 1g --cpuset-cpus "${CPUSET:-0-5}" --memory "${MEMORY:-7g}")
MOUNTS=(-v models:/models:ro -v "$PWD/docker/vllm/config.example.yaml:/etc/vllm/config.yaml:ro")

if [[ $MODE == stock ]]; then
  # Stock vLLM with its own --api-key, no guard middleware, no logging config.
  docker run -d --name "$NAME" --network "$NET" "${RESTRICTED[@]}" "${MOUNTS[@]}" \
    -e VLLM_LOGGING_CONFIG_PATH= --entrypoint vllm "$IMAGE" serve --config /etc/vllm/config.yaml --api-key "$KEY" \
    ${KV_BYTES:+--kv-cache-memory-bytes "$KV_BYTES"} >/dev/null
else
  docker run -d --name "$NAME" --network "$NET" "${RESTRICTED[@]}" "${MOUNTS[@]}" -e VLLM_API_KEY="$KEY" "$IMAGE" \
    ${KV_BYTES:+--kv-cache-memory-bytes "$KV_BYTES"} >/dev/null
fi
echo "started $NAME (mode=$MODE), waiting for /health ..."

set +e
docker run --rm --network "$NET" --user 10001:10001 -v "$PWD/scripts:/scripts:ro" -v "$PWD/$OUT:/out" \
  -e BASE_URL="http://$NAME:8000" -e API_KEY="$KEY" -e CANARY="$CANARY" -e ANSWER_FILE="/out/answer-$MODE.txt" \
  -e EXPECT_GUARD="$([[ $MODE == guard ]] && echo 1 || echo 0)" \
  --entrypoint python3 "$IMAGE" /scripts/smoke_client.py | tee "$OUT/checks-$MODE.jsonl"
rc=${PIPESTATUS[0]}
set -e

docker logs "$NAME" > "$OUT/container-$MODE.log" 2>&1
if [[ $MODE == guard ]]; then
  leaks=$(grep -c -F "$CANARY" "$OUT/container-$MODE.log" || true)
  # Each distinctive line of the generated answer (>= 12 chars) must be absent from the logs.
  answer_leaks=0
  while IFS= read -r line; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    (( ${#line} >= 12 )) || continue
    n=$(grep -c -F -- "$line" "$OUT/container-$MODE.log" || true)
    answer_leaks=$((answer_leaks + n))
  done < "$OUT/answer-$MODE.txt"
  token_leaks=$(grep -c -F "$KEY" "$OUT/container-$MODE.log" || true)
  facts=$(grep -c '"event": "request"' "$OUT/container-$MODE.log" || true)
  echo "{\"check\": \"logs_content_free\", \"ok\": $([[ $leaks == 0 && $answer_leaks == 0 && $token_leaks == 0 && $facts -gt 0 ]] && echo true || echo false), \"canary_hits\": $leaks, \"answer_hits\": $answer_leaks, \"token_hits\": $token_leaks, \"request_fact_lines\": $facts}" | tee -a "$OUT/checks-$MODE.jsonl"
  [[ $leaks == 0 && $answer_leaks == 0 && $token_leaks == 0 && $facts -gt 0 ]] || rc=1
fi
[[ ${KEEP:-0} == 1 ]] || docker rm -f "$NAME" >/dev/null
exit "$rc"
