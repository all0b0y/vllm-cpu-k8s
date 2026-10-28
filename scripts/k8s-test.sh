#!/usr/bin/env bash
# Phase-2 acceptance tests on the running cluster (namespace llm).
#   smoke     : black-box checks from an in-cluster client pod (role=client) through Service -> nginx -> guard -> vLLM
#   logs      : the canary / generated answer / API key must not appear in any container log of the vLLM pod
#   attacker  : a pod WITHOUT role=client must not reach the model at all (NetworkPolicy)
#   offline   : apply the overlay without any egress, restart the pod, it must become Ready from the PVC
# Usage: scripts/k8s-test.sh [smoke|logs|attacker|offline|all]   (OVERLAY=k8s/overlays/mac-m3 by default)
set -euo pipefail
cd "$(dirname "$0")/.."
NS=llm
OVERLAY=${OVERLAY:-k8s/overlays/mac-m3}
OFFLINE_OVERLAY=${OFFLINE_OVERLAY:-${OVERLAY}-offline}
IMAGE=${IMAGE:-vllm-cpu-guard:0.31.0}
OUT=${OUT:-results/k8s}
mkdir -p "$OUT"
API_SECRET=$(kubectl -n $NS get secret -l app.kubernetes.io/component=api-key -o jsonpath='{.items[0].metadata.name}')
API_KEY=$(kubectl -n $NS get secret "$API_SECRET" -o jsonpath='{.data.api-key}' | base64 -d)

client_pod() {  # name role command...
  local name=$1 role=$2; shift 2
  kubectl -n $NS delete pod "$name" --ignore-not-found --wait >/dev/null
  kubectl -n $NS create configmap smoke-scripts --from-file=scripts/smoke_client.py \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $name
  namespace: $NS
  labels: {role: $role}
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  enableServiceLinks: false
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: client
      image: $IMAGE
      imagePullPolicy: IfNotPresent
      command: $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")
      env:
        - {name: BASE_URL, value: "http://vllm.$NS.svc:8000"}
        - {name: API_KEY, valueFrom: {secretKeyRef: {name: $API_SECRET, key: api-key}}}
        - {name: CANARY, value: "${CANARY:-none}"}
        - {name: MODEL_NAME, value: "${MODEL_NAME:-qwen2.5-1.5b-instruct}"}
        - {name: DENY_STATUSES, value: "401,404"}
        - {name: PRINT_ANSWER, value: "1"}
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities: {drop: ["ALL"]}
      resources:
        requests: {cpu: 500m, memory: 512Mi}
        limits: {cpu: 500m, memory: 512Mi}
      volumeMounts:
        - {name: scripts, mountPath: /scripts, readOnly: true}
        - {name: tmp, mountPath: /tmp}
  volumes:
    - {name: scripts, configMap: {name: smoke-scripts}}
    - {name: tmp, emptyDir: {}}
EOF
  kubectl -n $NS wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$name" --timeout=1200s >/dev/null 2>&1 || true
  kubectl -n $NS logs "$name"
}

wait_ready() {
  kubectl -n $NS rollout status deploy/vllm --timeout=1200s
}

t_smoke() {
  CANARY="CANARY-$(openssl rand -hex 8)"
  export CANARY
  echo "$CANARY" > "$OUT/canary.txt"
  client_pod smoke client /opt/venv/bin/python /scripts/smoke_client.py | tee "$OUT/smoke.jsonl"
  [[ $(grep -c '"ok": true' "$OUT/smoke.jsonl") -ge 9 ]] && ! grep -q '"ok": false' "$OUT/smoke.jsonl"
}

t_logs() {
  local pod canary answer hits=0 ahits=0 khits=0
  pod=$(kubectl -n $NS get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')
  canary=$(cat "$OUT/canary.txt")
  for c in model-fetch vllm edge-proxy; do
    kubectl -n $NS logs "$pod" -c "$c" > "$OUT/log-$c.txt" 2>&1 || true
  done
  cat "$OUT"/log-*.txt > "$OUT/all-logs.txt"
  hits=$(grep -c -F "$canary" "$OUT/all-logs.txt" || true)
  khits=$(grep -c -F "$API_KEY" "$OUT/all-logs.txt" || true)
  answer=$(grep '"canary_answer"' "$OUT/smoke.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["canary_answer"])' || true)
  while IFS= read -r line; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    (( ${#line} >= 12 )) || continue
    ahits=$((ahits + $(grep -c -F -- "$line" "$OUT/all-logs.txt" || true)))
  done <<< "$answer"
  facts=$(grep -c '"event": "request"' "$OUT/log-vllm.txt" || true)
  proxy=$(grep -c '"event":"proxy"' "$OUT/log-edge-proxy.txt" || true)
  echo "{\"check\": \"logs_content_free\", \"ok\": $([[ $hits == 0 && $ahits == 0 && $khits == 0 && $facts -gt 0 ]] && echo true || echo false), \"canary_hits\": $hits, \"answer_hits\": $ahits, \"api_key_hits\": $khits, \"guard_request_lines\": $facts, \"proxy_request_lines\": $proxy}" | tee "$OUT/logs.json"
  [[ $hits == 0 && $ahits == 0 && $khits == 0 && $facts -gt 0 ]]
}

t_attacker() {
  # Same image, same namespace, but no role=client label -> NetworkPolicy must drop the connection.
  client_pod attacker none /opt/venv/bin/python -c "
import json, requests
for path in ['/health', '/invocations']:
    try:
        r = requests.get('http://vllm.$NS.svc:8000' + path, timeout=5)
        print(json.dumps({'check': 'attacker_' + path.strip('/'), 'ok': False, 'status': r.status_code}))
    except requests.RequestException as e:
        print(json.dumps({'check': 'attacker_' + path.strip('/'), 'ok': True, 'blocked': type(e).__name__}))
" | tee "$OUT/attacker.jsonl"
  [[ $(grep -c '"ok": true' "$OUT/attacker.jsonl") -eq 2 ]] && ! grep -q '"ok": false' "$OUT/attacker.jsonl"
}

t_offline() {
  # `kubectl apply` does not prune: the egress allow from the normal overlay must be deleted explicitly.
  kubectl apply -k "$OFFLINE_OVERLAY" >/dev/null
  kubectl -n $NS delete networkpolicy allow-model-fetch-egress --ignore-not-found
  kubectl -n $NS get networkpolicy -o name | tee "$OUT/offline-policies.txt"
  if grep -q allow-model-fetch-egress "$OUT/offline-policies.txt"; then echo "egress allow still present" >&2; return 1; fi
  local started; started=$(date +%s)
  kubectl -n $NS delete pod -l app=vllm --wait=false >/dev/null
  sleep 5
  wait_ready
  local pod; pod=$(kubectl -n $NS get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')
  kubectl -n $NS logs "$pod" -c model-fetch | tee "$OUT/offline-fetch.log"
  echo "{\"check\": \"offline_restart_ready\", \"ok\": true, \"seconds_to_ready\": $(( $(date +%s) - started ))}" | tee "$OUT/offline.json"
  grep -q '"network": false' "$OUT/offline-fetch.log"
}

case ${1:-all} in
  smoke) t_smoke ;;
  logs) t_logs ;;
  attacker) t_attacker ;;
  offline) t_offline ;;
  online) kubectl apply -k "$OVERLAY" ;;   # restore the normal overlay (egress for downloads)
  all) t_smoke && t_logs && t_attacker && t_offline && t_smoke && t_logs ;;
esac
