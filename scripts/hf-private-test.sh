#!/usr/bin/env bash
# HF token scenarios for the model-fetch initContainer against a PRIVATE repository.
#   no-token    : no Secret mounted                     -> must fail with exit 2 (access denied)
#   wrong-token : Secret with a syntactically valid but fake token -> must fail with exit 2
#   right-token : the real read-only Secret (hf-token-*)  -> must download + verify (exit 0)
# Each scenario is a pod with the loader image, the same restricted securityContext as the initContainer,
# an emptyDir instead of the PVC, and a temporary NetworkPolicy that allows HTTPS egress for these pods only.
# Usage: HF_REPO=user/repo HF_REV=<40-hex commit> scripts/hf-private-test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NS=llm
HF_REPO=${HF_REPO:?set HF_REPO=user/repo}
HF_REV=${HF_REV:?set HF_REV to the full commit sha}
IMAGE=${LOADER_IMAGE:-model-loader:1}
OUT=${OUT:-results/k8s}
mkdir -p "$OUT"
REAL_SECRET=$(kubectl -n $NS get secret -l app.kubernetes.io/component=hf-token -o jsonpath='{.items[0].metadata.name}')
TOKEN=$(kubectl -n $NS get secret "$REAL_SECRET" -o jsonpath='{.data.token}' | base64 -d)
[[ -n $TOKEN ]] || { echo "Secret $REAL_SECRET is empty: run scripts/make-secrets.sh and kubectl apply -k" >&2; exit 1; }

cleanup() {
  kubectl -n $NS delete pod -l app=hf-fetch-test --ignore-not-found --wait=false >/dev/null
  kubectl -n $NS delete networkpolicy hf-fetch-test-egress --ignore-not-found >/dev/null
  kubectl -n $NS delete secret hf-token-wrong --ignore-not-found >/dev/null
}
trap cleanup EXIT

kubectl apply -f - >/dev/null <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: hf-fetch-test-egress, namespace: $NS}
spec:
  podSelector: {matchLabels: {app: hf-fetch-test}}
  policyTypes: ["Egress"]
  egress:
    - to: [{ipBlock: {cidr: 0.0.0.0/0, except: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10"]}}]
      ports: [{protocol: TCP, port: 443}]
EOF
# Fake token: right format, random content (never a real credential).
kubectl -n $NS create secret generic hf-token-wrong --from-literal=token="hf_$(openssl rand -hex 17)" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

run_case() {  # name secret-or-empty expected-exit
  local name=$1 secret=$2 want=$3 vol="" mount="" env=""
  if [[ -n $secret ]]; then
    vol="- {name: hf-token, secret: {secretName: $secret, defaultMode: 0400, items: [{key: token, path: token}]}}"
    mount="- {name: hf-token, mountPath: /var/run/secrets/hf, readOnly: true}"
    env="- {name: HF_TOKEN_PATH, value: /var/run/secrets/hf/token}"
  fi
  kubectl -n $NS delete pod "hf-$name" --ignore-not-found --wait >/dev/null
  kubectl apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata: {name: hf-$name, namespace: $NS, labels: {app: hf-fetch-test}}
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  securityContext: {runAsNonRoot: true, runAsUser: 10001, runAsGroup: 10001, fsGroup: 10001, seccompProfile: {type: RuntimeDefault}}
  containers:
    - name: fetch
      image: $IMAGE
      imagePullPolicy: IfNotPresent
      env:
        - {name: MODEL_ID, value: "$HF_REPO"}
        - {name: MODEL_REVISION, value: "$HF_REV"}
        $env
      securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: {drop: ["ALL"]}}
      resources: {requests: {cpu: 500m, memory: 512Mi}, limits: {cpu: 500m, memory: 512Mi}}
      volumeMounts:
        - {name: models, mountPath: /models}
        - {name: tmp, mountPath: /tmp}
        $mount
  volumes:
    - {name: models, emptyDir: {sizeLimit: 1Gi}}
    - {name: tmp, emptyDir: {}}
    $vol
EOF
  kubectl -n $NS wait --for=jsonpath='{.status.phase}'=Succeeded "pod/hf-$name" --timeout=300s >/dev/null 2>&1 \
    || kubectl -n $NS wait --for=jsonpath='{.status.phase}'=Failed "pod/hf-$name" --timeout=5s >/dev/null 2>&1 || true
  local code log leak
  code=$(kubectl -n $NS get pod "hf-$name" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}')
  log=$(kubectl -n $NS logs "hf-$name" 2>&1 | tail -2 | tr '\n' ' ')
  leak=$(kubectl -n $NS logs "hf-$name" 2>&1 | grep -c -F "$TOKEN" || true)
  local ok=false
  [[ $code == "$want" && $leak == 0 ]] && ok=true
  python3 -c 'import json,sys; print(json.dumps({"check": "hf_"+sys.argv[1], "ok": sys.argv[2]=="true", "exit": int(sys.argv[3] or -1),
    "want": int(sys.argv[4]), "token_in_logs": int(sys.argv[5]), "last_log": sys.argv[6][-300:]}))' \
    "$name" "$ok" "$code" "$want" "$leak" "$log"
}

{
  run_case no-token "" 2
  run_case wrong-token hf-token-wrong 2
  run_case right-token "$REAL_SECRET" 0
} | tee "$OUT/hf-private.jsonl"
unset TOKEN
[[ $(grep -c '"ok": true' "$OUT/hf-private.jsonl") -eq 3 ]]
