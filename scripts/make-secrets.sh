#!/usr/bin/env bash
# Create the local secret files consumed by kustomize's secretGenerator (gitignored, mode 600).
#
#   api-key.txt  : random 256-bit Bearer token for API clients (kept if it already exists)
#   hf-token.txt : Hugging Face token for the model-fetch initContainer (READ-only tokens only)
#
# Usage (any shell, any working directory):
#   bash scripts/make-secrets.sh                  # asks for the HF token with a hidden prompt
#   bash scripts/make-secrets.sh --no-hf          # public models only (empty hf-token.txt)
#   HF_REPO=user/private-model bash scripts/make-secrets.sh   # also checks the token can read that repo
#   bash scripts/make-secrets.sh --check          # only show what is configured (never prints values)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/k8s/base/secrets"
mkdir -p "$DIR"
umask 077

say() { printf '%s\n' "$*" >&2; }

hf_get() {  # url -> body; the token goes in a header read from stdin, never on the command line
  curl -sS --max-time 20 -H @- "$1" <<< "Authorization: Bearer $TOKEN"
}

check_token() {
  local who name role code
  who=$(hf_get https://huggingface.co/api/whoami-v2) || { say "ERROR: cannot reach huggingface.co"; return 1; }
  if ! printf '%s' "$who" | grep -q '"name"'; then
    say "ERROR: Hugging Face rejected the token: $(printf '%s' "$who" | head -c 200)"
    return 1
  fi
  read -r name role < <(printf '%s' "$who" | python3 -c '
import json, sys
d = json.load(sys.stdin)
tok = (d.get("auth") or {}).get("accessToken") or {}
role = tok.get("role", "?")
if role == "fineGrained":
    fg = tok.get("fineGrained") or {}
    perms = {p for s in fg.get("scoped", []) for p in s.get("permissions", [])} | set(fg.get("global", []))
    role = "fineGrained-write" if any(p.endswith((".write", ".manage")) for p in perms) else "fineGrained-read"
print(d.get("name", "?"), role)')
  say "token belongs to: $name, type: $role"
  case $role in
    read|fineGrained-read) ;;
    *) say "ERROR: this token can WRITE. Create a read-only one: https://huggingface.co/settings/tokens"; return 1 ;;
  esac
  if [[ -n "${HF_REPO:-}" ]]; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 -H @- \
      "https://huggingface.co/api/models/$HF_REPO" <<< "Authorization: Bearer $TOKEN")
    if [[ $code == 200 ]]; then say "access to $HF_REPO: OK"; else say "ERROR: $HF_REPO -> HTTP $code"; return 1; fi
  fi
}

status() {
  local f
  for f in api-key.txt hf-token.txt; do
    if [[ -s "$DIR/$f" ]]; then say "$f: set ($(wc -c < "$DIR/$f" | tr -d ' ') bytes)"; else say "$f: empty/missing"; fi
  done
}

if [[ ${1:-} == --check ]]; then status; exit 0; fi

if [[ ! -s "$DIR/api-key.txt" ]]; then
  openssl rand -hex 32 | tr -d '\n' > "$DIR/api-key.txt"
  say "created api-key.txt"
fi

if [[ ${1:-} == --no-hf ]]; then
  : > "$DIR/hf-token.txt"
  say "hf-token.txt: empty (public models only)"
  status
  exit 0
fi

TOKEN=${HF_TOKEN:-}
if [[ -z "$TOKEN" ]]; then
  [[ -t 0 ]] || { say "ERROR: no terminal for the prompt; set HF_TOKEN or use --no-hf"; exit 1; }
  read -r -s -p "Paste Hugging Face READ token (input hidden): " TOKEN
  echo >&2
fi
TOKEN=$(printf '%s' "$TOKEN" | tr -d '[:space:]')   # strips CR/LF and spaces from copy-paste
[[ $TOKEN == hf_* ]] || { say "ERROR: a Hugging Face token starts with hf_"; exit 1; }

check_token
printf '%s' "$TOKEN" > "$DIR/hf-token.txt"
unset TOKEN
say "wrote hf-token.txt"
status
say "apply with: kubectl apply -k k8s/overlays/mac-m3   (Secret name hash changes -> pod restarts)"
