#!/usr/bin/env bash
# Create the local secret files consumed by kustomize's secretGenerator (gitignored).
#   api-key.txt : random 256-bit Bearer token for API clients (kept if it already exists)
#   hf-token.txt: Hugging Face READ token from $HF_TOKEN; empty is fine for public models
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/k8s/base/secrets"
mkdir -p "$DIR"
umask 077
if [[ ! -s "$DIR/api-key.txt" ]]; then
  openssl rand -hex 32 | tr -d '\n' > "$DIR/api-key.txt"
  echo "created $DIR/api-key.txt"
fi
if [[ -n "${HF_TOKEN:-}" || ! -e "$DIR/hf-token.txt" ]]; then
  printf '%s' "${HF_TOKEN:-}" > "$DIR/hf-token.txt"
  echo "wrote $DIR/hf-token.txt ($([[ -n "${HF_TOKEN:-}" ]] && echo 'token set' || echo 'empty: public models only'))"
fi
