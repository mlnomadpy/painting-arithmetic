#!/usr/bin/env bash
# Deploy demo/ as a static Hugging Face Space.
#
# Prerequisites:
#   1. `pip install huggingface_hub`
#   2. `huggingface-cli login` (paste your HF token)
#   3. You've already exported demo/model.onnx via scripts/export_onnx.py
#
# Usage:
#   ./marketing/deploy/deploy_hf_space.sh <hf-username> [space-name]
#
# Example:
#   ./marketing/deploy/deploy_hf_space.sh mlnomadpy painting-arithmetic
#
# This:
#   - Creates (if missing) a static HF Space at <user>/<space-name>.
#   - Uploads demo/index.html, demo/model.onnx, and the Space README.
#   - Prints the live URL.

set -euo pipefail

USER="${1:?usage: deploy_hf_space.sh <hf-username> [space-name]}"
SPACE="${2:-painting-arithmetic}"
REPO_ID="${USER}/${SPACE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$ROOT/demo/model.onnx" ]]; then
  echo "demo/model.onnx not found."
  echo "First train, then export:"
  echo "  painting-arithmetic-train --epochs 25 --train-size 60000"
  echo "  python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx"
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install huggingface_hub first:  pip install huggingface_hub"
  exit 1
fi

# Stage a temp dir with the Space layout.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$ROOT/demo/index.html"                                "$STAGE/index.html"
cp "$ROOT/demo/model.onnx"                                "$STAGE/model.onnx"
cp "$ROOT/marketing/deploy/hf_space_readme.md"            "$STAGE/README.md"
cp "$ROOT/marketing/social_card.png"                      "$STAGE/thumbnail.png" 2>/dev/null || true

echo "→ creating / updating Space $REPO_ID"
huggingface-cli repo create "$SPACE" --type space --space_sdk static --yes \
    --organization "$USER" 2>/dev/null || true   # idempotent

huggingface-cli upload "$REPO_ID" "$STAGE" --repo-type space \
    --commit-message "deploy painting-arithmetic demo" >/dev/null

echo "✓ deployed: https://huggingface.co/spaces/$REPO_ID"
