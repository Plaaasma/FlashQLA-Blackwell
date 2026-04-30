#!/bin/bash
# One-shot installer for the manual-integration path:
#
#   1. pip install this repo's flash_qla into the *current* Python env
#   2. patch vLLM's gdn_linear_attn.py (idempotent)
#
# Run this from the repo root, in the same Python env where vLLM is installed.
#
#   ./scripts/install_into_vllm.sh
#
# If your vLLM lives somewhere other than the system site-packages, edit
# VLLM_ROOT at the top of vllm/apply.py to point at it.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/2] pip install flash_qla ==="
pip install -v .

echo "=== [2/2] patching vLLM ==="
python3 vllm/apply.py

echo
echo "=== Done. ==="
echo "Restart vLLM and look for this line in the log to confirm:"
echo
echo "    Using FlashQLA TileLang GDN prefill kernel (Blackwell)"
echo
echo "If you see 'Using Triton/FLA GDN prefill kernel' instead, check that"
echo "your Python env is the same one vLLM imports from."
