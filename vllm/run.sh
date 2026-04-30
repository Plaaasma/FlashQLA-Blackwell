#!/bin/bash
# FlashQLA mod runner for spark-vllm-docker.  This script is what
# launch-cluster.sh executes inside the container at startup when a recipe
# lists `mods: - mods/flashqla`.
#
# Two pieces:
#   1. pip install flash_qla (with its tilelang + apache-tvm-ffi deps).
#   2. apply.py patches vllm's gdn_linear_attn.py so it picks our kernel.
#
# Idempotent: re-running detects existing install / sentinel and skips.
#
# Expected layout when this runs:
#   $SCRIPT_DIR/run.sh        (this file)
#   $SCRIPT_DIR/apply.py
#   $SCRIPT_DIR/flash_qla/    (the library source)
#   $SCRIPT_DIR/setup.py
#   $SCRIPT_DIR/LICENSE

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Applying FlashQLA mod ==="

if ! python3 -c 'import flash_qla' 2>/dev/null; then
    echo "[flashqla] installing flash_qla and TileLang dep"
    QLA_VERSION_SUFFIX="" pip install --no-cache-dir \
        "tilelang==0.1.8" "apache-tvm-ffi==0.1.9" 2>&1 | tail -5
    # Copy source out of the mod dir (typically mounted ro) so setup.py
    # can write build artifacts.
    BUILD_DIR="$(mktemp -d)"
    cp -r "$SCRIPT_DIR/flash_qla" "$SCRIPT_DIR/setup.py" "$SCRIPT_DIR/LICENSE" \
        "$BUILD_DIR/"
    cd "$BUILD_DIR"
    QLA_VERSION_SUFFIX="" pip install --no-cache-dir . 2>&1 | tail -5
    cd "$SCRIPT_DIR"
    rm -rf "$BUILD_DIR"
else
    echo "[flashqla] flash_qla already installed; skipping pip"
fi

# Patch gdn_linear_attn.py
python3 "$SCRIPT_DIR/apply.py"

# Bust torch.compile cache so the new graph is rebuilt.
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
echo "[flashqla] cleared torch.compile cache."

echo "=== FlashQLA mod applied successfully ==="
