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

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Applying FlashQLA mod ==="
echo "[flashqla] python: $(command -v python3)"
echo "[flashqla] torch:  $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 || echo 'IMPORT FAILED')"

# IMPORTANT: run the import check from /tmp, NOT from $SCRIPT_DIR.
# `python3 -c` puts cwd on sys.path[0], and $SCRIPT_DIR contains a
# `flash_qla/` source subdirectory next to setup.py — so an `import
# flash_qla` from $SCRIPT_DIR succeeds against the local source even
# when the wheel isn't actually installed.  The vLLM worker process
# runs from a different cwd and would then crash at warmup with
# `ModuleNotFoundError: No module named 'flash_qla'`.
if ! (cd /tmp && python3 -c 'import flash_qla' 2>/dev/null); then
    echo "[flashqla] installing tilelang + apache-tvm-ffi (full output)"
    QLA_VERSION_SUFFIX="" pip install --no-cache-dir \
        "tilelang==0.1.8" "apache-tvm-ffi==0.1.9"
    # Copy source out of the mod dir (typically mounted ro) so setup.py
    # can write build artifacts.
    BUILD_DIR="$(mktemp -d)"
    cp -r "$SCRIPT_DIR/flash_qla" "$SCRIPT_DIR/setup.py" "$SCRIPT_DIR/LICENSE" \
        "$BUILD_DIR/"
    cd "$BUILD_DIR"
    echo "[flashqla] installing flash_qla from $BUILD_DIR (full output)"
    QLA_VERSION_SUFFIX="" pip install --no-cache-dir .
    cd "$SCRIPT_DIR"
    rm -rf "$BUILD_DIR"

    # Verify import works *before* we let apply.py patch vllm — otherwise
    # the patched gdn_linear_attn.py will crash at warmup time with
    # ModuleNotFoundError, which is harder to debug.  Same cwd caveat as
    # the pre-install check above — verify from /tmp so the local source
    # dir doesn't shadow the installed wheel.
    if ! (cd /tmp && python3 -c 'import flash_qla; print("[flashqla] import OK:", flash_qla.__file__)'); then
        echo "[flashqla] ERROR: pip install reported success but 'import flash_qla' fails." >&2
        echo "[flashqla] pip list | grep -iE 'tilelang|tvm|flash':" >&2
        pip list 2>/dev/null | grep -iE 'tilelang|tvm|flash' >&2 || true
        exit 1
    fi
else
    echo "[flashqla] flash_qla already installed; skipping pip"
fi

# Patch gdn_linear_attn.py
python3 "$SCRIPT_DIR/apply.py"

# Bust torch.compile cache so the new graph is rebuilt.
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
echo "[flashqla] cleared torch.compile cache."

echo "=== FlashQLA mod applied successfully ==="
