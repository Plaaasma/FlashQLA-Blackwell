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

# Replace tilelang's bundled libcudart_stub.so with a symlink to the
# real libcudart.so.  Background:
#   - Tilelang's bundled TVM dlopens .../tilelang/lib/libcudart_stub.so
#     by absolute path during `import tilelang` (tvm/base.py:_load_lib).
#     The stub is intentionally minimal — it provides only the symbols
#     TVM itself touches.
#   - vLLM's compilation pass `allreduce_rms_fusion` does
#     `import flashinfer.comm`, which in turn does
#     `ctypes.CDLL("libcudart.so")`.  If the dynamic loader resolves
#     that to the tilelang stub (it sometimes does, depending on
#     RPATH/LD_LIBRARY_PATH), the next attribute lookup
#     (e.g. `cudaDeviceReset`) raises AttributeError, killing
#     EngineCore at compilation-backend init time before any model
#     load.
# Replacing the stub file with a symlink to the real libcudart fixes
# BOTH paths: tilelang's dlopen still finds a library at the expected
# location, and the library it gets has every CUDA Runtime symbol
# flashinfer (or anything else) might need.  Tilelang doesn't depend
# on the stub being a stub — it just needs to be loadable.
TILELANG_STUB="$(python3 -c 'import os, tilelang; print(os.path.join(os.path.dirname(tilelang.__file__), "lib", "libcudart_stub.so"))' 2>/dev/null || true)"
if [[ -n "$TILELANG_STUB" && -e "$TILELANG_STUB" && ! -L "$TILELANG_STUB" ]]; then
    REAL_CUDART="$(find /usr -name 'libcudart.so*' -not -path '*tilelang*' 2>/dev/null \
        | grep -E '/libcudart\.so(\.[0-9]+)?$' | head -n 1)"
    if [[ -n "$REAL_CUDART" ]]; then
        mv "$TILELANG_STUB" "${TILELANG_STUB}.orig"
        ln -s "$REAL_CUDART" "$TILELANG_STUB"
        echo "[flashqla] replaced tilelang stub with symlink: $TILELANG_STUB -> $REAL_CUDART"
    else
        echo "[flashqla] WARN: could not find a real libcudart.so to symlink; leaving stub in place." >&2
    fi
fi

# Patch gdn_linear_attn.py
python3 "$SCRIPT_DIR/apply.py"

# Bust torch.compile cache so the new graph is rebuilt.
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
echo "[flashqla] cleared torch.compile cache."

echo "=== FlashQLA mod applied successfully ==="
