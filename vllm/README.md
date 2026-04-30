# Using FlashQLA with vLLM

This directory ships a small "patcher" that hooks FlashQLA into vLLM's GDN
prefill path. After applying it, vLLM auto-selects FlashQLA on Blackwell GPUs
and falls back to the bundled Triton kernel everywhere else — so it's safe to
leave applied.

There are two integration paths:

1. **`spark-vllm-docker` recipe** — the easiest path if you're already running
   vLLM via the recipe runner most Spark users use. One YAML line.
2. **Manual integration** — if you have your own vLLM install, you run two
   commands inside that environment.

If you don't know what either of those means but you have a Spark and want to
run Qwen3.6, start with path 1.

---

## Path 1: spark-vllm-docker recipe (easiest)

This assumes you have [spark-vllm-docker](https://github.com/...) cloned
somewhere, and you launch vLLM with `./launch-cluster.sh <recipe>.yaml` or
similar.

### One-time setup

Copy this `vllm/` directory into your `spark-vllm-docker/mods/` tree as a
new mod called `flashqla`:

```bash
# from inside this repo
cp -r vllm $YOUR_SPARK_VLLM_DOCKER/mods/flashqla

# then copy the FlashQLA source alongside it (the mod's run.sh installs it)
cp -r flash_qla setup.py LICENSE $YOUR_SPARK_VLLM_DOCKER/mods/flashqla/
```

The mod runner expects this layout:

```
spark-vllm-docker/mods/flashqla/
├── run.sh              # bash hook the launcher runs at container start
├── apply.py            # patches vllm at runtime
├── flash_qla/          # the library source
├── setup.py
└── LICENSE
```

### In your recipe

Add one line under `mods:` in any recipe that uses a Qwen3.6 (or other GDN)
model:

```yaml
mods:
  - mods/flashqla
```

A complete sample is in [`recipes/qwen3.6-27b-fp8-mtp2.yaml`](recipes/qwen3.6-27b-fp8-mtp2.yaml)
— Qwen3.6-27B-FP8 with MTP-2 self-speculative decoding (lossless, no
separate draft model required).
Drop it in your `spark-vllm-docker/recipes/` and launch with your normal
launcher script.

### Verify it's active

When vLLM starts up, look for this line in the log:

```
Using FlashQLA TileLang GDN prefill kernel (Blackwell)
```

If you see `Using Triton/FLA GDN prefill kernel` instead, the mod isn't being
applied. Most common reason: the recipe doesn't list it under `mods:`.

You can also force-select it explicitly via the recipe's vLLM args:

```
--gdn-prefill-backend flashqla
```

---

## Path 2: Manual integration (no spark-vllm-docker)

If you maintain your own vLLM install (system pip, conda env, your own Docker
image, etc.):

```bash
# 1. Install the kernel into the same Python env that vLLM uses
cd FlashQLA-Blackwell
pip install -v .

# 2. Patch vLLM's gdn_linear_attn.py (idempotent; safe to re-run)
python3 vllm/apply.py

# 3. Restart vLLM
```

### What `apply.py` does

It edits one file inside your installed vLLM:

```
<your-vllm-site-packages>/vllm/model_executor/layers/mamba/gdn_linear_attn.py
```

(That path is hard-coded as `/usr/local/lib/python3.12/dist-packages/vllm` at
the top of the file. Edit `VLLM_ROOT` at the top of `apply.py` if your vLLM
lives somewhere else, e.g. a conda env or virtualenv.)

The edits are:

1. Insert a small wrapper function `_flashqla_chunk_gated_delta_rule` that
   lazy-imports `flash_qla` and translates vLLM's state layout `(B,H,V,K)` to
   FlashQLA's `(B,H,K,V)`.
2. Add a `forward_flashqla` method to the `ChunkGatedDeltaRule` class.
3. Modify `__init__` to detect Blackwell (compute capability major ≥ 10) and
   route to `forward_flashqla` automatically — keeps FlashInfer for Hopper and
   Triton everywhere else.

All edits are guarded by a `# [FLASHQLA PATCH]` sentinel so re-running
`apply.py` is a no-op.

### Reverting

`apply.py` doesn't auto-revert, but the edits are bounded and tagged. To undo
them:

```bash
pip uninstall vllm  # nuclear option, then reinstall
# OR: hand-remove the lines tagged `# [FLASHQLA PATCH]` in gdn_linear_attn.py
```

---

## What if I'm not on Blackwell?

The vLLM mod is gated on `torch.cuda.get_device_capability()[0] >= 10`. If you
apply it on a Hopper (SM_90) box, vLLM will keep using FlashInfer
(`forward_cuda`). On older GPUs it'll keep using Triton. So leaving the patch
applied is safe across mixed deployments.

If you *want* to force FlashQLA on Hopper, set the recipe's
`additional_config.gdn_prefill_backend` to `"flashqla"`. The kernel runs
correctly there too — that's its native target.

---

## Troubleshooting

### `ModuleNotFoundError: flash_qla`

vLLM is running in a different Python environment than the one you ran
`pip install .` in. Check `which python3` inside your vLLM container/env vs
where you installed.

### Output is gibberish

You probably skipped the K↔V transpose. The kernel's wrapper inside
`apply.py` (`_flashqla_chunk_gated_delta_rule`) handles this. If you call
`flash_qla.chunk_gated_delta_rule` directly from your own code, your
`initial_state` must be shaped `(B, H, K, V)`, not `(B, H, V, K)`.

### `TypeError: ChunkGatedDeltaRuleFunction.forward() takes from 6 to 10 positional arguments but 11 were given`

This is the upstream FlashQLA bug — it's already fixed in this fork. If you're
seeing it, you have an older `flash_qla` installed. `pip uninstall flash_qla`
then reinstall from this repo.

### Out-of-memory at startup

FlashQLA JIT-compiles a TileLang kernel the first time it runs. Allow
~30 seconds extra startup time and ~few GB extra memory headroom for the JIT
cache. On Spark (128 GB unified), drop `gpu_memory_utilization` to 0.75 and
`max_model_len` to 65536 if you're running anything else alongside vLLM. The
sample recipe is already tuned this way.

### Where is `gdn_linear_attn.py`?

```bash
python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
# then: ls <that>/model_executor/layers/mamba/gdn_linear_attn.py
```

If that file doesn't exist your vLLM is too old (GDN was added in vLLM 0.x —
needs a recent Qwen3-supporting build).
