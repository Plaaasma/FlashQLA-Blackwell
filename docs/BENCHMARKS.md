# Benchmarks

All numbers measured on **NVIDIA GB10 Grace Blackwell** (DGX Spark / ASUS GX10),
compute capability 12.1, 99 KiB opt-in shared memory, 273 GB/s LPDDR5X
unified memory.

Reference kernel: `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd`
from `fla-core==0.5.0` (the bundled Triton kernel vLLM uses today).

Measurement: `tests/test_gdr.py --no-cp --skip-bwd`, default config.

## Stage A — kernel-level forward speedup

```
Shape: B=1 Hk=64 Hv=64 T=32768
                fla  flash_qla
[fwd] csum    0.227      0.061
[fwd] solve   4.303      3.678
[fwd] wu     13.391        NaN  (fused into gdr in flash_qla)
[fwd] gdr    15.715     14.703
[fwd] o      15.381        NaN  (fused into gdr in flash_qla)
total        48.913     17.704
Speed up: 2.76x
```

**Correctness** (vs fp32 reference):

```
o_fla:  0.0017 / 0.3874   (~0.4% relative error)
o_qla:  0.0017 / 0.3874   (matches FLA)
h_qla:  0.0196 / 4.7475
s_qla:  0.0121 / 2.9764
```

`o_qla` is FlashQLA's output tensor; the first number is the abs-mean delta
to the fp32 reference, the second is the abs-mean of the reference itself.
Both kernels are within bf16 noise.

## End-to-end: vLLM serving Qwen3.6-27B-FP8

vLLM 0.x with the FlashQLA mod applied vs the same recipe with
`--gdn-prefill-backend triton`.

| Workload                      | Triton  | FlashQLA | Δ      |
| ----------------------------- | ------: | -------: | -----: |
| TTFT, 8K-token prompt prefill |   10.0s |     9.7s |  ~3%   |
| Generation, 700 tokens (warm) | 12.5/s  |   12.0/s | within noise |

**Why the gap between 2.76× kernel and 3% e2e?** Qwen3.6-27B has 38 GDN
layers, **but also** 16 full-attention layers (O(T²)) and 64 MLPs. On 8K
prefill, the full-attention layers dominate wall time. Even an infinitely
fast GDN kernel would only collapse the GDN-layer slice.

Decode is worse: vLLM's decode path goes through
`fused_recurrent_gated_delta_rule_packed_decode` — a separate Triton kernel
for the recurrent (single-token) form. **FlashQLA only replaces the chunked
prefill kernel; it does not touch decode.** To get a substantial decode
speedup on this model you would need to write a TileLang version of the
recurrent decode kernel from scratch.

Single-GPU decode is also bandwidth-bound on Spark: 27B FP8 weights / 273 GB/s
≈ 5 tok/s ceiling for a pure model-bandwidth-limited decode. DFlash spec
decoding is what gets you to 25 tok/s; kernel-level GDN work is well below
that ceiling either way.

## Where you'd actually see the kernel win

- **Long contexts** (T ≫ 32K). FA scales O(T²); GDN is O(T·K·V). Past some
  context length crossover, GDN dominates and the kernel's 2.76× shows up.
- **GDN-heavy models** (more linear-attention layers, fewer FA layers).
- **Prefill-heavy workloads.** Decode is bandwidth-bound; kernel work matters
  less.
- **Pre-training / fine-tuning style runs** that batch large `T` per step.

## Real-world tuning notes (Spark, dense 27B Qwen3.6)

These aren't FlashQLA-specific but matter if you're trying to figure out why
your wall-clock tok/s isn't matching forum claims. Surveyed against
NVIDIA Spark forum threads, the spark-vllm-docker repo, z-lab DFlash
discussions, and r/LocalLLaMA Spark/GB10 threads (2026-Q1 to 2026-Q2).

**The bandwidth ceiling.** Spark's GB10 has 273 GB/s LPDDR5X unified memory.
Decode on a dense 27B FP8 model is bandwidth-bound — 27 GB / 273 GB/s ≈ **10
tok/s** without speculation. Spec decoding multiplies this by mean-acceptance
length: DFlash-15 averages ~3.0 mean accept length and lands at ~25 tok/s
single-stream on this model. That is at or above the public state of the art
for FP8-dense 27B on a single Spark.

**Constraint-safe knobs that are worth flipping** (each maybe +5-15%):

- `VLLM_USE_FLASHINFER_SAMPLER=1` env var — bumps spec-decode acceptance by
  a few percentage points on both DFlash and MTP. The sample recipe in
  `vllm/recipes/` already has this.
- `--max-num-seqs 4` (with `--max-num-batched-tokens 16384` or higher) for
  pure single-stream benchmarking. Larger `max-num-seqs` doesn't *hurt*
  single-user tok/s much in practice but the fastest published configs
  uniformly use small batches.
- `--enable-chunked-prefill` (default-on in recent vLLM, but harmless to
  set explicitly).

**Things that look like they should help but don't on dense 27B:**

- `--kv-cache-dtype fp8`: no measured single-stream throughput gain on this
  model and outputs diverged measurably on the related 35B-A3B (hybrid
  linear+full attention seems to interact badly with fp8 KV scales).
- `MTP-3` (or higher `num_speculative_tokens` for MTP): regresses vs MTP-2
  because chained acceptance falls off fast.
- `DFlash` `num_speculative_tokens > 15`: acceptance peaks 7-8 then degrades.
- Prefix caching during single-stream micro-benchmarks: small acceptance hit
  due to cache-aware scheduling, but you almost certainly want it on for
  real workloads.
- `TREE_ATTN` or `FLEX_ATTENTION` outside their intended use cases. TREE_ATTN
  is required for DFlash's non-causal draft attention; FLEX has known
  upstream stability issues on this model class.

**Bigger levers that change the tradeoff:**

| Path | Approx tok/s | Cost |
|---|---:|---|
| FP8 + DFlash-15 + FlashQLA (current best on dense 27B) | 25-30 | none beyond what's documented here |
| **Switch to a sparse MoE** (e.g. Qwen3.6-35B-A3B with only ~3B active per token) | 40-50 | different model, but same family |
| **INT4 weight quantization** (AutoRound, GPTQ) of dense 27B | 50-70 | accuracy degradation — worth measuring per-task before committing |
| **MXFP4 / NVFP4 weight quantization** | 70-80+ | larger accuracy degradation; widely reported broken on certain Qwen3 variants |
| **Add a second Spark** (TP=2) | 40-60 | hardware |

If you're under all the same constraints (FP8+ weights, single Spark, no
NVFP4/MXFP4), 25-30 tok/s is the realistic ceiling on dense 27B. Going above
that means changing the model, the weight format, or the hardware.

## Reproducing

```bash
git clone <this-repo>
cd FlashQLA-Blackwell
pip install -v . pandas tabulate fla-core==0.5.0
python3 tests/test_gdr.py --no-cp --skip-bwd
```

The first run takes ~30s for TileLang to JIT-compile the kernel. Subsequent
runs are cached in `~/.tilelang/`.

To sweep more shapes, edit `tests/test_gdr.py` — the call sites take
`batch_size`, `num_tokens`, `num_k_heads`, `num_v_heads`, `head_dim_k`,
`head_dim_v` directly.
