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
