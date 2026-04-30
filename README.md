# FlashQLA-Blackwell

A fork of the [Qwen team's FlashQLA](https://github.com/QwenLM/FlashQLA) TileLang
kernels, ported to **NVIDIA GB10 Grace Blackwell** (SM_120 / SM_121) — the GPU
inside the **DGX Spark / ASUS GX10**. Drops into vLLM as a fast prefill kernel
for Qwen3.6 family models that use Gated Delta Net (GDN) linear attention.

## What this is, in plain words

Qwen3.6 (and similar "linear-attention" models) replaces some of its attention
layers with a thing called **Gated Delta Net (GDN)**. vLLM's bundled GDN
implementation uses a Triton kernel that runs fine but isn't tuned for Blackwell
consumer parts. The Qwen team published `FlashQLA`, a hand-tuned TileLang
kernel that's much faster — but it only ran on Hopper (SM 9.0). This repo is
that kernel **patched to work on Spark** (SM 12.x).

> **TL;DR — what changed vs upstream**
> - Bypassed the SM90 arch gate
> - Switched a `T.gemm_v1(transpose_B=True)` to `T.gemm_v2` (silently produces
>   wrong results on Blackwell — see `docs/BLACKWELL_FIXES.md`)
> - Shrunk shared-memory footprint to fit Blackwell consumer's 99 KiB
> - Fixed two upstream bugs (extra arg in `apply()`, K↔V dim mismatch with vLLM)
> - Added a vLLM mod that auto-selects this kernel on Blackwell

## Speedup

Measured on GB10 (DGX Spark), `B=1`, `T=32768`, vs vLLM's bundled FLA Triton
kernel. Run `python3 tests/test_gdr.py --no-cp --skip-bwd` to reproduce.

| Shape (B=1, T=32768) | FLA Triton (ms) | FlashQLA (ms) | Speedup |
| -------------------- | --------------: | ------------: | ------: |
| Hk=64, Hv=64         |           48.91 |         17.70 | **2.76×** |
| Hk=16, Hv=4 (probe)  |              ~  |             ~ | 1.73× |

End-to-end vLLM TTFT win on Qwen3.6-27B is much smaller (~3% on 8K prefill)
because that model's prefill is dominated by its 16 full-attention layers and
64 MLPs. The kernel still wins where GDN dominates (long contexts, GDN-heavy
models, pre-training-style workloads).

## Prerequisites

- **Hardware:** NVIDIA Blackwell GPU (compute capability ≥ 10.0). Tested on
  GB10. SM_120/121 specifically. Should also be fine on SM_100 (Hopper Next /
  Blackwell datacenter), but not tested.
- **Driver:** anything that supports CUDA 12.8+.
- **Python ≥ 3.10**, **PyTorch ≥ 2.8**.
- For the vLLM integration: a working vLLM install. If you have no idea what
  that means, see [`vllm/README.md`](vllm/README.md) — it has a copy-paste
  walkthrough for the DGX Spark Docker setup that most Spark users have.

This repo also works fine on Hopper (the upstream FlashQLA target) — none of
the Blackwell fixes break Hopper.

## Quick start (kernel only)

```bash
git clone <this-repo> FlashQLA-Blackwell
cd FlashQLA-Blackwell
pip install -v .

# Sanity-check it works on your GPU.
pip install pandas tabulate fla-core==0.5.0
python3 tests/test_gdr.py --no-cp --skip-bwd
```

You should see `o_qla` numbers very close to `o_fla` (correctness) and a
`Speed up: 2.x` line at the bottom.

## Quick start (with vLLM, on DGX Spark)

If you're using the [spark-vllm-docker](https://github.com/...) recipe runner
that ships with the Spark, see [`vllm/README.md`](vllm/README.md) — it walks
you through dropping this in as a "mod" with one line in your YAML recipe.

If you have a hand-rolled vLLM install, see
[`vllm/README.md#manual-integration`](vllm/README.md#manual-integration) — you
just run `pip install .` then `python3 vllm/apply.py` and restart vLLM.

## Use the kernel directly (no vLLM)

```python
import torch
from flash_qla import chunk_gated_delta_rule

o, final_state = chunk_gated_delta_rule(
    q=q,           # [B, T, H_q, K]    (bf16/fp16)
    k=k,           # [B, T, H_q, K]
    v=v,           # [B, T, H_v, V]
    g=g,           # [B, T, H_v]
    beta=beta,     # [B, T, H_v]
    initial_state=initial_state,    # optional, [B, H_v, K, V] — note: K, V (not V, K)
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=cu_seqlens,          # optional, for variable-length packing
)
```

See [`examples/standalone_gdr.py`](examples/standalone_gdr.py) for a full
runnable example.

> **Note on state layout.** The kernel expects `initial_state` shaped
> `(B, H, K, V)`. vLLM's `MambaStateShapeCalculator` uses `(B, H, V, K)` —
> the vLLM mod transposes for you, but if you're calling the kernel directly,
> match the kernel's convention.

## Repo layout

```
flash_qla/                 # The patched library — same API as upstream
tests/                     # Stage A correctness + speed test
examples/                  # Standalone usage examples
docs/                      # Technical notes (Blackwell fixes, benchmarks)
vllm/                      # vLLM integration (mod + recipe + walkthrough)
```

## Credits

- **Qwen team / Alibaba** — the original FlashQLA library and TileLang kernels
  ([upstream repo](https://github.com/QwenLM/FlashQLA)).
- **Tile-AI** — [TileLang](https://github.com/tile-ai/tilelang), the DSL the
  kernels are written in.
- This fork's Blackwell port is a small set of patches on top.

## License

MIT, same as upstream — see [LICENSE](LICENSE). Copyright (c) 2026 Qwen,
Alibaba; modifications copyright (c) 2026 contributors.

## Bug reports

Open an issue with:

1. `python3 -c "import torch; print(torch.cuda.get_device_capability())"` output
2. `pip show flash_qla tilelang apache-tvm-ffi`
3. The full `tests/test_gdr.py --no-cp --skip-bwd` output
