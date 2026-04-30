# Blackwell (SM_120 / SM_121) port notes

This is the diff against upstream FlashQLA, with reasoning. If you're trying
to port other TileLang kernels to consumer Blackwell, the gotchas here are
likely to bite you too.

## 1. `T.gemm_v1(transpose_B=True)` silently produces wrong results

**File:** `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`

The Q@Kᵀ gemm in the consumer warpgroup was:

```python
T.gemm_v1(q_shared[:, :], k_shared[:, :], p_fragment, transpose_B=True, clear_accum=True)
```

On SM_120/121 this compiles fine, runs without errors, and produces wrong
values — `p_fragment` differs from the reference by ~50% of magnitude. We
caught it by dumping the fragment from the kernel and comparing to
`scale * tril(Q @ K^T)`.

**Fix:** swap to `T.gemm_v2`. Same call signature, different lowering — works
on Blackwell:

```python
T.gemm_v2(q_shared[:, :], k_shared[:, :], p_fragment, transpose_B=True, clear_accum=True)
```

We did *not* see the same problem with `transpose_A` or with no transpose, and
not with the other gemms in this kernel (which use `clear_accum=False` over
already-laid-out shared mem). The bug appears specific to the `transpose_B`
path on consumer Blackwell — possibly a TileLang TMA-descriptor or
WGMMA-vs-MMA-Blackwell lowering issue. We've left a note in
`feedback_tilelang_blackwell_gemm.md`.

## 2. Shared-memory budget: 99 KiB on GB10 vs 228 KiB on Hopper

**File:** `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`

Hopper's H100/H200 expose 228 KiB of opt-in shared memory per CTA. GB10 only
gives you 99 KiB. The original FlashQLA kernel allocates with `block_DV=128`,
which busts the budget instantly.

**Fix:** for compute_capability ≥ 10:

```python
if torch.cuda.get_device_capability()[0] >= 10:
    block_DV = 32
```

at `block_DV=32` total smem usage is ~85 KiB, fits cleanly. We tried
`block_DV=64` with all buffers single-buffered (~97 KiB) — it works, but is
*slower* (1.06× vs 1.73× hybrid) because losing the v/a producer-consumer
double-buffering hurts more than the larger tile helps.

The current configuration is **hybrid**: q/k/g/b/p/h/o/vd/vn single-buffered,
v_shared/a_shared kept double-buffered (those two are consumer-modified mid-
iteration, so single-buffering them races).

## 3. `intra_card_cp_preprocess` runs `prepare_h` which won't fit

**File:** `flash_qla/ops/gated_delta_rule/chunk/__init__.py`

The "auto context-parallelism" preprocessing path calls a `prepare_h` kernel
that allocates >100 KiB of shmem. That kernel doesn't fit on Blackwell
consumer.

**Fix:** added an `auto_cp=False` branch that skips that path entirely:

```python
if auto_cp:
    initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
        intra_card_cp_preprocess(...)
    )
else:
    cp_seq_map = None
    raw_cu_seqlens = None
```

vLLM doesn't use intra-card CP (it does its own batching), so this is
free for the vLLM integration. Standalone callers that *want* CP on Blackwell
will need to either (a) re-tune `prepare_h` for 99 KiB or (b) live without it.

## 4. SM90 arch gate bypassed

**Files:**
- `flash_qla/ops/gated_delta_rule/chunk/__init__.py`
- `flash_qla/ops/gated_delta_rule/chunk/cp_context.py`

Both files hard-asserted `compute_version >= "9.0"` AND `< "10.0"` (i.e.
exactly Hopper). Bypassed the gate entirely; TileLang lowers to whatever the
GPU advertises. With the gemm_v2 fix above, the kernel works on Blackwell.

## 5. Two upstream FlashQLA bugs (not Blackwell-specific)

These would bite anyone, on any GPU, but had been masked by other paths in
upstream:

### Bug A: extra arg to `apply()`

**File:** `flash_qla/ops/gated_delta_rule/chunk/__init__.py`, line ~233.

`ChunkGatedDeltaRuleFunction.apply()` was called with 10 positional args, but
`forward()` only takes 9. The extra arg was `use_qk_l2norm_in_kernel`, which
was already applied above in the wrapper:

```python
if use_qk_l2norm_in_kernel:
    q = l2norm(q)
    k = l2norm(k)
```

Dropped the duplicate.

### Bug B: K↔V dim mismatch with vLLM

`vllm.model_executor.layers.mamba.MambaStateShapeCalculator.gated_delta_net_state_shape`
returns `(num_v_heads, head_v_dim, head_k_dim)` — i.e. state is laid out
**(B, H, V, K)**. FlashQLA's `chunk_gated_delta_rule_fwd` allocates state as
**(B, H, K, V)** — K and V swapped.

In Qwen3.6-27B the dims happen to be equal (K = V = 128), so the *shape* check
passes — but the *data* gets transposed. The kernel produces structured
gibberish ("Hello short argument argument" was a typical output before this
fix).

The vLLM mod's `_flashqla_chunk_gated_delta_rule` wrapper transposes the last
two dims on `initial_state` going in and on `final_state` coming out. The
patch lives in `vllm/apply.py`, not in the library — direct callers of the
kernel just need to use the kernel's native `(B, H, K, V)` convention.

## 6. Barrier consolidation (cosmetic)

`data_is_ready` and `data_is_free` were arrays of two barriers indexed by
`i_s % 2`. Collapsed to single barriers with the `i_s % 2` phase passed
directly to `barrier_wait`. No measured performance change; cleaner and
removes one source of off-by-one risk when toggling buffer counts.

## How to extend / re-tune

The hybrid `block_DV=32` config was found by sweeping `{16, 32, 64}` × {full
single, full double, hybrid} on B=1, T=32768, Hv∈{4, 64}. If you have a Hopper
or are tuning for SM_100, the budget loosens — try `block_DV=64` and full
double-buffering. The CTA threading (`threads=512`) is unchanged from upstream
and seems fine for both.
