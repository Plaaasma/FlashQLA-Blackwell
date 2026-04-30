"""Run FlashQLA's chunk_gated_delta_rule on synthetic inputs, no vLLM.

Useful as a 30-second sanity check that the kernel imports + runs + produces
finite outputs on your GPU before you wire it into anything bigger.

    python3 examples/standalone_gdr.py
"""

import torch
from flash_qla import chunk_gated_delta_rule


def main() -> None:
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    cap = torch.cuda.get_device_capability()
    print(f"GPU compute capability: {cap[0]}.{cap[1]}")
    if cap[0] < 9:
        print("WARNING: kernel is targeted at SM_90+; expect compile errors.")

    device = "cuda"
    dtype = torch.bfloat16
    B, T = 1, 4096
    Hq, Hv = 16, 4
    K, V = 128, 128

    torch.manual_seed(0)
    q = torch.randn(B, T, Hq, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, Hq, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, Hv, V, device=device, dtype=dtype) * 0.1
    g = torch.randn(B, T, Hv, device=device, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, Hv, device=device, dtype=dtype))

    print(f"Inputs: B={B} T={T} Hq={Hq} Hv={Hv} K={K} V={V}  ({dtype})")
    print("Compiling kernel (first run does TileLang JIT, ~30s)...")

    o, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    print(f"o:           shape={tuple(o.shape)} dtype={o.dtype} "
          f"finite={torch.isfinite(o).all().item()}")
    print(f"final_state: shape={tuple(final_state.shape)} dtype={final_state.dtype} "
          f"finite={torch.isfinite(final_state).all().item()}")
    print(f"o stats:     mean={o.float().mean().item():+.4f} "
          f"std={o.float().std().item():.4f}")

    # Time a few iterations.
    torch.cuda.synchronize()
    import time
    for _ in range(3):  # warmup
        chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"Avg latency over {iters} iters: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
