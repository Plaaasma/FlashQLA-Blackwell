#!/usr/bin/env python3
"""Patch vllm/model_executor/layers/mamba/gdn_linear_attn.py so that the
prefill chunk_gated_delta_rule kernel uses FlashQLA on Blackwell instead
of falling back to FLA Triton.

The upstream ChunkGatedDeltaRule class only knows about FlashInfer
(SM90 path) and FLA Triton (catch-all).  GB10 (SM_120/121) reports
compute capability 12.1 — `is_device_capability(90)` returns False there
even though the GPU is newer than Hopper, so without this patch GB10
falls all the way back to Triton and leaves a 1.73x speedup on the
table.

Edits:
  1. Insert a flash_qla wrapper at module scope that converts kwargs to
     match flash_qla's signature.
  2. Add a forward_flashqla method to ChunkGatedDeltaRule.
  3. Modify __init__ to detect Blackwell consumer (compute_major >= 10)
     and route to forward_flashqla when the user hasn't picked a backend
     explicitly.

Idempotent: guards on a sentinel.
"""
from __future__ import annotations

import sys
from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
GDN = VLLM_ROOT / "model_executor/layers/mamba/gdn_linear_attn.py"
SENTINEL = "# [FLASHQLA PATCH]"


HELPER_BLOCK = f'''

{SENTINEL}
# FlashQLA path — TileLang fused GDN forward, faster than the bundled
# FLA Triton kernel on Blackwell consumer (sm_120/121).  Imported lazily
# so that the import error (if flash_qla isn't installed) doesn't kill
# vLLM startup on systems that don't use this mod.
def _flashqla_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=True,
):
    from flash_qla import chunk_gated_delta_rule as _fqla_kernel
    # vLLM's GDN state is laid out as (B, H, V, K) -- see
    # MambaStateShapeCalculator.gated_delta_net_state_shape -- but
    # FlashQLA's chunk_gated_delta_rule_fwd allocates (B, H, K, V).
    # Transpose the last two dims on the way in and on the way out.
    if initial_state is not None:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    o, final_state = _fqla_kernel(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return o, final_state

'''


# Anchor for inserting helper: just before the @CustomOp.register decorator
HELPER_ANCHOR = '@CustomOp.register("chunk_gated_delta_rule")\n'


# Patch the __init__ to add Blackwell detection.  We replace the entire
# backend-selection block with a version that knows about flashqla.
INIT_OLD = '''    def __init__(self) -> None:
        super().__init__()
        backend_cfg = get_current_vllm_config().additional_config.get(
            "gdn_prefill_backend", "auto"
        )
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel", scope="local")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
                scope="local",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel", scope="local")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )'''

INIT_NEW = '''    def __init__(self) -> None:
        super().__init__()
        backend_cfg = get_current_vllm_config().additional_config.get(
            "gdn_prefill_backend", "auto"
        )
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )
        # ''' + SENTINEL + '''
        # Blackwell consumer (sm_120/121, GB10): use FlashQLA TileLang kernel.
        # is_device_capability(90) returns False on sm_12x because that helper
        # checks for an exact major/minor (Hopper SM 9.0); we look at the major
        # version directly to detect anything Blackwell-or-later.
        try:
            import torch as _torch
            _major, _ = _torch.cuda.get_device_capability(0)
            supports_flashqla = current_platform.is_cuda() and _major >= 10
        except Exception:
            supports_flashqla = False

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            use_flashqla = False
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
            use_flashqla = False
        elif backend == "flashqla":
            use_flashinfer = False
            use_flashqla = supports_flashqla
            if not use_flashqla:
                logger.warning_once(
                    "GDN prefill backend 'flashqla' is selected but "
                    "the current GPU is pre-Blackwell. Falling back to Triton/FLA."
                )
        else:
            # auto: prefer FlashQLA on Blackwell, FlashInfer on Hopper, else Triton.
            use_flashqla = supports_flashqla
            use_flashinfer = supports_flashinfer and not supports_flashqla

        if use_flashqla:
            logger.info_once(
                "Using FlashQLA TileLang GDN prefill kernel (Blackwell)",
                scope="local",
            )
            self._forward_method = self.forward_flashqla
        elif use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel", scope="local")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
                scope="local",
            )
            self._forward_method = self.forward_cuda
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel", scope="local")
            self._forward_method = self.forward_native'''


# Add the forward_flashqla method to ChunkGatedDeltaRule.  We insert it
# right after the `def __init__` block, before `forward_cuda`.  The
# parameter list mirrors forward_cuda's exactly (including chunk_indices /
# chunk_offsets that vLLM threads through but flash_qla doesn't use) so
# call sites continue to work.
METHOD_OLD = '''    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule('''

METHOD_NEW = '''    def forward_flashqla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):  # ''' + SENTINEL + '''
        return _flashqla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule('''


def main() -> int:
    if not GDN.exists():
        print(f"ERROR: missing {GDN}", file=sys.stderr)
        return 1

    src = GDN.read_text()
    if SENTINEL in src:
        print(f"[OK] {GDN.name} already patched")
        return 0

    # Insert helper just before the @CustomOp.register decorator.
    if src.count(HELPER_ANCHOR) != 1:
        print(
            f"ERROR: helper anchor not found in {GDN.name} "
            f"(expected 1 occurrence of '@CustomOp.register(\"chunk_gated_delta_rule\")')",
            file=sys.stderr,
        )
        return 2
    src = src.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)

    for label, old, new in [
        ("init_block", INIT_OLD, INIT_NEW),
        ("forward_flashqla_method", METHOD_OLD, METHOD_NEW),
    ]:
        n = src.count(old)
        if n != 1:
            print(
                f"ERROR: pattern '{label}' expected 1 match, found {n} in {GDN.name}",
                file=sys.stderr,
            )
            return 2
        src = src.replace(old, new, 1)

    GDN.write_text(src)
    print(f"[OK] patched {GDN.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
