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
# vLLM moved the ChunkGatedDeltaRule CustomOp around between releases:
#   - <= 0.19:  model_executor/layers/mamba/gdn_linear_attn.py
#   - >= 0.20:  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
#               (the GDN code was split into a gdn/ subpackage per model
#               family: base.py, qwen_gdn_linear_attn.py, kimi_*, olmo_*).
# We try each known location and patch the first one that actually defines
# the chunk_gated_delta_rule CustomOp (detected via HELPER_ANCHOR below).
GDN_CANDIDATES = [
    VLLM_ROOT / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    VLLM_ROOT / "model_executor/layers/mamba/gdn_linear_attn.py",
]
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
        # version directly to detect anything Blackwell-or-later.  We ALSO
        # require that `flash_qla` is importable — without it the forward
        # method would crash at warmup time.  The mod's run.sh installs the
        # wheel, but if someone applies this patch by other means (manual
        # apply.py invocation, copying files, etc.) the wheel may be missing.
        try:
            import torch as _torch
            _major, _ = _torch.cuda.get_device_capability(0)
            _has_blackwell = current_platform.is_cuda() and _major >= 10
        except Exception:
            _has_blackwell = False
        try:
            import importlib as _importlib
            _importlib.import_module("flash_qla")
            _has_flashqla_module = True
        except ImportError:
            _has_flashqla_module = False
        supports_flashqla = _has_blackwell and _has_flashqla_module

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
                if not _has_blackwell:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but "
                        "the current GPU is pre-Blackwell. Falling back to "
                        "Triton/FLA."
                    )
                else:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but "
                        "the `flash_qla` module is not installed. Falling "
                        "back to Triton/FLA. Install via the flashqla mod "
                        "or `pip install flash_qla`."
                    )
        else:
            # auto: prefer FlashQLA on Blackwell, FlashInfer on Hopper, else Triton.
            use_flashqla = supports_flashqla
            use_flashinfer = supports_flashinfer and not supports_flashqla
            if _has_blackwell and not _has_flashqla_module:
                logger.warning_once(
                    "FlashQLA patch is present but `flash_qla` module is "
                    "not installed; falling back to Triton/FLA. Install "
                    "via the flashqla mod or `pip install flash_qla`."
                )

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


# V2 — matches upstream main as of 2026-05-04. Two drifts vs INIT_OLD:
#  (a) backend_cfg fetch was split into 3 lines with an assert, and
#  (b) `scope="local"` was removed from all logger.info_once() calls.
# We emit a matching INIT_NEW_V2 that follows the same surrounding style.
INIT_OLD_V2 = '''    def __init__(self) -> None:
        super().__init__()
        additional_config = get_current_vllm_config().additional_config
        assert isinstance(additional_config, dict)
        backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
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
            logger.info_once("Using FlashInfer GDN prefill kernel")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )'''

INIT_NEW_V2 = '''    def __init__(self) -> None:
        super().__init__()
        additional_config = get_current_vllm_config().additional_config
        assert isinstance(additional_config, dict)
        backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )
        # ''' + SENTINEL + '''
        # Blackwell consumer (sm_120/121, GB10): use FlashQLA TileLang kernel.
        # See INIT_NEW above for the rationale on the GPU + module checks.
        try:
            import torch as _torch
            _major, _ = _torch.cuda.get_device_capability(0)
            _has_blackwell = current_platform.is_cuda() and _major >= 10
        except Exception:
            _has_blackwell = False
        try:
            import importlib as _importlib
            _importlib.import_module("flash_qla")
            _has_flashqla_module = True
        except ImportError:
            _has_flashqla_module = False
        supports_flashqla = _has_blackwell and _has_flashqla_module

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
                if not _has_blackwell:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but "
                        "the current GPU is pre-Blackwell. Falling back to "
                        "Triton/FLA."
                    )
                else:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but "
                        "the `flash_qla` module is not installed. Falling "
                        "back to Triton/FLA. Install via the flashqla mod "
                        "or `pip install flash_qla`."
                    )
        else:
            # auto: prefer FlashQLA on Blackwell, FlashInfer on Hopper, else Triton.
            use_flashqla = supports_flashqla
            use_flashinfer = supports_flashinfer and not supports_flashqla
            if _has_blackwell and not _has_flashqla_module:
                logger.warning_once(
                    "FlashQLA patch is present but `flash_qla` module is "
                    "not installed; falling back to Triton/FLA. Install "
                    "via the flashqla mod or `pip install flash_qla`."
                )

        if use_flashqla:
            logger.info_once(
                "Using FlashQLA TileLang GDN prefill kernel (Blackwell)"
            )
            self._forward_method = self.forward_flashqla
        elif use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
            )
            self._forward_method = self.forward_cuda
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel")
            self._forward_method = self.forward_native'''


# V3 — matches vLLM >= 0.20, where the backend-selection logic was
# extracted into the module-level `_resolve_gdn_prefill_backend()` helper
# and the class __init__ shrank to: resolve -> log -> dispatch.  A third
# backend ("cutedsl") and a forward_cutedsl method were also added.  We
# wrap the dispatch tail so flashqla wins on Blackwell without touching
# the upstream resolver.
INIT_OLD_V3 = '''    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native'''

INIT_NEW_V3 = '''    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)
        # ''' + SENTINEL + '''
        # Blackwell consumer (sm_120/121, GB10): prefer the FlashQLA TileLang
        # kernel.  `_resolve_gdn_prefill_backend` doesn't know about flashqla
        # (it returns one of triton/flashinfer/cutedsl), so we detect Blackwell
        # + an importable flash_qla here and override the dispatch.  `backend`
        # is the *requested* string, so an explicit `gdn_prefill_backend:
        # flashqla` shows up here even though the resolver mapped it to triton.
        # See INIT_NEW (v1) for the rationale on the GPU + module checks.
        try:
            import torch as _torch
            _major, _ = _torch.cuda.get_device_capability(0)
            _has_blackwell = current_platform.is_cuda() and _major >= 10
        except Exception:
            _has_blackwell = False
        try:
            import importlib as _importlib
            _importlib.import_module("flash_qla")
            _has_flashqla_module = True
        except ImportError:
            _has_flashqla_module = False
        _supports_flashqla = _has_blackwell and _has_flashqla_module

        if backend == "flashqla":
            _use_flashqla = _supports_flashqla
            if not _use_flashqla:
                if not _has_blackwell:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but the "
                        "current GPU is pre-Blackwell. Falling back to the "
                        "resolved backend '%s'.",
                        active_backend,
                    )
                else:
                    logger.warning_once(
                        "GDN prefill backend 'flashqla' is selected but the "
                        "`flash_qla` module is not installed. Falling back to "
                        "the resolved backend '%s'. Install via the flashqla "
                        "mod or `pip install flash_qla`.",
                        active_backend,
                    )
        elif backend == "auto":
            # auto: prefer FlashQLA on Blackwell over flashinfer/cutedsl/triton.
            _use_flashqla = _supports_flashqla
            if _has_blackwell and not _has_flashqla_module:
                logger.warning_once(
                    "FlashQLA patch is present but `flash_qla` module is not "
                    "installed; using the resolved backend '%s' instead. "
                    "Install via the flashqla mod or `pip install flash_qla`.",
                    active_backend,
                )
        else:
            # An explicit triton/flashinfer/cutedsl request is honoured as-is.
            _use_flashqla = False

        if _use_flashqla:
            logger.info_once(
                "Using FlashQLA TileLang GDN prefill kernel (Blackwell)"
            )
            self.gdn_prefill_backend = "flashqla"
            self._forward_method = self.forward_flashqla
        elif active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
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


# V2 - matches upstream as of 5/12/2026 (vllm commit 8f89381)
METHOD_OLD_V2 = '''    def forward_cuda(
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
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule('''

METHOD_NEW_V2 = '''    def forward_flashqla(
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
        core_attn_out: torch.Tensor | None = None,
    ):  # ''' + SENTINEL + '''
        o, final_state = _flashqla_chunk_gated_delta_rule(
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
        # Mirror forward_cuda: if the caller passed a preallocated output
        # buffer, copy our result into it (current call sites use the return
        # value and leave this None, but match the signature for parity).
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

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
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule('''


def main() -> int:
    # Locate the file that defines the chunk_gated_delta_rule CustomOp.
    # Across vLLM releases it lives at different paths (see GDN_CANDIDATES);
    # pick the first existing candidate that actually contains the anchor.
    existing = [p for p in GDN_CANDIDATES if p.exists()]
    if not existing:
        print(
            "ERROR: no GDN file found. Tried:\n  "
            + "\n  ".join(str(p) for p in GDN_CANDIDATES),
            file=sys.stderr,
        )
        return 1

    gdn = None
    for p in existing:
        if HELPER_ANCHOR in p.read_text():
            gdn = p
            break
    if gdn is None:
        # Fall back to the first existing file so the anchor error below
        # reports against a real path rather than silently bailing.
        gdn = existing[0]

    src = gdn.read_text()
    if SENTINEL in src:
        print(f"[OK] {gdn.name} already patched")
        return 0

    # Insert helper just before the @CustomOp.register decorator.
    if src.count(HELPER_ANCHOR) != 1:
        print(
            f"ERROR: helper anchor not found in {gdn} "
            f"(expected 1 occurrence of '@CustomOp.register(\"chunk_gated_delta_rule\")')",
            file=sys.stderr,
        )
        return 2
    src = src.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)

    # Each label has one or more candidate (old, new) pairs — we try them
    # in order and use the first one that matches exactly once.  This lets
    # the patch survive upstream drift (log-call signature changes, backend
    # resolver refactors, the gdn/ subpackage split) without needing a new
    # mod per vLLM release.  Newer variants are listed last; order among
    # them doesn't matter since at most one matches a given source file.
    candidates = [
        (
            "init_block",
            [
                ("v1", INIT_OLD, INIT_NEW),
                ("v2", INIT_OLD_V2, INIT_NEW_V2),
                ("v3", INIT_OLD_V3, INIT_NEW_V3),
            ],
        ),
        (
            "forward_flashqla_method",
            [
                ("v1", METHOD_OLD, METHOD_NEW),
                ("v2", METHOD_OLD_V2, METHOD_NEW_V2),
            ],
        ),
    ]
    for label, variants in candidates:
        match_counts = []
        applied = False
        for variant_name, old, new in variants:
            n = src.count(old)
            match_counts.append(f"{variant_name}={n}")
            if n == 1:
                src = src.replace(old, new, 1)
                print(f"[OK] applied {label} ({variant_name})")
                applied = True
                break
        if not applied:
            print(
                f"ERROR: no variant of '{label}' matched in {gdn} "
                f"(tried: {', '.join(match_counts)})",
                file=sys.stderr,
            )
            return 2

    gdn.write_text(src)
    print(f"[OK] patched {gdn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
