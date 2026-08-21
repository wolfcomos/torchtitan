# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused MXFP8 grouped-MLP routed experts over the torchao cuDNN-frontend ops.

The composite autograd Function runs the full routed-expert SwiGLU MLP with
four torchao custom ops that wrap the ``cudnn.grouped_gemm_*_wrapper_sm100``
CuTe DSL kernels (cuDNN frontend >= 1.27, no TransformerEngine involved):

* forward:  casts -> ``torchao::mxfp8_grouped_gemm_swiglu_fwd`` (FC1 grouped
  GEMM + SwiGLU + rowwise/colwise MXFP8 RCEIL quantization of the activation)
  -> ``torchao::mxfp8_grouped_gemm`` FC2.
* backward: casts -> ``torchao::mxfp8_grouped_gemm_dswiglu_bwd`` (FC2 dgrad +
  dSwiGLU + dual quantization) -> ``torchao::mxfp8_grouped_gemm`` FC1
  dgrad -> ``torchao::mxfp8_grouped_gemm_wgrad`` twice.

Every cast in this module is non-CuTe (triton/CUDA-extension torchao kernels);
only the cudnn package's own kernels are CuTe DSL. This module must never call
into ``cute_utils``.

ABI preconditions (metadata-validated by the ops; offset VALUES are the
dispatcher's contract): ``D`` and ``F`` are positive multiples of 128, the
allocated row count is a multiple of 256, and every per-expert row count
``m[g]`` is a nonnegative multiple of **256** — the cuDNN FE kernels hard-code
``FIX_PAD_SIZE = 256``, and per-expert splits that are only 128-multiples
corrupt the chain SILENTLY and NONDETERMINISTICALLY (the corruption locus
migrates between identical-input reruns; no smoke test can prove such a
configuration safe). The %256 guarantee comes from a ``TorchAOTokenDispatcher``
with ``pad_multiple=256``, which the override factory installs itself — hence
it targets ``RoutedExperts.Config`` (the one node owning both the dispatcher
and the inner experts configs).

The fused ``w13`` parameter is stored ``[E, 2F, D]`` in the cuDNN 32-block GLU
row order ``[gate_0..31 | up_0..31 | gate_32..63 | up_32..63 | ...]`` so that
no per-step layout remap exists anywhere: the FC1 kernel consumes the rows
as-is and the wgrad op emits ``dw13`` directly in parameter order. Checkpoints
still save/load the stock ``w1_EFD``/``w3_EFD`` layout through this module's
state-dict hooks.

Activate with ``--override.imports
torchtitan.overrides.mxfp8_grouped_mlp.mxfp8_grouped_experts`` on a STOCK
``RoutedExperts.Config``: the override is self-contained. Its factory swaps
the token dispatcher for the padded TorchAO variant itself and needs no
quantization converter (the composite quantizes every grouped GEMM). There
is no silent fallback: configurations the kernels cannot execute (missing
torchao ops, non-SM100 hardware, converter-quantized or otherwise non-stock
experts, dims violating the 128-alignment contract) raise an actionable
error at config-application time rather than training the unfused path
silently.
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)

# Importing the wrapper module registers the four torchao:: custom ops; the
# cudnn package is only imported lazily inside the op bodies at first launch.
# The module is newer than several torchao releases, so its absence must
# surface as the factory's actionable config-time error (with this reason),
# not an ImportError at override-import time. Tests skip on this flag too.
try:
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_grouped_mlp import (
        _mxfp8_grouped_mlp_kernels_available,
        is_supported,
    )
except ImportError as _exc:
    is_supported = None
    _TORCHAO_GROUPED_MLP_OPS_AVAILABLE = False
    _TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON = (
        "the installed torchao has no torchao.prototype.moe_training.kernels."
        f"mxfp8.cutedsl_grouped_mlp module (a torchao build that ships the "
        f"fused grouped-MLP custom ops is required): {_exc}"
    )
else:
    _TORCHAO_GROUPED_MLP_OPS_AVAILABLE = bool(_mxfp8_grouped_mlp_kernels_available)
    _TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON = (
        ""
        if _TORCHAO_GROUPED_MLP_OPS_AVAILABLE
        else (
            "torchao cuDNN-frontend grouped-MLP ops are unavailable in this "
            "environment (needs the cudnn python package >= 1.27 with the "
            "grouped_gemm_*_wrapper_sm100 kernels)."
        )
    )
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import (
    _to_mxfp8_dim1_kernel_wrapper,
    to_blocked,
)
from torchao.quantization.quantize_.common import KernelPreference

from torchtitan.components.quantization.utils import swap_token_dispatcher
from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.overrides.fused_swiglu import _fuse_w13_grouped_experts_sharding

__all__ = [
    "MXFP8FusedGroupedExperts",
    "mxfp8_fused_grouped_mlp",
    "mxfp8_grouped_experts",
]

_BLOCK_SIZE = 32
_SCALING_MODE = "rceil"
# Per-expert row groups must be 256-multiples (cuDNN FE FIX_PAD_SIZE); feature
# dims must be 128-multiples (blocked-scale tiles). The row guarantee is the
# dispatcher's pad_multiple; the ops re-validate R % 256 statically.
_ROW_ALIGNMENT = 256
_DIM_ALIGNMENT = 128


def _cast_rowwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x32 rowwise RCEIL cast: row-major qdata + whole-matrix blocked scales
    (identical to the per-group concatenation because every per-expert row
    count is a 256-multiple, so 128-row scale tiles never straddle groups)."""
    qdata, scales = triton_to_mxfp8_dim0(t, _BLOCK_SIZE, _SCALING_MODE)
    return qdata, to_blocked(scales)


def _cast_weight_rowwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, N, K]`` quantized along K: contiguous qdata + per-group blocked
    scales for logical ``(N, K/32)`` per group — the rowwise ``b`` operand of
    the fwd/mm ops."""
    qdata, scales = triton_to_mxfp8_dim0(w, _BLOCK_SIZE, _SCALING_MODE)
    return qdata, triton_mx_block_rearrange_per_group_3d(scales)


def _cast_weight_colwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, N, K]`` quantized along N: k-major per-group qdata +
    per-group blocked scales for logical ``(K, N/32)``.

    Batched: ONE (32x1 RCEIL) cast of the flat ``[G*N, K]`` view along dim0
    + ONE ``K_groups`` swizzle with uniform scale-column offsets. Exact
    because N is a 256-multiple, so 32-row quantization blocks never
    straddle groups, and every group's N/32 scale columns are 4-multiples,
    so the swizzle packs the same per-group ``to_blocked`` bytes densely
    from the buffer start. qdata, scales, AND downstream op outputs are
    BITWISE-equal to a naive per-group ``triton_to_mxfp8_dim1`` +
    ``to_blocked`` loop (measured on GB200) at ~4x less time and ~6*G
    fewer launches per weight.

    The cast's native ``[G, N, K]`` view carries an interleaved batch
    stride ``(N, 1, G*N)``, which the cudnn wrappers reject (B must be
    per-group-contiguous, k- or n-major); one fp8 repack to k-major — the
    same major the rowwise casts pass — restores an accepted layout.
    """
    g, n, k = w.shape
    mx = _to_mxfp8_dim1_kernel_wrapper(
        w.reshape(g * n, k),
        _BLOCK_SIZE,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=w.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    scale_offsets = (
        torch.arange(1, g + 1, device=w.device, dtype=torch.int32)
        * (n // _BLOCK_SIZE)
    )
    # Same pow2 quirk as _cast_colwise_grouped: the K_groups swizzle's
    # tl.arange needs a power-of-2 group count; repeated end-offsets are
    # zero-size groups the kernel skips.
    g_pow2 = 1 << (g - 1).bit_length()
    if g_pow2 != g:
        scale_offsets = torch.cat(
            [scale_offsets, scale_offsets[-1:].expand(g_pow2 - g)]
        )
    col_scales = triton_mx_block_rearrange_2d_K_groups(mx.scale, scale_offsets)
    k_pad = -(-k // 128) * 128
    flat = col_scales.reshape(-1)[: k_pad * (g * n // _BLOCK_SIZE)]
    qdata = mx.qdata.view(k, g, n).permute(1, 2, 0).contiguous()
    return qdata, flat.view(g, -1)


def _cast_colwise_grouped(
    t: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged colwise (32x1) RCEIL cast of ``[R, N]`` for the wgrad operands:
    torchao-native qdata (``[R, N]``-logical, ``(1, R)`` strides — the cudnn
    wgrad kernel accepts this major directly, verified on GB200) + PER-GROUP
    blocked scales via ``triton_mx_block_rearrange_2d_K_groups``.

    Quantizing the whole ragged tensor in one launch is safe ONLY because
    every per-expert row count is a 256-multiple, so 32-row quantization
    blocks never straddle an expert boundary.

    The K_groups swizzle's ``tl.arange(0, num_groups)`` requires a power-of-2
    bound, so the scale-column offsets are padded to the next power of 2 by
    repeating the final offset — repeated end-offsets are zero-size groups the
    kernel skips. Its output also carries 4 trailing padding columns per
    group slot (d2h-sync avoidance); with 256-multiple groups the real blocks
    pack densely from the start of the buffer (total real content =
    ``round_up(N, 128) * offsets[-1]/32`` elements <= ``... * R/32``), so the
    flat buffer is statically sliced to ``round_up(N, 128) * R/32`` — the op's
    documented maximum — without any device sync; the wgrad kernel never reads
    past the offsets-bounded span.
    """
    mx = _to_mxfp8_dim1_kernel_wrapper(
        t,
        _BLOCK_SIZE,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=t.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    scale_offsets = offsets // _BLOCK_SIZE
    g = scale_offsets.shape[0]
    g_pow2 = 1 << (g - 1).bit_length()
    if g_pow2 != g:
        scale_offsets = torch.cat(
            [scale_offsets, scale_offsets[-1:].expand(g_pow2 - g)]
        )
    col_scales = triton_mx_block_rearrange_2d_K_groups(mx.scale, scale_offsets)
    r, n = t.shape
    n_pad = -(-n // 128) * 128
    # mx.qdata is [N, R]-shaped; .t() presents the op's [R, N]-logical view.
    return mx.qdata.t(), col_scales.reshape(-1)[: n_pad * (r // _BLOCK_SIZE)]


class _MXFP8GroupedMLP(torch.autograd.Function):
    """Composite MXFP8 grouped SwiGLU MLP over the four cudnn-FE ops.

    All inputs are plain BF16 CUDA tensors (the module prologue casts and
    un-DTensors them); ``dy`` arrives as contiguous BF16 ``[R, D]``. ``w13``
    is ``[G, 2F, D]`` in 32-block GLU row order. All backward-only casts are
    lazy: forward quantizes only what forward consumes (the rowwise views);
    backward requantizes the colwise weight views and the colwise ``x`` from
    the saved BF16 references — safe because the same-step backward always
    precedes the optimizer update (an update in between trips the autograd
    version counter), and cheaper under per-op SAC because the forward (and
    thus any forward-side cast) re-runs in the recompute pass.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        x_row_q, x_row_sf = _cast_rowwise(x)
        w13_row_q, w13_row_sf = _cast_weight_rowwise_3d(w13)
        z, h_row_q, h_row_sf, h_col_q, h_col_sf = (
            torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
                x_row_q,
                x_row_sf,
                w13_row_q,
                w13_row_sf.reshape(-1),
                offsets,
            )
        )
        w2_row_q, w2_row_sf = _cast_weight_rowwise_3d(w2)
        # FC2 forward: b [G, N=D, K=F] rowwise (quantized along F = the
        # contraction), row-major as cast.
        y = torch.ops.torchao.mxfp8_grouped_gemm(
            h_row_q,
            h_row_sf,
            w2_row_q,
            w2_row_sf.reshape(-1),
            offsets,
        )
        # x is saved BF16; its colwise cast is deferred to backward. Under
        # per-op SAC the whole forward re-runs in the recompute pass, so a
        # forward-side cast would execute twice per step for one consumer
        # (the FC1 wgrad) — deferring makes it run exactly once. Safe for the
        # same reason the weight casts are lazy: the same-step backward always
        # precedes the optimizer update.
        ctx.save_for_backward(z, h_col_q, h_col_sf, x, offsets, w13, w2)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        z, h_col_q, h_col_sf, x, offsets, w13, w2 = ctx.saved_tensors
        if x.shape[0] == 0:
            # A rank whose local experts received zero routed tokens: every
            # grad is zero by construction, and torchao's CUDA colwise cast
            # rejects 0-row inputs, so skip the cast/GEMM chain outright.
            # (The forward needs no such guard: its rowwise casts accept 0
            # rows and the ops early-return at R == 0.)
            return torch.empty_like(x), torch.zeros_like(w13), torch.zeros_like(w2), None
        # The casts assert contiguity; dy is contiguous today (BF16 [R, D]
        # stride (D, 1)) but that is a live invariant, not a given.
        dy = dy.contiguous()
        dy_row_q, dy_row_sf = _cast_rowwise(dy)
        # w2 colwise (quantized along D = the dgrad contraction): the bwd op's
        # ABI takes the [G, D, F]-logical cast output as-is.
        w2_col_q, w2_col_sf = _cast_weight_colwise_3d(w2)
        dz_row_q, dz_row_sf, dz_col_q, dz_col_sf = (
            torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
                dy_row_q,
                dy_row_sf,
                w2_col_q,
                w2_col_sf.reshape(-1),
                z,
                offsets,
            )
        )
        # FC1 dgrad: b [G, N=D, K=2F] quantized along 2F. The colwise cast
        # yields [G, 2F, D]; the mm op's b orientation is [G, N, K], so the
        # call site transposes (unlike ``torch._scaled_grouped_mm``, whose
        # [G, K, N] mat2 convention would take the cast output as-is).
        w13_col_q, w13_col_sf = _cast_weight_colwise_3d(w13)
        dx = torch.ops.torchao.mxfp8_grouped_gemm(
            dz_row_q,
            dz_row_sf,
            w13_col_q.transpose(-2, -1),
            w13_col_sf.reshape(-1),
            offsets,
        )
        dy_col_q, dy_col_sf = _cast_colwise_grouped(dy, offsets)
        x_col_q, x_col_sf = _cast_colwise_grouped(x, offsets)
        # dw2 [G, D, F] = dy^T @ h per expert; dw13 [G, 2F, D] = dz^T @ x per
        # expert, landing directly in the 32-block parameter order.
        dw2 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dy_col_q, dy_col_sf, h_col_q, h_col_sf, offsets
        )
        dw13 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dz_col_q, dz_col_sf, x_col_q, x_col_sf, offsets
        )
        return dx, dw13, dw2, None


def mxfp8_fused_grouped_mlp(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Fused MXFP8 grouped SwiGLU MLP: ``x [R, D] -> y [R, D]`` (BF16).

    Args:
        x: BF16 ``[R, D]`` expert-major padded rows; every per-expert group
            is a multiple of 256 rows.
        w13: BF16 ``[G, 2F, D]`` fused gate/up weight in 32-block GLU row
            order (32 gate rows, then the same features' 32 up rows, ...).
        w2: BF16 ``[G, D, F]`` down-projection weight.
        offsets: int32 CUDA ``[G]`` exclusive per-expert end rows,
            ``offsets[-1] <= R``. Rows past ``offsets[-1]`` of ``y`` (and of
            ``dx`` in backward) are left UNWRITTEN.
    """
    return _MXFP8GroupedMLP.apply(x, w13, w2, offsets)


def _stock_pair_to_w13(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Stock ``w1_EFD``/``w3_EFD`` ``[E, F, D]`` pair -> the 32-block GLU
    row-ordered ``w13 [E, 2F, D]`` (the inverse of ``_split_w13_on_save``'s
    view)."""
    e, f, d = w1.shape
    return (
        torch.stack([w1, w3], dim=2)  # [E, F, 2, D]
        .view(e, f // 32, 32, 2, d)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, 2 * f, d)
    )


def _make_w13_init(gate_init, up_init):
    """Initializer for the 32-block-ordered ``w13 [E, 2F, D]`` from the stock
    per-half initializers.

    Each half is initialized IN PLACE through a strided sub-view of ``w13``
    (the same ``(E, F/32, 2, 32, D)`` view the save hook uses), mirroring
    ``_make_fused_gate_up_init``: initializing the parameter itself keeps
    DTensor init semantics under parallelism — shard-distinct, globally
    consistent draws through the DTensor RNG tracker. (Plain-tensor
    temporaries would draw IDENTICAL values on every rank — torchtitan
    seeds all non-PP ranks the same — silently duplicating experts across
    EP/FSDP shards; and a plain ``copy_`` into a DTensor raises.) The cost:
    fan-computing initializers (e.g. ``nn.init.xavier_uniform_``) would see
    the blocked 4-D sub-view geometry instead of stock ``(E, F, D)``; every
    in-tree ``w1_EFD``/``w3_EFD`` initializer is a fixed-std
    ``trunc_normal_``, which is shape-agnostic."""

    def _init(t: torch.Tensor) -> None:
        e, two_f, d = t.shape
        v = t.view(e, two_f // 64, 2, 32, d)
        gate_init(v[:, :, 0])  # gate (stock w1) half
        up_init(v[:, :, 1])  # up (stock w3) half

    return _init


def _w13_grouped_experts_param_init(param_init: dict | None) -> dict | None:
    """Remap ``w1_EFD`` / ``w3_EFD`` initializers onto the 32-block ``w13``.

    Other entries (e.g. ``w2_EDF``) are kept as-is.
    """
    if param_init is None:
        return None
    w1_init = param_init.get("w1_EFD")
    w3_init = param_init.get("w3_EFD")
    fused = {k: v for k, v in param_init.items() if k not in ("w1_EFD", "w3_EFD")}
    if w1_init is not None and w3_init is not None:
        fused["w13"] = _make_w13_init(w1_init, w3_init)
    return fused or None


class MXFP8FusedGroupedExperts(GroupedExperts):
    """Routed experts computed by the cudnn-FE MXFP8 grouped-MLP composite.

    ``w13`` has shape ``(num_experts, 2*hidden_dim, dim)`` in the cuDNN
    32-block GLU row order (NOT the FusedGroupedExperts ``(E, F, 2, D)``
    element interleave); checkpoints save/load the stock ``w1_EFD``/``w3_EFD``
    layout through this class's own hooks. The layout is fixed at
    parameter-registration time so the training loop performs zero per-step
    remaps and the wgrad op emits ``dw13`` directly in parameter order.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        # No new fields in v1. derive() carries param_init / sharding_config /
        # dim / hidden_dim / num_experts from the stock config by name;
        # any future knob must be re-declared here or derive() drops it.
        pass

    def __init__(self, config: Config):
        super().__init__(config)

        # delete separate w1/w3 and fuse in 32-block GLU row order
        del self.w1_EFD
        del self.w3_EFD
        self.w13 = torch.nn.Parameter(
            torch.empty(config.num_experts, 2 * config.hidden_dim, config.dim)
        )

        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w13, DTensor):
            w13 = self.w13.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
        else:
            w13 = self.w13
            w2_EDF = self.w2_EDF

        # The factory gate can only validate the GLOBAL dims (the config
        # carries sharding placements, not mesh degrees), so the local shard
        # dims are validated here at first call: under dense tensor
        # parallelism (expert_parallel_degree=1, tensor_parallel_degree>1)
        # w13/w2 are Shard(1)/Shard(2)-split on hidden_dim, and a TP degree
        # with hidden_dim/tp not a 128-multiple would otherwise fail deep
        # inside the first fused op launch.
        local_f = w13.shape[1] // 2
        local_d = w13.shape[2]
        if local_f % _DIM_ALIGNMENT != 0 or local_d % _DIM_ALIGNMENT != 0:
            raise ValueError(
                f"mxfp8_grouped_experts: the LOCAL expert shard dims "
                f"(D={local_d}, F={local_f} from w13 of local shape "
                f"{tuple(w13.shape)}) must be positive multiples of "
                f"{_DIM_ALIGNMENT} for the fused cuDNN-FE grouped-MLP "
                "kernels. This typically means tensor parallelism split "
                "hidden_dim into a non-128-multiple shard; choose a "
                "tensor_parallel_degree such that hidden_dim / tp stays a "
                f"multiple of {_DIM_ALIGNMENT}, or drop the "
                "mxfp8_grouped_experts override import to use the unfused "
                "MXFP8 path."
            )

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        # The .bfloat16() casts stay OUTSIDE the Function so autograd handles
        # high-precision master-weight configs and dy reaches backward() BF16.
        y_RD = _MXFP8GroupedMLP.apply(
            x_RD.bfloat16(), w13.bfloat16(), w2_EDF.bfloat16(), offsets_E
        )
        return y_RD.type_as(x_RD)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Save the 32-block fused weight as stock ``w1_EFD`` / ``w3_EFD``."""
        w13 = state_dict.pop(f"{prefix}w13")
        e, two_f, d = w13.shape
        f = two_f // 2
        v = w13.view(e, f // 32, 2, 32, d)
        state_dict[f"{prefix}w1_EFD"] = v[:, :, 0].reshape(e, f, d)
        state_dict[f"{prefix}w3_EFD"] = v[:, :, 1].reshape(e, f, d)

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Combine stock ``w1_EFD`` / ``w3_EFD`` into the 32-block ``w13``."""
        w1_key, w3_key = f"{prefix}w1_EFD", f"{prefix}w3_EFD"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = _stock_pair_to_w13(
                state_dict.pop(w1_key), state_dict.pop(w3_key)
            )


@override(
    target=RoutedExperts.Config,
    description="cuDNN-frontend fused MXFP8 grouped-MLP for routed experts.",
)
def mxfp8_grouped_experts(cfg: RoutedExperts.Config) -> RoutedExperts.Config:
    """Swap stock grouped experts for the cudnn-FE fused MXFP8 composite.

    Targets ``RoutedExperts.Config`` because the composite constrains BOTH
    children: the ``inner_experts`` (replaced with the fused module) and the
    ``token_dispatcher``, which the factory swaps for the padded TorchAO
    variant that guarantees the ABI's ``m[g] % 256``. Self-contained and
    fail-loud: no converter is involved, and any configuration the kernels
    cannot execute raises at config-application time instead of silently
    training the unfused path.
    """
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
    ):
        raise ValueError(
            "mxfp8_grouped_experts requires CUDA device capability exactly "
            "(10, 0) (the torchao ops wrap cudnn grouped_gemm_*_wrapper_sm100 "
            "kernels); remove the override or run on supported hardware."
        )
    if not _TORCHAO_GROUPED_MLP_OPS_AVAILABLE:
        raise ValueError(
            f"mxfp8_grouped_experts: {_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON}"
        )
    if type(cfg) is not RoutedExperts.Config:
        raise ValueError(
            "mxfp8_grouped_experts targets the stock RoutedExperts.Config, "
            f"got {type(cfg).__qualname__}; narrow this override's fqns or "
            "remove the conflicting override."
        )
    experts = cfg.inner_experts
    if type(experts) is not GroupedExperts.Config:
        raise ValueError(
            "mxfp8_grouped_experts requires the stock GroupedExperts.Config, "
            f"but inner_experts is {type(experts).__qualname__}. The "
            "composite quantizes every grouped GEMM itself — do not combine "
            "it with the MXFP8 grouped-experts converter."
        )
    # GLOBAL dims only: the config carries sharding placements (e.g. the TP
    # Shard(1) on hidden_dim) but not mesh degrees, so the per-rank shard dims
    # cannot be computed here. MXFP8FusedGroupedExperts.forward re-validates
    # the LOCAL dims at first call and raises with the config fix.
    if not is_supported(experts.dim, experts.hidden_dim):
        raise ValueError(
            f"mxfp8_grouped_experts: is_supported(D={experts.dim}, "
            f"F={experts.hidden_dim}) is False; both dims must be positive "
            f"multiples of {_DIM_ALIGNMENT}."
        )

    # The %256 padding contract is the factory's own work (no converter
    # involved): the cuDNN FE kernels require per-expert groups padded to
    # 256 (FIX_PAD_SIZE) — 128-multiple-only splits corrupt silently and
    # nondeterministically.
    dispatcher = cfg.token_dispatcher
    if isinstance(dispatcher, TorchAOTokenDispatcher.Config):
        dispatcher.pad_multiple = _ROW_ALIGNMENT
    elif isinstance(dispatcher, AllToAllTokenDispatcher.Config):
        swap_token_dispatcher(cfg, pad_multiple=_ROW_ALIGNMENT)
    else:
        raise ValueError(
            f"mxfp8_grouped_experts: token_dispatcher is "
            f"{type(dispatcher).__qualname__}; only the TorchAO padded "
            "dispatcher (swapped in from the stock all-to-all) is validated "
            "for the per-expert 256-row contract."
        )

    fused = derive(experts, MXFP8FusedGroupedExperts.Config)
    # The w1_EFD/w3_EFD -> w13 param-init remap is factory work (it is not
    # inherited through derive()); the sharding remap simply carries w1's
    # layout onto w13 (both rank-3, same shard axes), never a gate:
    # deepseek EP=1 configs carry TP Shard(1) on w1_EFD unconditionally, so
    # raising on a fused-dim Shard would reject exactly the single-GPU debug
    # mode.
    fused.param_init = _w13_grouped_experts_param_init(fused.param_init)
    if fused.sharding_config is not None:
        fused.sharding_config = _fuse_w13_grouped_experts_sharding(
            fused.sharding_config
        )
    cfg.inner_experts = fused
    return cfg
