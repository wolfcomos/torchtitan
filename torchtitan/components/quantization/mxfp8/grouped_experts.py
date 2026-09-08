# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 SwiGLU MLPs for routed experts.

Two fusion plans over the stock expert MLP ``silu(x @ w1.T) * (x @ w3.T) @ w2.T``,
``swiglu`` and ``grouped_gemm_swiglu``, selected by
``MXFP8GroupedExpertsConverter.Config.fusion_plan`` (which documents what each
plan fuses and which torchao kernels run it) and run through the
``GroupedExperts._grouped_mlp`` seam.

Both composites take plain BF16 CUDA tensors in expert-major padded row order:
``x_RD`` with every expert's token group zero-padded to the plan's row multiple
(128 for ``swiglu``, 256 for ``grouped_gemm_swiglu``; the padded token
dispatcher provides it), ``offsets_E`` as the int32 exclusive-end cumsum of the
padded group sizes, and the gate/up weights packed at call time from the stock
``w1_EFD``/``w3_EFD`` parameters, so weight gradients land on the stock
parameters and checkpoints stay stock. Configurations the kernels cannot
execute raise; there is no silent fallback.

Tensor shape suffixes:
    R: routed token rows (padded)
    D: model dimension
    F: expert hidden dimension
    E: (local) experts
"""

import spmd_types as spmd
import torch

from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _compute_dgrad_sm100,
    _compute_fwd_sm100,
)
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper, to_blocked
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from .converter import _FUSED_DIM_ALIGNMENT
from .tensor import _MXFP8_BLOCK_SIZE


_ELEM_DTYPE = torch.float8_e4m3fn
_KERNEL_PREFERENCE = KernelPreference.AUTO
# The activation casts take a scaling mode; the cuDNN kernels' in-kernel
# requantization and torchao's grouped GEMM path both use RCEIL. Pin every cast
# to it so all operands of a GEMM round their E8M0 scales the same way.
_SCALE_MODE = ScaleCalculationMode.RCEIL


def _wrap_rowwise(qdata, scales, orig_dtype):
    return MXTensor.from_qdata_and_scales(
        qdata,
        scales,
        orig_dtype,
        block_size=_MXFP8_BLOCK_SIZE,
        kernel_preference=_KERNEL_PREFERENCE,
        is_swizzled_scales=True,
    )


def _swiglu_forward_casts(gated):
    """SwiGLU of the ``[gate | up]`` GEMM output plus its rowwise and
    columnwise MXFP8 casts, in one kernel.

    Returns ``(h_rowwise_qdata, h_colwise_qdata, h_rowwise_scales,
    h_colwise_scales)``.
    """
    # Lazy: the kernel module imports the CuTe DSL runtime at module scope.
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
        gated_act_mxfp8_cutedsl_forward,
    )

    return gated_act_mxfp8_cutedsl_forward(gated, rowwise=True, colwise=True)


def _swiglu_backward_casts(grad_h, gated):
    """Gradient of the SwiGLU w.r.t. its ``[gate | up]`` input plus its
    rowwise and columnwise MXFP8 casts, in one kernel.

    Returns ``(d_rowwise_qdata, d_colwise_qdata, d_rowwise_scales,
    d_colwise_scales)``.
    """
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
        gated_act_mxfp8_cutedsl_backward,
    )

    return gated_act_mxfp8_cutedsl_backward(grad_h, gated, rowwise=True, colwise=True)


def _reblock_scales_k_groups(scales, n_rows, m_total, offs):
    # The CuTeDSL kernels emit blocked scales in full-tensor row-block-major
    # tile order; a 2d-2d grouped GEMM contracting over tokens needs the tiles
    # regrouped per token group (row-block-major within each group). With every
    # group a multiple of 128 rows the two layouts hold identical (128, 4)
    # tiles, so this is a pure tile gather.
    rb = n_rows // 128
    cb = m_total // 128
    ends = (offs // 128).long()
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    sizes = (ends - starts).clamp(min=1)
    t = torch.arange(rb * cb, device=scales.device)
    # The dispatcher may pad the token buffer globally past offs[-1]; tiles in
    # that tail belong to no group and are never read by the grouped GEMM, so
    # clamping them anywhere in bounds is enough to keep the gather valid.
    g = torch.searchsorted(ends * rb, t, right=True).clamp(max=ends.numel() - 1)
    local = t - starts[g] * rb
    src = ((local // sizes[g]) * cb + starts[g] + local % sizes[g]).clamp(
        max=rb * cb - 1
    )
    return scales.view(rb * cb, 512)[src].view(n_rows, -1)


def _wgrad_k_groups(a_qdata, a_scales, b, offs, out_dtype):
    # grad[e] = a[start:end].T @ b[start:end] for each token group. `a` arrives
    # colwise-quantized from the SwiGLU boundary ((M, Ka) with strides (1, M)
    # plus flat blocked scales); `b` gets the same GEMM-operand colwise cast the
    # existing grouped wgrad path uses.
    m, ka = a_qdata.shape
    b_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        b,
        _MXFP8_BLOCK_SIZE,
        elem_dtype=_ELEM_DTYPE,
        hp_dtype=b.dtype,
        kernel_preference=_KERNEL_PREFERENCE,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALE_MODE,
    )
    b_scales = triton_mx_block_rearrange_2d_K_groups(
        b_t_mx.scale, offs // _MXFP8_BLOCK_SIZE
    )
    a_scales_2d = _reblock_scales_k_groups(a_scales, ka, m, offs)
    return torch._scaled_grouped_mm(
        a_qdata.t(),
        b_t_mx.qdata.transpose(-2, -1),
        a_scales_2d,
        b_scales,
        offs=offs,
        out_dtype=out_dtype,
    )


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUFusionFunction(torch.autograd.Function):
    """``x_RD [R, D] -> y_RD [R, D]`` over two MXFP8 grouped GEMMs with the
    fused SwiGLU + cast kernel between them.

    ``w13_E2FD`` is the gate and up weights concatenated per expert to
    ``(E, 2F, D)``; ``w2_t_EFD`` is the down weight transposed to ``(E, F, D)``;
    ``offs_E`` holds int32 exclusive-end row offsets of the 128-row-padded groups.
    """

    @staticmethod
    def forward(ctx, x_RD, w13_E2FD, w2_t_EFD, offs_E):
        x_RD = x_RD.contiguous()
        gated = _compute_fwd_sm100(
            x_RD,
            w13_E2FD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            x_RD.dtype,
            _SCALE_MODE,
        )
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated)
        y_RD = _compute_fwd_sm100(
            _wrap_rowwise(h_rw, hs_rw, x_RD.dtype),
            w2_t_EFD,
            offs_E,
            _MXFP8_BLOCK_SIZE,
            x_RD.dtype,
            _SCALE_MODE,
        )
        ctx.save_for_backward(x_RD, w13_E2FD, w2_t_EFD, offs_E, gated, h_cw, hs_cw)
        return y_RD

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_RD):
        x_RD, w13_E2FD, w2_t_EFD, offs_E, gated, h_cw, hs_cw = ctx.saved_tensors
        grad_y_RD = grad_y_RD.contiguous()
        grad_h = _compute_dgrad_sm100(
            grad_y_RD, w2_t_EFD, offs_E, _MXFP8_BLOCK_SIZE, grad_y_RD.dtype, _SCALE_MODE
        )
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(grad_h, gated)
        grad_x_RD = _compute_dgrad_sm100(
            _wrap_rowwise(d_rw, ds_rw, grad_y_RD.dtype),
            w13_E2FD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            grad_y_RD.dtype,
            _SCALE_MODE,
        )
        grad_w13_E2FD = _wgrad_k_groups(d_cw, ds_cw, x_RD, offs_E, grad_y_RD.dtype)
        grad_w2_t_EFD = _wgrad_k_groups(h_cw, hs_cw, grad_y_RD, offs_E, grad_y_RD.dtype)
        return grad_x_RD, grad_w13_E2FD, grad_w2_t_EFD, None


# Marks the function local-only so SPMD type checking can propagate through
# an autograd function it cannot see into.
# TODO(anijain2305, pianpwk): drop this once register_local_autograd_function
# is removed tree-wide (see the same note on _MXFP8LinearFunction).
spmd.register_local_autograd_function(_MXFP8SwiGLUFusionFunction)


def _validate_swiglu_inputs(x_RD, w1_EFD, w2_t_EFD):
    """Static-shape checks for the ``swiglu`` plan at the module's forward.

    The caller guarantees plain local BF16 CUDA tensors in the module's own
    shapes; only the expert dimensions (local shards under tensor parallelism)
    and the routing-dependent token count need checking.
    """
    _, f, d = w1_EFD.shape
    m = x_RD.shape[0]
    d_out = w2_t_EFD.shape[2]
    if (
        f % _FUSED_DIM_ALIGNMENT != 0
        or d % _FUSED_DIM_ALIGNMENT != 0
        or d_out % _FUSED_DIM_ALIGNMENT != 0
    ):
        raise ValueError(
            "MXFP8 fusion_plan='swiglu' requires the local expert dimensions to "
            f"be multiples of {_FUSED_DIM_ALIGNMENT}, got hidden_dim={f}, dim={d}, "
            f"out_dim={d_out}. Choose a tensor_parallel_degree that keeps the "
            "shards aligned, or use fusion_plan='none' for this model."
        )
    # Group boundaries must additionally be 128-row aligned (the dispatcher's
    # pad_multiple guarantees it); checking offs here would sync. M is
    # routing-dependent under compile (an unbacked SymInt), so the M
    # conditions use identity tests: literal bools raise immediately,
    # symbolic ones become deferred runtime asserts. The m >= 128 and m % 32
    # forms are redundant with m % 128 but recorded separately: downstream
    # kernel wrappers and GEMM metas check exactly those forms, and the
    # symbolic engine matches expressions rather than deriving them from
    # mod-128.
    for cond, requirement in (
        (m >= 128, "at least 128"),
        (
            m % 128 == 0,
            "a multiple of 128 (configure the token dispatcher with "
            "pad_multiple=128)",
        ),
        (m % 32 == 0, "a multiple of 32"),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8 fusion_plan='swiglu': token count {m} must be "
                f"{requirement}; there is no silent fallback."
            )
        if cond is not True:
            torch._check(cond)


def _pad_offsets_pow2(scale_offsets_E: torch.Tensor) -> torch.Tensor:
    # The K_groups swizzle's tl.arange needs a power-of-two group count;
    # repeated end-offsets are zero-size groups the kernel skips.
    e = scale_offsets_E.shape[0]
    e_pow2 = 1 << (e - 1).bit_length()
    if e_pow2 == e:
        return scale_offsets_E
    return torch.cat([scale_offsets_E, scale_offsets_E[-1:].expand(e_pow2 - e)])


def _cast_rowwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x32 rowwise RCEIL cast: row-major qdata + whole-matrix blocked scales
    (identical to the per-group concatenation because every per-expert row
    count is a 256-multiple, so 128-row scale tiles never straddle groups)."""
    qdata, scales = triton_to_mxfp8_dim0(t, _MXFP8_BLOCK_SIZE, _SCALE_MODE.value)
    return qdata, to_blocked(scales)


def _cast_weight_rowwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` quantized along K: contiguous qdata + per-group blocked
    scales for logical ``(N, K/32)`` per group -- the rowwise ``b`` operand of
    the fwd/mm ops."""
    qdata, scales = triton_to_mxfp8_dim0(w, _MXFP8_BLOCK_SIZE, _SCALE_MODE.value)
    return qdata, triton_mx_block_rearrange_per_group_3d(scales)


def _cast_weight_colwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` quantized along N: k-major per-group qdata + per-group
    blocked scales for logical ``(K, N/32)``.

    Batched: ONE (32x1 RCEIL) cast of the flat ``[E*N, K]`` view along dim0
    + ONE ``K_groups`` swizzle with uniform scale-column offsets. Exact
    because N is a 256-multiple, so 32-row quantization blocks never
    straddle groups, and every group's N/32 scale columns are 4-multiples,
    so the swizzle packs the same per-group ``to_blocked`` bytes densely
    from the buffer start (bitwise-equal to a per-group loop, at a fraction
    of the launches).

    The cast's native ``[E, N, K]`` view carries an interleaved batch
    stride ``(N, 1, E*N)``, which the fused-op wrappers reject (B must be
    per-group-contiguous, k- or n-major); one fp8 repack to k-major -- the
    same major the rowwise casts pass -- restores an accepted layout.
    """
    e, n, k = w.shape
    mx = _to_mxfp8_dim1_kernel_wrapper(
        w.reshape(e * n, k),
        _MXFP8_BLOCK_SIZE,
        elem_dtype=_ELEM_DTYPE,
        hp_dtype=w.dtype,
        kernel_preference=_KERNEL_PREFERENCE,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALE_MODE,
    )
    scale_offsets = torch.arange(1, e + 1, device=w.device, dtype=torch.int32) * (
        n // _MXFP8_BLOCK_SIZE
    )
    col_scales = triton_mx_block_rearrange_2d_K_groups(
        mx.scale, _pad_offsets_pow2(scale_offsets)
    )
    k_pad = -(-k // 128) * 128
    flat = col_scales.reshape(-1)[: k_pad * (e * n // _MXFP8_BLOCK_SIZE)]
    qdata = mx.qdata.view(k, e, n).permute(1, 2, 0).contiguous()
    return qdata, flat.view(e, -1)


def _cast_colwise_grouped(
    t: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged colwise (32x1) RCEIL cast of ``[R, N]`` for the wgrad operands:
    torchao-native qdata (``[R, N]``-logical, ``(1, R)`` strides -- the fused
    wgrad kernel accepts this major directly) + PER-GROUP blocked scales via
    ``triton_mx_block_rearrange_2d_K_groups``.

    Quantizing the whole ragged tensor in one launch is safe ONLY because
    every per-expert row count is a 256-multiple, so 32-row quantization
    blocks never straddle an expert boundary.

    The swizzle output carries 4 trailing padding columns per group slot
    (d2h-sync avoidance); with 256-multiple groups the real blocks pack densely
    from the start of the buffer, so the flat buffer is statically sliced to
    ``round_up(N, 128) * R / 32`` -- the op's documented allocated-row sizing --
    without any device sync; the wgrad kernel never reads past the
    offsets-bounded span.
    """
    mx = _to_mxfp8_dim1_kernel_wrapper(
        t,
        _MXFP8_BLOCK_SIZE,
        elem_dtype=_ELEM_DTYPE,
        hp_dtype=t.dtype,
        kernel_preference=_KERNEL_PREFERENCE,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALE_MODE,
    )
    col_scales = triton_mx_block_rearrange_2d_K_groups(
        mx.scale, _pad_offsets_pow2(offsets // _MXFP8_BLOCK_SIZE)
    )
    r, n = t.shape
    n_pad = -(-n // 128) * 128
    # mx.qdata is [N, R]-shaped; .t() presents the op's [R, N]-logical view.
    return mx.qdata.t(), col_scales.reshape(-1)[: n_pad * (r // _MXFP8_BLOCK_SIZE)]


@torch._dynamo.allow_in_graph
class _MXFP8GroupedGemmSwiGLUFusionFunction(torch.autograd.Function):
    """``x_RD [R, D] -> y_RD [R, D]`` over the four fused cuDNN grouped-GEMM ops.

    ``w13_E2FD`` is ``(E, 2F, D)`` in 32-block GLU row order (32 gate rows,
    then the same features' 32 up rows, ...); ``w2_EDF`` is the stock down
    weight; ``offsets_E`` holds int32 exclusive-end row offsets of the
    256-row-padded groups, ``offsets_E[-1] <= R``. Rows past ``offsets_E[-1]``
    of ``y_RD`` (and of ``grad_x_RD``) are left unwritten.

    All backward-only casts are lazy: forward quantizes only what forward
    consumes (the rowwise views); backward requantizes the colwise weight
    views and the colwise ``x`` from the saved BF16 references -- safe because
    the same-step backward always precedes the optimizer update (an update in
    between trips the autograd version counter), and cheaper under per-op
    activation checkpointing because the forward re-runs in the recompute pass.
    """

    @staticmethod
    def forward(ctx, x_RD, w13_E2FD, w2_EDF, offsets_E):
        from torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp import (
            mxfp8_grouped_gemm_cudnn,
            mxfp8_grouped_gemm_swiglu_fwd_cudnn,
        )

        x_RD = x_RD.contiguous()
        x_row_q, x_row_sf = _cast_rowwise(x_RD)
        w13_row_q, w13_row_sf = _cast_weight_rowwise_3d(w13_E2FD)
        z, h_row_q, h_row_sf, h_col_q, h_col_sf = mxfp8_grouped_gemm_swiglu_fwd_cudnn(
            x_row_q,
            x_row_sf,
            w13_row_q,
            w13_row_sf.reshape(-1),
            offsets_E,
        )
        w2_row_q, w2_row_sf = _cast_weight_rowwise_3d(w2_EDF)
        # Down GEMM: b [E, N=D, K=F] rowwise (quantized along F = the
        # contraction), row-major as cast.
        y_RD = mxfp8_grouped_gemm_cudnn(
            h_row_q,
            h_row_sf,
            w2_row_q,
            w2_row_sf.reshape(-1),
            offsets_E,
        )
        ctx.save_for_backward(z, h_col_q, h_col_sf, x_RD, offsets_E, w13_E2FD, w2_EDF)
        return y_RD

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_RD):
        from torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp import (
            mxfp8_grouped_gemm_cudnn,
            mxfp8_grouped_gemm_dswiglu_bwd_cudnn,
            mxfp8_grouped_gemm_wgrad_cudnn,
        )

        z, h_col_q, h_col_sf, x_RD, offsets_E, w13_E2FD, w2_EDF = ctx.saved_tensors
        grad_y_RD = grad_y_RD.contiguous()
        dy_row_q, dy_row_sf = _cast_rowwise(grad_y_RD)
        # w2 colwise (quantized along D = the dgrad contraction): the bwd op's
        # ABI takes the [E, D, F]-logical cast output as-is.
        w2_col_q, w2_col_sf = _cast_weight_colwise_3d(w2_EDF)
        dz_row_q, dz_row_sf, dz_col_q, dz_col_sf = mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
            dy_row_q,
            dy_row_sf,
            w2_col_q,
            w2_col_sf.reshape(-1),
            z,
            offsets_E,
        )
        # Up/gate dgrad: b [E, N=D, K=2F] quantized along 2F. The colwise cast
        # yields [E, 2F, D]; the mm op's b orientation is [E, N, K], so the
        # call site transposes (unlike ``torch._scaled_grouped_mm``, whose
        # [E, K, N] mat2 convention would take the cast output as-is).
        w13_col_q, w13_col_sf = _cast_weight_colwise_3d(w13_E2FD)
        grad_x_RD = mxfp8_grouped_gemm_cudnn(
            dz_row_q,
            dz_row_sf,
            w13_col_q.transpose(-2, -1),
            w13_col_sf.reshape(-1),
            offsets_E,
        )
        dy_col_q, dy_col_sf = _cast_colwise_grouped(grad_y_RD, offsets_E)
        x_col_q, x_col_sf = _cast_colwise_grouped(x_RD, offsets_E)
        # grad_w2 [E, D, F] = dy^T @ h per expert; grad_w13 [E, 2F, D] =
        # dz^T @ x per expert, landing directly in the 32-block w13 operand
        # order.
        grad_w2_EDF = mxfp8_grouped_gemm_wgrad_cudnn(
            dy_col_q, dy_col_sf, h_col_q, h_col_sf, offsets_E
        )
        grad_w13_E2FD = mxfp8_grouped_gemm_wgrad_cudnn(
            dz_col_q, dz_col_sf, x_col_q, x_col_sf, offsets_E
        )
        return grad_x_RD, grad_w13_E2FD, grad_w2_EDF, None


# Marks the function local-only so SPMD type checking can propagate through
# an autograd function it cannot see into.
# TODO(anijain2305, pianpwk): drop this once register_local_autograd_function
# is removed tree-wide (see the same note on _MXFP8LinearFunction).
spmd.register_local_autograd_function(_MXFP8GroupedGemmSwiGLUFusionFunction)


def _pack_w13_blocks(w1_EFD: torch.Tensor, w3_EFD: torch.Tensor) -> torch.Tensor:
    # Stock (E, F, D) gate/up pair packed to the fused kernels' 32-block GLU
    # row-ordered (E, 2F, D): 32 gate rows, then the same features' 32 up
    # rows, ... The reshape of the permuted view copies.
    e, f, d = w1_EFD.shape
    return (
        torch.stack([w1_EFD, w3_EFD], dim=2)  # (E, F, 2, D)
        .view(e, f // _MXFP8_BLOCK_SIZE, _MXFP8_BLOCK_SIZE, 2, d)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, 2 * f, d)
    )


def _validate_grouped_gemm_inputs(x_RD: torch.Tensor, w1_EFD: torch.Tensor) -> None:
    """Static-shape checks for the ``grouped_gemm_swiglu`` plan at the
    module's forward.

    The converter validates the GLOBAL dims at config time; the LOCAL shard
    dims are only known here (under dense tensor parallelism the weights are
    Shard-split on hidden_dim). The total row count is checked against the
    256-row contract with the same identity-test / ``torch._check`` idiom as
    the ``swiglu`` plan; the per-group alignment is the padded dispatcher's
    guarantee (the cuDNN ops do not validate, and 128-only groups corrupt
    silently).
    """
    local_f, local_d = w1_EFD.shape[1], w1_EFD.shape[2]
    if local_f % _FUSED_DIM_ALIGNMENT != 0 or local_d % _FUSED_DIM_ALIGNMENT != 0:
        raise ValueError(
            "MXFP8 fusion_plan='grouped_gemm_swiglu' requires the local expert "
            f"dimensions to be multiples of {_FUSED_DIM_ALIGNMENT}, got hidden_dim="
            f"{local_f}, dim={local_d} from w1_EFD of local shape "
            f"{tuple(w1_EFD.shape)}. Choose a tensor_parallel_degree that "
            "keeps hidden_dim / tp aligned, or use fusion_plan='none' for this "
            "model."
        )
    m = x_RD.shape[0]
    for cond, requirement in (
        (m >= 256, "at least 256"),
        (
            m % 256 == 0,
            "a multiple of 256 (configure the token dispatcher with "
            "pad_multiple=256)",
        ),
        (m % 32 == 0, "a multiple of 32"),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8 fusion_plan='grouped_gemm_swiglu': token count {m} must "
                f"be {requirement}; there is no silent fallback."
            )
        if cond is not True:
            torch._check(cond)
