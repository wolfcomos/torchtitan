# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 SwiGLU MLPs for routed experts.

The autograd composites behind the ``swiglu`` and ``grouped_gemm_swiglu`` plans
of ``MXFP8GroupedExpertsConverter`` (its ``Config.fusion_plan`` documents what
each plan fuses and which torchao kernels run it). Called from the converter's
``GroupedExperts._grouped_mlp`` override with BF16 CUDA tensors whose expert
token groups are padded to the plan's row multiple.

Tensor shape suffixes:
    R: routed token rows (padded)
    D: model dimension
    F: expert hidden dimension
    E: (local) experts
"""

import spmd_types as spmd
import torch

from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_cuda_3d,
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

from .converter import _FUSED_DIM_ALIGNMENT, _FUSION_PLAN_PAD_MULTIPLES
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


def _swiglu_forward_casts(gate, up):
    # Lazy: the kernel module imports the CuTe DSL runtime at module scope.
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
        gated_act_mxfp8_cutedsl_forward,
    )

    return gated_act_mxfp8_cutedsl_forward(gate, up, rowwise=True, colwise=True)


def _swiglu_backward_casts(grad_h, gate, up):
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
        gated_act_mxfp8_cutedsl_backward,
    )

    return gated_act_mxfp8_cutedsl_backward(
        grad_h, gate, up, rowwise=True, colwise=True
    )


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


def _cast_colwise_k_groups(b, offs):
    # The GEMM-operand colwise cast the existing grouped wgrad path uses, for
    # the `b` side of `_wgrad_k_groups`: (M, Kb) -> qdata (Kb, M) plus per-token-
    # group blocked scales. Cast once when one operand feeds several wgrads.
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
    return b_t_mx.qdata, b_scales


def _wgrad_k_groups(a_qdata, a_scales, b_cast, offs, out_dtype):
    # grad[e] = a[start:end].T @ b[start:end] for each token group. `a` arrives
    # colwise-quantized from the SwiGLU boundary ((M, Ka) with strides (1, M)
    # plus flat blocked scales); `b_cast` is `_cast_colwise_k_groups` of the
    # other operand.
    m, ka = a_qdata.shape
    b_t_qdata, b_scales = b_cast
    a_scales_2d = _reblock_scales_k_groups(a_scales, ka, m, offs)
    return torch._scaled_grouped_mm(
        a_qdata.t(),
        b_t_qdata.transpose(-2, -1),
        a_scales_2d,
        b_scales,
        offs=offs,
        out_dtype=out_dtype,
    )


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUFusionFunction(torch.autograd.Function):
    """``x_RD [R, D] -> y_RD [R, D]`` over the stock MLP's three MXFP8 grouped
    GEMMs with the fused SwiGLU + cast kernel between them.

    ``w1_EFD`` and ``w3_EFD`` are the stock gate and up weights ``(E, F, D)``;
    ``w2_t_EFD`` is the down weight transposed to ``(E, F, D)``;
    ``offs_E`` holds int32 exclusive-end row offsets of the 128-row-padded groups.
    """

    @staticmethod
    def forward(ctx, x_RD, w1_EFD, w3_EFD, w2_t_EFD, offs_E):
        x_RD = x_RD.contiguous()
        # The rowwise cast `_compute_fwd_sm100` would run on BF16 `x_RD`, hoisted
        # so one cast feeds both the gate and the up GEMM.
        x_q, x_scales = mxfp8_quantize_2d_1x32_cutedsl(
            x_RD, scaling_mode=_SCALE_MODE.value.lower(), offs=offs_E
        )
        x_mx = _wrap_rowwise(x_q, x_scales, x_RD.dtype)
        gate = _compute_fwd_sm100(
            x_mx,
            w1_EFD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            x_RD.dtype,
            _SCALE_MODE,
        )
        up = _compute_fwd_sm100(
            x_mx,
            w3_EFD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            x_RD.dtype,
            _SCALE_MODE,
        )
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gate, up)
        y_RD = _compute_fwd_sm100(
            _wrap_rowwise(h_rw, hs_rw, x_RD.dtype),
            w2_t_EFD,
            offs_E,
            _MXFP8_BLOCK_SIZE,
            x_RD.dtype,
            _SCALE_MODE,
        )
        ctx.save_for_backward(
            x_RD, w1_EFD, w3_EFD, w2_t_EFD, offs_E, gate, up, h_cw, hs_cw
        )
        return y_RD

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_RD):
        (
            x_RD,
            w1_EFD,
            w3_EFD,
            w2_t_EFD,
            offs_E,
            gate,
            up,
            h_cw,
            hs_cw,
        ) = ctx.saved_tensors
        grad_y_RD = grad_y_RD.contiguous()
        grad_h = _compute_dgrad_sm100(
            grad_y_RD, w2_t_EFD, offs_E, _MXFP8_BLOCK_SIZE, grad_y_RD.dtype, _SCALE_MODE
        )
        (
            dgate_rw,
            dgate_cw,
            dgs_rw,
            dgs_cw,
            dup_rw,
            dup_cw,
            dus_rw,
            dus_cw,
        ) = _swiglu_backward_casts(grad_h, gate, up)
        grad_x_RD = _compute_dgrad_sm100(
            _wrap_rowwise(dgate_rw, dgs_rw, grad_y_RD.dtype),
            w1_EFD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            grad_y_RD.dtype,
            _SCALE_MODE,
        ) + _compute_dgrad_sm100(
            _wrap_rowwise(dup_rw, dus_rw, grad_y_RD.dtype),
            w3_EFD.transpose(-2, -1),
            offs_E,
            _MXFP8_BLOCK_SIZE,
            grad_y_RD.dtype,
            _SCALE_MODE,
        )
        x_cast = _cast_colwise_k_groups(x_RD, offs_E)
        grad_w1_EFD = _wgrad_k_groups(dgate_cw, dgs_cw, x_cast, offs_E, grad_y_RD.dtype)
        grad_w3_EFD = _wgrad_k_groups(dup_cw, dus_cw, x_cast, offs_E, grad_y_RD.dtype)
        grad_w2_t_EFD = _wgrad_k_groups(
            h_cw,
            hs_cw,
            _cast_colwise_k_groups(grad_y_RD, offs_E),
            offs_E,
            grad_y_RD.dtype,
        )
        return grad_x_RD, grad_w1_EFD, grad_w3_EFD, grad_w2_t_EFD, None


# Local-only for SPMD type checking; see the note on _MXFP8LinearFunction.
spmd.register_local_autograd_function(_MXFP8SwiGLUFusionFunction)


def _validate_inputs(x_RD, w1_EFD, fusion_plan):
    """Static-shape checks at the module's forward: the local expert dims
    (shards under tensor parallelism) and the routing-dependent token count."""
    _, f, d = w1_EFD.shape
    if f % _FUSED_DIM_ALIGNMENT or d % _FUSED_DIM_ALIGNMENT:
        raise ValueError(
            f"MXFP8 fusion_plan={fusion_plan!r} requires the local expert "
            f"dimensions to be multiples of {_FUSED_DIM_ALIGNMENT}, got "
            f"hidden_dim={f}, dim={d}. Choose a tensor_parallel_degree that "
            "keeps the shards aligned, or use fusion_plan='none' for this model."
        )
    # Group boundaries must also be aligned (the dispatcher's pad_multiple
    # guarantees it; checking offs here would sync). M is an unbacked SymInt
    # under compile, so identity tests: literal bools raise, SymBools become
    # deferred runtime asserts. The >= and % 32 forms are implied by the row
    # multiple but recorded separately: downstream kernel wrappers and GEMM
    # metas check exactly those forms, and the symbolic engine matches
    # expressions rather than deriving them.
    rows = _FUSION_PLAN_PAD_MULTIPLES[fusion_plan]
    m = x_RD.shape[0]
    for cond, requirement in (
        (m >= rows, f"at least {rows}"),
        (
            m % rows == 0,
            f"a multiple of {rows} (configure the token dispatcher with "
            f"pad_multiple={rows})",
        ),
        (m % _MXFP8_BLOCK_SIZE == 0, f"a multiple of {_MXFP8_BLOCK_SIZE}"),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8 fusion_plan={fusion_plan!r}: token count {m} must be "
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
    """``[E, N, K]`` quantized along N (torchao's 3D dim1 cast, as in its
    grouped dgrad): per-expert column-major qdata + per-group blocked scales
    for logical ``(K, N/32)`` -- the ops' colwise ``b`` contract, taken as-is
    by the bwd op and transposed for the mm op."""
    return mxfp8_quantize_cuda_3d(
        w,
        _MXFP8_BLOCK_SIZE,
        scale_block_dim1=_MXFP8_BLOCK_SIZE,
        scale_block_dim2=1,
        scaling_mode=_SCALE_MODE.value,
    )


def _cast_colwise_grouped(
    t: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged colwise (32x1) cast of ``[R, N]`` for the wgrad operands: qdata
    ``[R, N]``-logical with ``(1, R)`` strides (the dim1 wrapper hands back the
    ``[N, R]`` view, hence the ``.t()``) plus PER-GROUP blocked scales. One
    launch over the ragged tensor is exact because every group is a
    256-multiple (32-row blocks never straddle experts); the swizzle's trailing
    per-group padding span is dropped by the static slice to the op's
    ``N * R/32`` sizing, without a device sync."""
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
    return mx.qdata.t(), col_scales.reshape(-1)[: n * (r // _MXFP8_BLOCK_SIZE)]


@torch._dynamo.allow_in_graph
class _MXFP8GroupedGemmSwiGLUFusionFunction(torch.autograd.Function):
    """``x_RD [R, D] -> y_RD [R, D]`` over the four fused cuDNN grouped-GEMM ops.

    ``w13_E2FD`` is ``(E, 2F, D)`` in 32-block GLU row order (see
    ``_pack_w13_blocks``); ``w2_EDF`` is the stock down weight; ``offsets_E``
    holds int32 exclusive-end offsets of the 256-row-padded groups,
    ``offsets_E[-1] <= R``; rows past it in ``y_RD``/``grad_x_RD`` are unwritten.
    Forward quantizes only the rowwise views; backward recasts the colwise
    weight views and ``x`` from the saved BF16 tensors (safe: the same-step
    backward precedes the optimizer update, and cheaper under per-op
    activation checkpointing, where forward re-runs in the recompute pass).
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
        # Down GEMM: b = w2 rowwise [E, N=D, K=F], quantized along F.
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
        # w2 colwise (quantized along D, the dgrad contraction), as cast.
        w2_col_q, w2_col_sf = _cast_weight_colwise_3d(w2_EDF)
        dz_row_q, dz_row_sf, dz_col_q, dz_col_sf = mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
            dy_row_q,
            dy_row_sf,
            w2_col_q,
            w2_col_sf.reshape(-1),
            z,
            offsets_E,
        )
        # Up/gate dgrad: w13 colwise casts to [E, 2F, D]; transpose into the
        # op's [E, N=D, K=2F] b orientation.
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
        # grad_w2 [E, D, F] = dy^T @ h; grad_w13 [E, 2F, D] = dz^T @ x, already
        # in the 32-block w13 order.
        grad_w2_EDF = mxfp8_grouped_gemm_wgrad_cudnn(
            dy_col_q, dy_col_sf, h_col_q, h_col_sf, offsets_E
        )
        grad_w13_E2FD = mxfp8_grouped_gemm_wgrad_cudnn(
            dz_col_q, dz_col_sf, x_col_q, x_col_sf, offsets_E
        )
        return grad_x_RD, grad_w13_E2FD, grad_w2_EDF, None


# Local-only for SPMD type checking; see the note on _MXFP8LinearFunction.
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
