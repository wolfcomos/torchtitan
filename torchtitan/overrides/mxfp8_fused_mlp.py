# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Composite MXFP8 MLP overrides.

One autograd function covers the full SwiGLU MLP

    x -> MXFP8 w13 GEMM -> [gate | up] -> silu(gate) * up -> MXFP8 w2 GEMM

with both directions of every quantization done by the CuTeDSL kernels. The
modules keep the stock parameters (``w1``/``w2``/``w3`` submodules dense,
``w1_EFD``/``w2_EDF``/``w3_EFD`` grouped) -- checkpoints, initialization, and
sharding are exactly the stock modules' -- and stack the gate and up weights
into the composite's fused ``w13`` operand at forward time. The
``fuse_activation`` flag selects how the activation boundary is quantized:
``True`` runs the unified SwiGLU+MXFP8 kernel (the BF16 activation never
reaches global memory); ``False`` materializes it in BF16 and quantizes with
the standalone 1x32 / 32x1 kernels. Everything outside that boundary is
identical between the two modes, so an A/B comparison isolates the fused
kernel. Configurations the kernels cannot execute (missing CuTeDSL runtime,
DTensor operands, non-BF16 dtypes, dimensions violating the 128-alignment
contract) raise an actionable error; there is no silent fallback.

``mxfp8_fused_mlp`` (dense ``FeedForward``) builds :class:`MXFP8FusedMLP`;
``mxfp8_fused_grouped_mlp`` (``RoutedExperts``) builds
:class:`MXFP8FusedGroupedMLP` and swaps the token dispatcher for the padded
variant the selected grouped composite requires. Activate by naming the
factories in ``--override.imports``; both accept a ``fuse_activation`` kwarg
(and the grouped factory ``fusion_plan``) via ``(target, kwargs)`` imports
entries. The composites quantize every GEMM themselves, so do not combine
these overrides with the MXFP8 linear / grouped-experts converters (the
factories raise).

The grouped override selects its composite with the ``fusion_plan`` kwarg:

* ``"swiglu"`` (default): the composite above -- ``torch._scaled_grouped_mm``
  FC1/FC2 around the SwiGLU+MXFP8 boundary. Token groups are padded to
  multiples of 128.
* ``"grouped_gemm_swiglu"``: the whole expert MLP runs on the four fused
  torchao grouped-GEMM ops (``torchao::mxfp8_grouped_gemm_swiglu_fwd`` /
  ``mxfp8_grouped_gemm`` / ``mxfp8_grouped_gemm_dswiglu_bwd`` /
  ``mxfp8_grouped_gemm_wgrad``), so the FC1 GEMM, the SwiGLU boundary, and
  every activation quantization are in-kernel. The stock ``w1_EFD``/``w3_EFD``
  weights are packed at forward time into the kernels' 32-block GLU
  row-ordered ``[E, 2F, D]`` operand, and token groups must be padded to
  multiples of 256 (128-only splits corrupt silently and
  nondeterministically). ``fuse_activation`` has no unfused arm here -- the
  factory raises on ``fuse_activation=False`` -- and the factory raises
  actionably when the torchao ops are unavailable.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _compute_dgrad_sm100,
    _compute_fwd_sm100,
)

# Importing the wrapper module registers the four torchao:: custom ops the
# grouped_gemm_swiglu plan runs on; the cudnn python package is only imported
# lazily inside the op bodies at first launch. The module is newer than
# several torchao releases, so its absence must surface as the factory's
# actionable config-time error (with this reason), not an ImportError at
# override-import time. Tests skip on this flag too.
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
            "torchao fused grouped-MLP ops are unavailable in this "
            "environment (needs the cudnn python package >= 1.27 with the "
            "grouped_gemm_*_wrapper_sm100 kernels)."
        )
    )
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import (
    _to_mxfp8_dim1_kernel_wrapper,
    to_blocked,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from torchtitan.components.quantization.utils import swap_token_dispatcher
from torchtitan.config import derive, override
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.tools.utils import has_cuda_capability

__all__ = [
    "MXFP8FusedGroupedMLP",
    "MXFP8FusedMLP",
    "mxfp8_fused_grouped_mlp",
    "mxfp8_fused_mlp",
    "mxfp8_mlp_w13",
]

_BLOCK_SIZE = 32
_ELEM_DTYPE = torch.float8_e4m3fn
_KERNEL_PREFERENCE = KernelPreference.AUTO
_SCALE_MODE = ScaleCalculationMode.RCEIL
_INT32_MAX = 2**31 - 1
# grouped_gemm_swiglu plan: per-expert row groups must be 256-multiples
# (the fused kernels hard-code FIX_PAD_SIZE = 256); feature dims must be
# 128-multiples (blocked-scale tiles). The row guarantee is the dispatcher's
# pad_multiple; the ops re-validate R % 256 statically.
_ROW_ALIGNMENT = 256
_DIM_ALIGNMENT = 128


def _wrap_rowwise(qdata, scales, orig_dtype):
    return MXTensor.from_qdata_and_scales(
        qdata,
        scales,
        orig_dtype,
        block_size=_BLOCK_SIZE,
        kernel_preference=_KERNEL_PREFERENCE,
        is_swizzled_scales=True,
    )


def _wrap_colwise(qdata, scales, orig_dtype):
    # Colwise kernel outputs are (M, N) with strides (1, M); wrapping the
    # transpose keeps qdata row-major, which torch.mm's MXFP8 dispatch
    # requires. The flat 1D blocked scales are unaffected by the transpose.
    return _wrap_rowwise(qdata.t(), scales, orig_dtype)


def _mx_rowwise(t):
    qdata, scales = mxfp8_quantize_2d_1x32_cutedsl(t, scaling_mode=_SCALE_MODE.value)
    return _wrap_rowwise(qdata, scales, t.dtype)


def _mx_colwise(t):
    # Returns the MXTensor for t.t() quantized along t's rows (32x1 blocks).
    return _to_mxfp8_dim1_kernel_wrapper(
        t,
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        t.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUTEDSL,
        _SCALE_MODE,
    )


def _pack_w13(w13):
    # (H, 2, D) with w13[:, 0] = gate and w13[:, 1] = up, packed to (2H, D)
    # with all gate rows first -- the layout whose GEMM output feeds the
    # SwiGLU kernel's [gate | up] contract.
    hidden, _, dim = w13.shape
    return w13.transpose(0, 1).reshape(2 * hidden, dim).contiguous()


def _swiglu_forward_hp(gated):
    k = gated.shape[1] // 2
    return (F.silu(gated[:, :k].float()) * gated[:, k:].float()).to(gated.dtype)


def _swiglu_forward_casts(gated, fuse_activation):
    if fuse_activation:
        # Lazy: the kernel module imports the CuTe DSL runtime at module
        # scope; the standalone-cast arm must work without it.
        from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
            gated_act_mxfp8_cutedsl_forward,
        )

        return gated_act_mxfp8_cutedsl_forward(gated, rowwise=True, colwise=True)
    h = _swiglu_forward_hp(gated)
    h_rw, hs_rw = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    h_cw, hs_cw = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    return h_rw, h_cw, hs_rw, hs_cw


def _swiglu_backward_hp(grad_h, gated):
    k = gated.shape[1] // 2
    gate = gated[:, :k].float()
    up = gated[:, k:].float()
    grad_h_f = grad_h.float()
    # Same evaluation order as the unified kernel (which contracts `deriv`
    # into one FMA), so the two modes differ only in sigmoid lowering and
    # that contraction, not in association.
    sigmoid_gate = torch.sigmoid(gate)
    silu = gate * sigmoid_gate
    deriv = gate * (1.0 - sigmoid_gate) + 1.0
    return torch.cat(
        [
            ((grad_h_f * up) * (sigmoid_gate * deriv)).to(gated.dtype),
            (grad_h_f * silu).to(gated.dtype),
        ],
        dim=1,
    )


def _swiglu_backward_casts(grad_h, gated, fuse_activation):
    if fuse_activation:
        from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
            gated_act_mxfp8_cutedsl_backward,
        )

        return gated_act_mxfp8_cutedsl_backward(
            grad_h, gated, rowwise=True, colwise=True
        )
    d = _swiglu_backward_hp(grad_h, gated)
    d_rw, ds_rw = mxfp8_quantize_2d_1x32_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    d_cw, ds_cw = mxfp8_quantize_2d_32x1_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    return d_rw, d_cw, ds_rw, ds_cw


@torch._dynamo.allow_in_graph
class _MXFP8MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, fuse_activation):
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        w13_packed = _pack_w13(w13)
        gated = torch.mm(_mx_rowwise(x2d), _mx_rowwise(w13_packed).t())
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = torch.mm(_wrap_rowwise(h_rw, hs_rw, x2d.dtype), _mx_rowwise(w2).t())
        ctx.save_for_backward(x2d, w13_packed, w2, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w13_packed, w2, gated, h_cw, hs_cw = ctx.saved_tensors
        hidden = w13_packed.shape[0] // 2
        go = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_h = torch.mm(_mx_rowwise(go), _mx_colwise(w2).t())
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = torch.mm(
            _wrap_rowwise(d_rw, ds_rw, go.dtype), _mx_colwise(w13_packed).t()
        )
        grad_w13_packed = torch.mm(
            _wrap_colwise(d_cw, ds_cw, go.dtype), _mx_colwise(x2d).t()
        )
        grad_w2 = torch.mm(_mx_colwise(go), _wrap_colwise(h_cw, hs_cw, go.dtype).t())
        grad_w13 = grad_w13_packed.view(2, hidden, -1).transpose(0, 1).contiguous()
        return grad_x.reshape(ctx.x_shape), grad_w13, grad_w2, None


def _require_kernels(op_name):
    if not _mxfp8_cutedsl_kernels_available:
        raise NotImplementedError(
            f"{op_name} requires the MXFP8 CuTeDSL kernels (nvidia-cutlass-dsl "
            "on an SM100-class GPU); install the runtime or exclude this module "
            "from the MXFP8 converter."
        )


def _validate_dense_inputs(x, w13, w2):
    _require_kernels("mxfp8_mlp_w13")
    if isinstance(x, DTensor) or isinstance(w13, DTensor) or isinstance(w2, DTensor):
        raise ValueError(
            "mxfp8_mlp_w13 takes plain local tensors, not DTensor; pass "
            "local shards or exclude this module from the fused MXFP8 path."
        )
    if not x.is_cuda:
        raise ValueError(
            f"mxfp8_mlp_w13 requires CUDA tensors, got device {x.device}"
        )
    if (
        x.dtype != torch.bfloat16
        or w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        raise ValueError(
            "mxfp8_mlp_w13 requires BF16 inputs and weights, got "
            f"x={x.dtype}, w13={w13.dtype}, down_weight={w2.dtype}"
        )
    if w13.ndim != 3 or w13.shape[1] != 2 or w2.ndim != 2:
        raise ValueError(
            "expected w13 of shape (H, 2, D) and down_weight of shape (D_out, H), "
            f"got w13={tuple(w13.shape)}, down_weight={tuple(w2.shape)}"
        )
    hidden, _, dim = w13.shape
    n = w2.shape[0]
    m = x.numel() // x.shape[-1]
    if x.shape[-1] != dim or w2.shape[1] != hidden:
        raise ValueError(
            f"shape mismatch: x={tuple(x.shape)}, w13={tuple(w13.shape)}, "
            f"down_weight={tuple(w2.shape)}"
        )
    if m % 128 != 0 or hidden % 128 != 0 or dim % 128 != 0 or n % 128 != 0:
        raise ValueError(
            "the MXFP8 CuTeDSL kernels require every dimension to be a multiple "
            f"of 128, got tokens={m}, hidden={hidden}, dim={dim}, d_out={n}; "
            "exclude this module from the fused MXFP8 path if its shapes cannot "
            "satisfy this."
        )
    # 32-bit index-math limit over BOTH A/B arms: the unfused arm's standalone
    # casts of the (m, 2*hidden) backward tensor reach element 2*hidden*m - 1,
    # past the unified kernel's own bound, and those kernels do not validate;
    # gating on the max keeps the two arms' accepted shapes identical.
    if 2 * hidden * m - 1 > _INT32_MAX:
        raise ValueError(
            "tokens*hidden exceeds the kernels' 32-bit index math: "
            f"2*{hidden}*{m} - 1 > {_INT32_MAX}"
        )


def mxfp8_mlp_w13(x, w13, down_weight, *, fuse_activation=True):
    """Dense MXFP8 SwiGLU MLP with a fused (H, 2, D) w13 weight.

    ``x`` is BF16 of shape (..., D); ``w13[:, 0]`` is the gate (w1) and
    ``w13[:, 1]`` the up (w3) projection; ``down_weight`` is (D_out, H).
    Returns a BF16 tensor of shape (..., D_out). Raises instead of falling
    back when the kernels are unavailable or the inputs violate their
    contract (every dimension must be a multiple of 128).
    """
    _validate_dense_inputs(x, w13, down_weight)
    return _MXFP8MLP.apply(x, w13, down_weight, fuse_activation)


def _pack_w13_grouped(w13):
    # (E, F, 2, D) with w13[:, :, 0] = gate and w13[:, :, 1] = up, packed to
    # (E, 2F, D) with all gate rows first per expert (the [gate | up] layout
    # the SwiGLU kernel consumes). The reshape of the transposed view copies.
    e, f, _, d = w13.shape
    return w13.transpose(1, 2).reshape(e, 2 * f, d)


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
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        b.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUDA,
        _SCALE_MODE,
    )
    b_scales = triton_mx_block_rearrange_2d_K_groups(b_t_mx.scale, offs // _BLOCK_SIZE)
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
class _MXFP8GroupedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2_t, offs, fuse_activation):
        x = x.contiguous()
        w13_packed = _pack_w13_grouped(w13)
        gated = _compute_fwd_sm100(
            x, w13_packed.transpose(-2, -1), offs, _BLOCK_SIZE, x.dtype, _SCALE_MODE
        )
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = _compute_fwd_sm100(
            _wrap_rowwise(h_rw, hs_rw, x.dtype),
            w2_t,
            offs,
            _BLOCK_SIZE,
            x.dtype,
            _SCALE_MODE,
        )
        ctx.save_for_backward(x, w13_packed, w2_t, offs, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w13_packed, w2_t, offs, gated, h_cw, hs_cw = ctx.saved_tensors
        e, two_f, d = w13_packed.shape
        go = grad_out.contiguous()
        grad_h = _compute_dgrad_sm100(
            go, w2_t, offs, _BLOCK_SIZE, go.dtype, _SCALE_MODE
        )
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = _compute_dgrad_sm100(
            _wrap_rowwise(d_rw, ds_rw, go.dtype),
            w13_packed.transpose(-2, -1),
            offs,
            _BLOCK_SIZE,
            go.dtype,
            _SCALE_MODE,
        )
        grad_w13_packed = _wgrad_k_groups(d_cw, ds_cw, x, offs, go.dtype)
        grad_w2_t = _wgrad_k_groups(h_cw, hs_cw, go, offs, go.dtype)
        grad_w13 = grad_w13_packed.view(e, 2, two_f // 2, d).transpose(1, 2)
        return grad_x, grad_w13, grad_w2_t, None, None


def _validate_grouped_inputs(x, w13, w2_t, offs):
    # The only caller is MXFP8FusedGroupedMLP.forward, which guarantees plain
    # local BF16 tensors in the module's own shapes; only environment, config
    # dims, and the routing-dependent token count need checking.
    _require_kernels("MXFP8FusedGroupedMLP")
    _, f, _, d = w13.shape
    m = x.shape[0]
    d_out = w2_t.shape[2]
    if f % 128 != 0 or d % 128 != 0 or d_out % 128 != 0:
        raise ValueError(
            "the MXFP8 CuTeDSL kernels require expert dimensions to be "
            f"multiples of 128, got hidden={f}, dim={d}, d_out={d_out}; exclude "
            "this module from the fused MXFP8 path if its shapes cannot "
            "satisfy this."
        )
    # Group boundaries must additionally be 128-row aligned (the dispatcher's
    # pad_multiple guarantees it); checking offs here would sync. M is
    # routing-dependent under compile (an unbacked SymInt), so the M
    # conditions use identity tests: literal bools raise immediately,
    # symbolic ones become deferred runtime asserts. The m >= 128 and m % 32
    # forms are redundant with m % 128 but recorded separately: downstream
    # kernel wrappers and GEMM metas check exactly those forms, and the
    # symbolic engine matches expressions rather than deriving them from
    # mod-128. The last condition is the 32-bit index-math limit over BOTH
    # A/B arms (the unfused arm's standalone casts reach element 2*f*m - 1,
    # and those kernels do not validate).
    for cond, requirement in (
        (m >= 128, "at least 128"),
        (
            m % 128 == 0,
            "a multiple of 128 (configure the token dispatcher with "
            "pad_multiple=128)",
        ),
        (m % 32 == 0, "a multiple of 32"),
        (
            2 * f * m - 1 <= _INT32_MAX,
            "small enough for the kernels' 32-bit index math "
            "(2*hidden*tokens - 1 <= 2**31 - 1)",
        ),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8FusedGroupedMLP: token count {m} (hidden={f}) "
                f"must be {requirement}; there is no silent fallback."
            )
        if cond is not True:
            torch._check(cond)


def _cast_rowwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x32 rowwise RCEIL cast: row-major qdata + whole-matrix blocked scales
    (identical to the per-group concatenation because every per-expert row
    count is a 256-multiple, so 128-row scale tiles never straddle groups)."""
    qdata, scales = triton_to_mxfp8_dim0(t, _BLOCK_SIZE, _SCALE_MODE.value)
    return qdata, to_blocked(scales)


def _cast_weight_rowwise_3d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, N, K]`` quantized along K: contiguous qdata + per-group blocked
    scales for logical ``(N, K/32)`` per group — the rowwise ``b`` operand of
    the fwd/mm ops."""
    qdata, scales = triton_to_mxfp8_dim0(w, _BLOCK_SIZE, _SCALE_MODE.value)
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
    stride ``(N, 1, G*N)``, which the fused-op wrappers reject (B must be
    per-group-contiguous, k- or n-major); one fp8 repack to k-major — the
    same major the rowwise casts pass — restores an accepted layout.
    """
    g, n, k = w.shape
    mx = _to_mxfp8_dim1_kernel_wrapper(
        w.reshape(g * n, k),
        _BLOCK_SIZE,
        elem_dtype=_ELEM_DTYPE,
        hp_dtype=w.dtype,
        kernel_preference=_KERNEL_PREFERENCE,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALE_MODE,
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
    torchao-native qdata (``[R, N]``-logical, ``(1, R)`` strides — the fused
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
        elem_dtype=_ELEM_DTYPE,
        hp_dtype=t.dtype,
        kernel_preference=_KERNEL_PREFERENCE,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=_SCALE_MODE,
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


@torch._dynamo.allow_in_graph
class _MXFP8GroupedGemmMLP(torch.autograd.Function):
    """Fully-fused MXFP8 grouped SwiGLU MLP over the four torchao fused
    grouped-GEMM ops: ``x [R, D] -> y [R, D]`` (BF16).

    All inputs are plain BF16 CUDA tensors (the module prologue casts and
    un-DTensors them): expert-major padded rows ``x [R, D]`` with every
    per-expert group a multiple of 256 rows; ``w13 [G, 2F, D]`` in 32-block
    GLU row order (32 gate rows, then the same features' 32 up rows, ...);
    ``w2 [G, D, F]``; int32 CUDA ``offsets [G]`` exclusive per-expert end
    rows, ``offsets[-1] <= R``. Rows past ``offsets[-1]`` of ``y`` (and of
    ``dx`` in backward) are left UNWRITTEN; ``dy`` arrives as contiguous
    BF16 ``[R, D]``. All backward-only casts are lazy: forward quantizes
    only what forward consumes (the rowwise views); backward requantizes the
    colwise weight views and the colwise ``x`` from the saved BF16
    references — safe because the same-step backward always precedes the
    optimizer update (an update in between trips the autograd version
    counter), and cheaper under per-op SAC because the forward (and thus any
    forward-side cast) re-runs in the recompute pass.
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
            return (
                torch.empty_like(x),
                torch.zeros_like(w13),
                torch.zeros_like(w2),
                None,
            )
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
        # expert, landing directly in the 32-block w13 operand order.
        dw2 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dy_col_q, dy_col_sf, h_col_q, h_col_sf, offsets
        )
        dw13 = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
            dz_col_q, dz_col_sf, x_col_q, x_col_sf, offsets
        )
        return dx, dw13, dw2, None


def _pack_w13_blocks(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    # Stock (E, F, D) gate/up pair packed to the fused kernels' 32-block GLU
    # row-ordered (E, 2F, D): 32 gate rows, then the same features' 32 up
    # rows, ... The reshape of the permuted view copies.
    e, f, d = w1.shape
    return (
        torch.stack([w1, w3], dim=2)  # (E, F, 2, D)
        .view(e, f // 32, 32, 2, d)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, 2 * f, d)
    )


class MXFP8FusedMLP(FeedForward):
    """Stock :class:`FeedForward` whose forward runs the composite MXFP8
    SwiGLU MLP, stacking ``w1``/``w3`` into the fused ``w13`` operand.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        fuse_activation: bool = True
        """Quantize the SwiGLU boundary with the unified SwiGLU+MXFP8 kernel
        (False: standalone BF16 + cast kernels; identical GEMMs either way)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.fuse_activation = config.fuse_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "MXFP8FusedMLP does not support DTensor activations (dense "
                "tensor parallelism); narrow the mxfp8_fused_mlp override's "
                "fqns or drop it for this module."
            )
        # (H, 2, D) with [:, 0] = gate (w1) and [:, 1] = up (w3).
        w13 = torch.stack([self.w1.weight, self.w3.weight], dim=1)
        output = mxfp8_mlp_w13(
            x,
            w13,
            self.w2.weight,
            fuse_activation=self.fuse_activation,
        )
        if self.w2.bias is not None:
            output = output + self.w2.bias.to(output.dtype)
        return output


class MXFP8FusedGroupedMLP(GroupedExperts):
    """Routed experts whose forward runs a fusion-plan-selected MXFP8 SwiGLU
    grouped-MLP composite.

    Keeps the stock ``w1_EFD``/``w2_EDF``/``w3_EFD`` parameters and packs the
    gate and up weights into the selected composite's ``w13`` operand at
    forward time: ``(num_experts, hidden_dim, 2, dim)`` element-interleaved
    for the ``swiglu`` plan, ``(num_experts, 2 * hidden_dim, dim)`` in the
    fused kernels' 32-block GLU row order for the ``grouped_gemm_swiglu``
    plan. Requires token groups padded to multiples of 128 (``swiglu``) or
    256 (``grouped_gemm_swiglu``) rows (zero-filled) -- the
    ``mxfp8_fused_grouped_mlp`` factory swaps the token dispatcher
    accordingly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        fuse_activation: bool = True
        """Quantize the SwiGLU boundary with the unified SwiGLU+MXFP8 kernel
        (False: standalone BF16 + cast kernels; identical GEMMs either way).
        Only meaningful for the ``swiglu`` fusion plan."""

        fusion_plan: Literal["swiglu", "grouped_gemm_swiglu"] = "swiglu"
        """How much of the expert MLP one kernel covers: ``swiglu`` fuses the
        activation+quantization boundary between two
        ``torch._scaled_grouped_mm`` GEMMs; ``grouped_gemm_swiglu`` runs the
        whole MLP on the four fused torchao grouped-GEMM ops."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.fuse_activation = config.fuse_activation
        self.fusion_plan = config.fusion_plan

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if self.fusion_plan == "grouped_gemm_swiglu":
            # The factory gate can only validate the GLOBAL dims (the config
            # carries sharding placements, not mesh degrees), so the local
            # shard dims are validated here at first call: under dense tensor
            # parallelism (expert_parallel_degree=1,
            # tensor_parallel_degree>1) the weights are Shard-split on
            # hidden_dim, and a TP degree with hidden_dim/tp not a
            # 128-multiple would otherwise fail deep inside the first fused
            # op launch.
            local_f, local_d = w1_EFD.shape[1], w1_EFD.shape[2]
            if local_f % _DIM_ALIGNMENT != 0 or local_d % _DIM_ALIGNMENT != 0:
                raise ValueError(
                    f"MXFP8FusedGroupedMLP: the LOCAL expert shard dims "
                    f"(D={local_d}, F={local_f} from w1_EFD of local shape "
                    f"{tuple(w1_EFD.shape)}) must be positive multiples of "
                    f"{_DIM_ALIGNMENT} for the fused grouped-MLP kernels. "
                    "This typically means tensor parallelism split "
                    "hidden_dim into a non-128-multiple shard; choose a "
                    "tensor_parallel_degree such that hidden_dim / tp stays "
                    f"a multiple of {_DIM_ALIGNMENT}, or drop the "
                    "grouped_gemm_swiglu fusion plan for this module."
                )
            # The .bfloat16() casts stay OUTSIDE the Function so autograd
            # handles high-precision master-weight configs and dy reaches
            # backward() BF16; dw13 flows back through the pack to the stock
            # parameters.
            y_RD = _MXFP8GroupedGemmMLP.apply(
                x_RD.bfloat16(),
                _pack_w13_blocks(w1_EFD, w3_EFD).bfloat16(),
                w2_EDF.bfloat16(),
                offsets_E,
            )
            return y_RD.type_as(x_RD)

        x = x_RD.bfloat16()
        # (E, F, 2, D) with [:, :, 0] = gate (w1_EFD) and [:, :, 1] = up
        # (w3_EFD).
        w13 = torch.stack([w1_EFD, w3_EFD], dim=2).bfloat16()
        w2_t = w2_EDF.bfloat16().transpose(-2, -1)
        _validate_grouped_inputs(x, w13, w2_t, offsets_E)
        return _MXFP8GroupedMLP.apply(
            x,
            w13,
            w2_t,
            offsets_E,
            self.fuse_activation,
        ).type_as(x_RD)


@override(
    target=FeedForward.Config,
    description="Dense SwiGLU FFN via the composite MXFP8 MLP.",
)
def mxfp8_fused_mlp(
    cfg: FeedForward.Config,
    *,
    fuse_activation: bool = True,
) -> "MXFP8FusedMLP.Config":
    # Config-application-time gate; the composite re-validates at runtime.
    if not has_cuda_capability(10, 0):
        raise ValueError(
            "mxfp8_fused_mlp requires SM100 or later; remove the override "
            "or run on supported hardware."
        )
    # Composing with another FFN variant or a linear quantization converter
    # is a config error, not a silent no-op.
    if type(cfg) is not FeedForward.Config:
        raise ValueError(
            "mxfp8_fused_mlp targets the stock FeedForward.Config, got "
            f"{type(cfg).__qualname__}; narrow this override's fqns or remove "
            "the conflicting override/converter."
        )
    for name in ("w1", "w2", "w3"):
        sub = getattr(cfg, name)
        if type(sub) is not Linear.Config:
            raise ValueError(
                "mxfp8_fused_mlp requires stock Linear.Config projections, "
                f"but {name} is {type(sub).__qualname__}. The composite "
                "quantizes every GEMM itself -- do not combine it with a "
                "linear quantization converter on the same module."
            )
    if cfg.w1.bias or cfg.w3.bias:
        raise ValueError(
            "mxfp8_fused_mlp supports a bias on w2 only; the composite has "
            "no w1/w3 bias path."
        )

    return derive(cfg, MXFP8FusedMLP.Config, fuse_activation=fuse_activation)


@override(
    target=RoutedExperts.Config,
    description="Routed experts via a composite MXFP8 grouped MLP "
    "(fusion_plan selects the fused boundary).",
)
def mxfp8_fused_grouped_mlp(
    cfg: RoutedExperts.Config,
    *,
    fuse_activation: bool = True,
    fusion_plan: Literal["swiglu", "grouped_gemm_swiglu"] = "swiglu",
) -> RoutedExperts.Config:
    # Config-application-time gate; the composite re-validates at runtime.
    if not has_cuda_capability(10, 0):
        raise ValueError(
            "mxfp8_fused_grouped_mlp requires SM100 or later; remove the "
            "override or run on supported hardware."
        )
    if fusion_plan not in ("swiglu", "grouped_gemm_swiglu"):
        raise ValueError(
            f"mxfp8_fused_grouped_mlp: unknown fusion_plan {fusion_plan!r}; "
            "expected 'swiglu' or 'grouped_gemm_swiglu'."
        )
    if fusion_plan == "grouped_gemm_swiglu":
        if not fuse_activation:
            raise ValueError(
                "mxfp8_fused_grouped_mlp: fuse_activation=False (the "
                "standalone-cast test-reference arm) exists only for the "
                "'swiglu' fusion plan; the grouped_gemm_swiglu plan always "
                "fuses the activation. Drop fuse_activation=False or use "
                "fusion_plan='swiglu'."
            )
        if not _TORCHAO_GROUPED_MLP_OPS_AVAILABLE:
            raise ValueError(
                "mxfp8_fused_grouped_mlp: "
                f"{_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON}"
            )
    # Targets RoutedExperts.Config because the composites constrain BOTH the
    # experts and the token dispatcher: their kernels need every token group
    # padded (zero-filled) to the plan's row multiple, which only the padded
    # dispatch path produces.
    if type(cfg) is not RoutedExperts.Config:
        raise ValueError(
            "mxfp8_fused_grouped_mlp targets the stock "
            f"RoutedExperts.Config, got {type(cfg).__qualname__}; narrow this "
            "override's fqns or remove the conflicting override."
        )
    inner = cfg.inner_experts
    if type(inner) is not GroupedExperts.Config:
        raise ValueError(
            "mxfp8_fused_grouped_mlp requires the stock "
            f"GroupedExperts.Config, but inner_experts is "
            f"{type(inner).__qualname__}. The composite quantizes every "
            "grouped GEMM itself -- do not combine it with the MXFP8 "
            "grouped-experts converter."
        )

    if fusion_plan == "grouped_gemm_swiglu":
        # GLOBAL dims only: the config carries sharding placements (e.g. a
        # TP shard on hidden_dim) but not mesh degrees, so the per-rank
        # shard dims cannot be computed here. MXFP8FusedGroupedMLP.forward
        # re-validates the LOCAL dims at first call and raises with the
        # config fix.
        if not is_supported(inner.dim, inner.hidden_dim):
            raise ValueError(
                f"mxfp8_fused_grouped_mlp: is_supported(D={inner.dim}, "
                f"F={inner.hidden_dim}) is False; both dims must be positive "
                f"multiples of {_DIM_ALIGNMENT}."
            )
        # The %256 padding contract is the factory's own work: the fused
        # kernels require per-expert groups padded to 256 (FIX_PAD_SIZE) --
        # 128-multiple-only splits corrupt silently and nondeterministically.
        dispatcher = cfg.token_dispatcher
        if isinstance(dispatcher, TorchAOTokenDispatcher.Config):
            dispatcher.pad_multiple = _ROW_ALIGNMENT
        elif isinstance(dispatcher, AllToAllTokenDispatcher.Config):
            swap_token_dispatcher(cfg, pad_multiple=_ROW_ALIGNMENT)
        else:
            raise ValueError(
                f"mxfp8_fused_grouped_mlp: token_dispatcher is "
                f"{type(dispatcher).__qualname__}; only the TorchAO padded "
                "dispatcher (swapped in from the stock all-to-all) is "
                "validated for the per-expert 256-row contract."
            )
    else:
        swap_token_dispatcher(cfg, pad_multiple=128)

    cfg.inner_experts = derive(
        inner,
        MXFP8FusedGroupedMLP.Config,
        fuse_activation=fuse_activation,
        fusion_plan=fusion_plan,
    )
    return cfg
