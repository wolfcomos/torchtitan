# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for the MXFP8 grouped-experts ``fusion_plan`` composites.

The fixtures reproduce the padded token dispatcher's output contract rather
than running the dispatcher: rows arrive expert-major, every expert's group is
zero-padded to the plan's ``pad_multiple``, an expert that received no tokens
still owns exactly ``pad_multiple`` zero rows, and the buffer carries a tail
past ``offsets[-1]`` that no expert owns. The tail is NaN-poisoned so a kernel
that reads it shows up; outputs and input gradients are compared on the active
rows only, since the composites (like ``torch._grouped_mm``) leave the tail
unwritten.
"""

from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

import torchtitan.components.quantization.mxfp8.converter as converter_mod  # noqa: E402
from torchtitan.components.quantization.mxfp8.converter import (  # noqa: E402
    _get_mxfp8_grouped_experts_cls,
)
from torchtitan.models.common.moe import GroupedExperts  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]

_DIM = 256
# Distinct from _DIM so a weight quantized along the wrong axis cannot cancel.
_HIDDEN_DIM = 512
# A power of two: torchao's per-group scale swizzle
# (triton_mx_block_rearrange_2d_K_groups) sizes a tl.arange by the group count,
# so the per-GEMM path these tests compare against needs one.
_NUM_EXPERTS = 4
_FUSED_PLANS = ("swiglu", "grouped_gemm_swiglu")
_PLANS = ("none", *_FUSED_PLANS)
# Plan none has no row contract of its own; 128 is what the sm_100 CuTeDSL
# quantization kernels of the per-GEMM path want.
_PAD_MULTIPLES = {"none": 128, **converter_mod._FUSION_PLAN_PAD_MULTIPLES}
_ZERO_TOKEN_EXPERT = 1

_MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)


def _require_fusion_plan(fusion_plan):
    """Skip unless the plan's torchao kernels and their runtime are installed.

    Returns the composites module for the fused plans; it is imported lazily
    because it imports torchao's kernels at module scope.
    """
    if fusion_plan == "none":
        return None
    if fusion_plan == "swiglu":
        pytest.importorskip(
            "torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8"
        )
    else:
        pytest.importorskip(
            "torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp"
        )
        cudnn = pytest.importorskip("cudnn")
        if not hasattr(cudnn, "grouped_gemm_glu_wrapper_sm100"):
            pytest.skip("cudnn-frontend lacks the grouped_gemm_*_wrapper_sm100 kernels")
        if torch.cuda.get_device_capability() != (10, 0):
            pytest.skip("the cuDNN grouped-MLP kernels are built for SM 10.0 only")
    return pytest.importorskip(
        "torchtitan.components.quantization.mxfp8.grouped_experts"
    )


class _RoutedRows(NamedTuple):
    x: torch.Tensor
    grad_y: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    active: int
    """Rows owned by an expert, ``offsets[-1]``; the rest is the tail."""


def _routed_rows(pad_multiple, *, tail_value=float("nan"), seed=0) -> _RoutedRows:
    """Expert-major ``x`` and ``grad_y`` shaped like the padded dispatcher's output.

    Expert ``_ZERO_TOKEN_EXPERT`` receives no tokens and expert 3 only a few, so
    the buffer holds one all-zero group and one mostly-padding group. Routed
    rows are random, padding rows zero (as the dispatcher gathers them from its
    zero sentinel row), and ``pad_multiple`` tail rows past the last group hold
    ``tail_value``.
    """
    routed = [pad_multiple - 3, 0, 2 * pad_multiple - 1, 7]
    assert routed[_ZERO_TOKEN_EXPERT] == 0 and len(routed) == _NUM_EXPERTS
    padded = [max(-(-n // pad_multiple), 1) * pad_multiple for n in routed]
    active = sum(padded)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x_RD = torch.zeros(active + pad_multiple, _DIM, device="cuda", dtype=torch.bfloat16)
    grad_y_RD = torch.zeros_like(x_RD)
    start = 0
    for n_routed, n_padded in zip(routed, padded):
        for buffer in (x_RD, grad_y_RD):
            buffer[start : start + n_routed] = torch.randn(
                n_routed,
                _DIM,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
        start += n_padded
    x_RD[active:] = tail_value
    grad_y_RD[active:] = tail_value
    return _RoutedRows(x_RD, grad_y_RD, torch.tensor(padded, device="cuda"), active)


def _build_stock_experts(seed=0) -> GroupedExperts:
    module = (
        GroupedExperts.Config(
            dim=_DIM, hidden_dim=_HIDDEN_DIM, num_experts=_NUM_EXPERTS
        )
        .build()
        .cuda()
        .bfloat16()
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0.0, _DIM**-0.5, generator=generator)
    return module


def _build_mxfp8_experts(fusion_plan, stock: GroupedExperts):
    """The dynamic MXFP8 class with the plan, carrying the stock module's weights."""
    module = (
        _MXFP8GroupedExperts.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            fusion_plan=fusion_plan,
        )
        .build()
        .cuda()
        .bfloat16()
    )
    module.load_state_dict(stock.state_dict())
    return module


def _run(module, rows: _RoutedRows, *, forward=None):
    """One forward/backward through ``forward`` (the module by default).

    Returns the active rows of ``y`` and ``x.grad`` and the full weight
    gradients, keyed by name.
    """
    # Not ``forward or module``: truth-testing a compiled module calls its
    # ``__len__``, which raises.
    forward = module if forward is None else forward
    x_RD = rows.x.detach().clone().requires_grad_()
    module.zero_grad(set_to_none=True)
    y_RD = forward(x_RD, rows.num_tokens_per_expert)
    y_RD.backward(rows.grad_y)
    return {
        "y": y_RD.detach()[: rows.active],
        "x.grad": x_RD.grad[: rows.active],
        "w1_EFD.grad": module.w1_EFD.grad.clone(),
        "w2_EDF.grad": module.w2_EDF.grad.clone(),
        "w3_EFD.grad": module.w3_EFD.grad.clone(),
    }


def _sqnr_db(reference, actual):
    reference, actual = reference.float(), actual.float()
    return (20 * torch.log10(reference.norm() / (reference - actual).norm())).item()


def test_plan_none_delegates_to_the_stock_mlp_bitwise():
    """The ``_grouped_mlp`` override for plan none is the pre-refactor path.

    Before the seam existed the MXFP8 subclass only overrode ``_grouped_mm``
    and inherited the stock MLP; calling that stock MLP on the module directly
    must give the same bits, forward and backward.
    """
    module = _build_mxfp8_experts("none", _build_stock_experts())
    rows = _routed_rows(_PAD_MULTIPLES["none"])

    def stock_mlp(x_RD, num_tokens_per_expert_E):
        return GroupedExperts._grouped_mlp(
            module,
            x_RD=x_RD,
            w1_EFD=module.w1_EFD,
            w2_EDF=module.w2_EDF,
            w3_EFD=module.w3_EFD,
            offsets_E=torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32),
        ).type_as(x_RD)

    got = _run(module, rows)
    expected = _run(module, rows, forward=stock_mlp)

    for name, value in got.items():
        assert torch.equal(value, expected[name]), name


# The standalone-cast arm of the swiglu A/B: the SwiGLU boundary in BF16
# eager math (ported from the pre-fusion composite) followed by torchao's
# standalone 1x32 and 32x1 cast kernels. Everything else in the composite is
# shared, so the comparison isolates the fused SwiGLU + cast kernel.


def _swiglu_forward_hp(gated):
    k = gated.shape[1] // 2
    return (F.silu(gated[:, :k].float()) * gated[:, k:].float()).to(gated.dtype)


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


def test_swiglu_plan_matches_the_standalone_cast_arm(monkeypatch):
    """A/B of the fused SwiGLU + cast kernel against standalone casts.

    Only the SwiGLU boundary differs between the arms (SwiGLU and both casts
    in one kernel vs. BF16 eager SwiGLU followed by torchao's standalone 1x32
    and 32x1 cast kernels), so the comparison isolates the fused kernel.

    Measured on GB200 (torch 2.14, ao#4743 kernels) over three seeds of this
    fixture (640 active rows, dim 256, hidden 512, 4 experts): ``y`` bitwise
    equal on every seed; ``x.grad`` and the weight gradients bitwise equal on
    two seeds and 89.6-92.2 dB on the third (22 of 163840 ``x.grad`` and 43 of
    524288 ``w1`` elements differ, max |diff| 0.0078 and 0.0625). Forward is
    pinned bitwise; backward gates at 40 dB.
    """
    grouped_experts = _require_fusion_plan("swiglu")
    from torchao.prototype.moe_training.kernels.mxfp8.quant import (
        mxfp8_quantize_2d_1x32_cutedsl,
        mxfp8_quantize_2d_32x1_cutedsl,
    )

    scaling_mode = grouped_experts._SCALE_MODE.value

    def standalone_forward_casts(gated):
        h = _swiglu_forward_hp(gated)
        h_rw, hs_rw = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode=scaling_mode)
        h_cw, hs_cw = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode=scaling_mode)
        return h_rw, h_cw, hs_rw, hs_cw

    def standalone_backward_casts(grad_h, gated):
        d = _swiglu_backward_hp(grad_h, gated)
        d_rw, ds_rw = mxfp8_quantize_2d_1x32_cutedsl(d, scaling_mode=scaling_mode)
        d_cw, ds_cw = mxfp8_quantize_2d_32x1_cutedsl(d, scaling_mode=scaling_mode)
        return d_rw, d_cw, ds_rw, ds_cw

    module = _build_mxfp8_experts("swiglu", _build_stock_experts())
    rows = _routed_rows(_PAD_MULTIPLES["swiglu"])
    fused = _run(module, rows)
    monkeypatch.setattr(
        grouped_experts, "_swiglu_forward_casts", standalone_forward_casts
    )
    monkeypatch.setattr(
        grouped_experts, "_swiglu_backward_casts", standalone_backward_casts
    )
    standalone = _run(module, rows)

    assert torch.equal(fused["y"], standalone["y"])
    for name in ("x.grad", "w1_EFD.grad", "w2_EDF.grad", "w3_EFD.grad"):
        assert _sqnr_db(standalone[name], fused[name]) >= 40.0, name


@pytest.mark.parametrize("fusion_plan", _FUSED_PLANS)
def test_fused_plan_tracks_the_unfused_experts(fusion_plan):
    """Fused plans stay within the quantization band of the per-GEMM path.

    Measured on GB200 (torch 2.14) over three seeds, across ``y``, ``x.grad``
    and the three weight gradients: swiglu 39.1-40.9 dB vs the per-GEMM MXFP8
    path and 23.1-24.2 dB vs BF16; grouped_gemm_swiglu 33.1-36.9 dB and
    23.1-24.2 dB (the cuDNN kernels apply SwiGLU to their FP32 accumulators
    where the other paths round to BF16 first). The per-GEMM path itself
    scores 23.1-24.2 dB vs BF16, so the BF16 gate tracks the MXFP8 noise
    floor; a dataflow or layout bug lands near 0 dB on both.
    """
    _require_fusion_plan(fusion_plan)
    stock = _build_stock_experts()
    rows = _routed_rows(_PAD_MULTIPLES[fusion_plan])

    fused = _run(_build_mxfp8_experts(fusion_plan, stock), rows)
    unfused = _run(_build_mxfp8_experts("none", stock), rows)
    bf16 = _run(stock, rows)

    for name, value in fused.items():
        assert _sqnr_db(unfused[name], value) >= 30.0, name
        assert _sqnr_db(bf16[name], value) >= 21.0, name


@pytest.mark.parametrize("fusion_plan", _FUSED_PLANS)
def test_fused_plan_reads_only_the_active_rows(fusion_plan):
    """Rows no expert owns never reach an active output or a gradient.

    The composites cast whole buffers (rowwise and columnwise) and slice the
    per-group scales afterwards, so a tail that leaked would show up as NaN in
    an active row or a changed weight gradient. An expert that received no
    tokens owns only zero rows and gets exactly zero weight gradients.
    """
    _require_fusion_plan(fusion_plan)
    module = _build_mxfp8_experts(fusion_plan, _build_stock_experts())
    pad_multiple = _PAD_MULTIPLES[fusion_plan]

    poisoned = _run(module, _routed_rows(pad_multiple))
    clean = _run(module, _routed_rows(pad_multiple, tail_value=0.0))

    for name, value in poisoned.items():
        assert torch.isfinite(value).all(), name
        assert torch.equal(value, clean[name]), name
    for name in ("w1_EFD.grad", "w2_EDF.grad", "w3_EFD.grad"):
        assert torch.count_nonzero(poisoned[name][_ZERO_TOKEN_EXPERT]) == 0, name


@pytest.mark.parametrize(
    ("fusion_plan", "rows", "match"),
    [("swiglu", 64, "at least 128"), ("grouped_gemm_swiglu", 128, "at least 256")],
)
def test_fused_plan_rejects_a_short_token_buffer(fusion_plan, rows, match):
    """A buffer below the plan's row multiple raises before any kernel runs."""
    _require_fusion_plan(fusion_plan)
    module = _build_mxfp8_experts(fusion_plan, _build_stock_experts())
    x_RD = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
    num_tokens_per_expert_E = torch.zeros(
        _NUM_EXPERTS, device="cuda", dtype=torch.int64
    )
    num_tokens_per_expert_E[0] = rows

    with pytest.raises(ValueError, match=match):
        module(x_RD, num_tokens_per_expert_E)


@pytest.mark.parametrize("fusion_plan", _FUSED_PLANS)
def test_fused_plan_config_rejects_unaligned_dims(fusion_plan):
    with pytest.raises(ValueError, match="divisible by 128"):
        _MXFP8GroupedExperts.Config(
            dim=_DIM, hidden_dim=96, num_experts=_NUM_EXPERTS, fusion_plan=fusion_plan
        )


@pytest.mark.parametrize("fusion_plan", _PLANS)
@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_fusion_plan_runs_outside_plain_eager(execution_mode, fusion_plan):
    """Compile and non-reentrant checkpointing reproduce eager.

    Both re-enter the composite: compile traces through the allow_in_graph
    Functions, checkpointing recomputes the forward during backward. The
    kernels are deterministic, so the fused plans match eager bit for bit
    (measured so on GB200 with torch 2.14, with and without ``fullgraph``,
    stable across compiled runs); anything else means the data movement
    around the kernels changed. Plan none under compile is the exception:
    inductor fuses the per-GEMM path's BF16 ``F.silu(h1) * h3`` in FP32 and
    drops the BF16 rounding of ``silu(h1)`` (measured 39.1-40.9 dB vs eager
    over two seeds, and bitwise equal to eager with an FP32 SwiGLU), so that
    case gates on SQNR.
    """
    _require_fusion_plan(fusion_plan)
    module = _build_mxfp8_experts(fusion_plan, _build_stock_experts())
    rows = _routed_rows(_PAD_MULTIPLES[fusion_plan])
    eager = _run(module, rows)

    if execution_mode == "compile":
        torch._dynamo.reset()
        forward = torch.compile(module)
    else:

        def forward(x_RD, num_tokens_per_expert_E):
            return checkpoint(
                module, x_RD, num_tokens_per_expert_E, use_reentrant=False
            )

    got = _run(module, rows, forward=forward)

    bitwise = not (execution_mode == "compile" and fusion_plan == "none")
    for name, value in got.items():
        if bitwise:
            assert torch.equal(value, eager[name]), name
        else:
            assert _sqnr_db(eager[name], value) >= 30.0, name


def test_grouped_gemm_swiglu_contract_matches_the_torchao_ops():
    """The converter's row and dim contract for the cuDNN plan is torchao's."""
    cudnn_grouped_mlp = pytest.importorskip(
        "torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp"
    )

    assert (
        converter_mod._FUSION_PLAN_PAD_MULTIPLES["grouped_gemm_swiglu"]
        == cudnn_grouped_mlp.ROW_GROUP_ALIGNMENT
    )
    assert converter_mod._FUSED_DIM_ALIGNMENT == cudnn_grouped_mlp.DIM_ALIGNMENT
