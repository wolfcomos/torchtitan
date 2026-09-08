# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import pytest
import torch
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

import torchtitan.components.quantization.mxfp8.converter as converter_mod  # noqa: E402
from torchao.float8.float8_utils import compute_error  # noqa: E402
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
_PAD_MULTIPLES = converter_mod._FUSION_PLAN_PAD_MULTIPLES
_ZERO_TOKEN_EXPERT = 1

_MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)


def _require_fusion_plan(fusion_plan):
    """Skip unless the plan's torchao kernels and their runtime are installed.

    Returns the composites module for the fused plans; it is imported lazily
    because it imports torchao's kernels at module scope.
    """
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


def _routed_rows(pad_multiple, *, tail_value=float("nan")) -> _RoutedRows:
    """Expert-major ``x`` and ``grad_y`` shaped like the padded dispatcher's output.

    Expert ``_ZERO_TOKEN_EXPERT`` receives no tokens and expert 3 only a few, so
    the buffer holds one all-zero group and one mostly-padding group. Routed
    rows are random, padding rows zero (as the dispatcher gathers them from its
    zero sentinel row), and ``pad_multiple`` tail rows past the last group hold
    ``tail_value``.
    """
    # Expert _ZERO_TOKEN_EXPERT receives no tokens; expert 3 only a few.
    routed = [pad_multiple - 3, 0, 2 * pad_multiple - 1, 7]
    padded = [max(-(-n // pad_multiple), 1) * pad_multiple for n in routed]
    active = sum(padded)
    generator = torch.Generator(device="cuda").manual_seed(0)
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
    return compute_error(reference.float(), actual.float()).item()


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
@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_fused_plan_runs_outside_plain_eager(execution_mode, fusion_plan):
    """Compile and non-reentrant checkpointing reproduce eager bit for bit.

    Both re-enter the composite: compile traces through the allow_in_graph
    Functions, checkpointing recomputes the forward during backward. The
    kernels are deterministic, so anything but equality means the data
    movement around the kernels changed.
    """
    _require_fusion_plan(fusion_plan)
    module = _build_mxfp8_experts(fusion_plan, _build_stock_experts())
    rows = _routed_rows(_PAD_MULTIPLES[fusion_plan])
    eager = _run(module, rows)

    if execution_mode == "compile":
        forward = torch.compile(module, fullgraph=True)
    else:

        def forward(x_RD, num_tokens_per_expert_E):
            return checkpoint(
                module, x_RD, num_tokens_per_expert_E, use_reentrant=False
            )

    got = _run(module, rows, forward=forward)

    for name, value in got.items():
        assert torch.equal(value, eager[name]), name
