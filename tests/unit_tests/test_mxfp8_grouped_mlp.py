# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cuDNN-frontend MXFP8 fused grouped-MLP override
(mxfp8_grouped_mlp.py).

The five test groups:

1. Composite numerics: forward+backward vs an independent quantized-unfused
   reference built here from standalone RCEIL casts (``to_mx``), raw
   ``torch._scaled_grouped_mm``, and first-principles eager SwiGLU
   forward/backward -- deliberately NOT the override module's own cast
   helpers. Shapes include a D != F case, zero-token experts (asserting
   exactly-zero param grads), an all-experts-empty R == 0 case (the ops'
   documented early-outs), and a strict inactive tail (A < R) filled with
   deliberate garbage including NaN (a NaN-poisoned-inactive-tail check --
   critical because every e2e gate is force-balanced and never exercises
   ragged routing).
2. Composition: the named fused config through the real ``apply_overrides``
   pipeline. The override is self-contained: stock experts in, fused experts
   plus the factory-installed pad_multiple=256 TorchAO dispatcher out, no
   converter involved -- and fail-loud: converter-quantized experts raise
   instead of falling back. Activation evidence is module/config TYPE only.
3. Autograd/AC: under the real SelectiveAC policy the composite forward runs
   exactly twice (save-from-recompute), gradients match no-AC bitwise, and
   the by-reference parameter save survives optimizer.step into the next
   step.
4. State dict / param layout: the 32-block ``w13 [E, 2F, D]`` hooks
   round-trip the stock ``w1_EFD``/``w3_EFD`` checkpoint layout, and the
   param-init remap initializes the gate/up 32-row blocks with their own
   initializers.
5. Trace shape: one fwd+bwd calls exactly 1x fwd op, 2x mm op (FC2 + FC1
   dgrad), 1x bwd op, 2x wgrad op, counted via ``torchao::`` op names ONLY.

Every per-expert row count in this file is a 256-multiple: the cuDNN FE
kernels hard-code FIX_PAD_SIZE=256 and 128-multiple-only splits corrupt
silently and NONDETERMINISTICALLY (the corruption locus migrates between
identical-input reruns, so no passing run proves such a split safe) -- there
is deliberately no "sub-256 splits still work" fixture.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)):
    pytest.skip("Requires CUDA SM 10.0 (Blackwell)", allow_module_level=True)

# The override module's own availability flag is the single source of truth
# for whether the torchao fused grouped-MLP ops exist AND their cudnn-frontend
# kernels are usable; skipping on it (with its reason) instead of a bare
# try/except keeps a broken environment loud rather than silently skipped.
from torchtitan.overrides.mxfp8_grouped_mlp import (
    _TORCHAO_GROUPED_MLP_OPS_AVAILABLE,
    _TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON,
)

if not _TORCHAO_GROUPED_MLP_OPS_AVAILABLE:
    pytest.skip(
        "torchao fused grouped-MLP ops unavailable: "
        f"{_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON}",
        allow_module_level=True,
    )

from torch.profiler import ProfilerActivity, profile
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.utils import compute_error

from torchtitan.config import apply_overrides, derive
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_mxfp8_grouped_mlp,
    deepseek_v3_16b_mxfp8_p256,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_hybridep,
    deepseek_v3_debugmodel_mxfp8,
    deepseek_v3_debugmodel_mxfp8_grouped_mlp,
)
from torchtitan.overrides.mxfp8_grouped_mlp import (
    MXFP8FusedGroupedExperts,
    _make_w13_init,
    mxfp8_fused_grouped_mlp,
    mxfp8_grouped_experts,
)

# The activation string is part of the frozen override interface.
_OVERRIDE_TARGET = "torchtitan.overrides.mxfp8_grouped_mlp.mxfp8_grouped_experts"

_OP_FWD = "torchao::mxfp8_grouped_gemm_swiglu_fwd"
_OP_MM = "torchao::mxfp8_grouped_gemm"
_OP_BWD = "torchao::mxfp8_grouped_gemm_dswiglu_bwd"
_OP_WGRAD = "torchao::mxfp8_grouped_gemm_wgrad"
_ALL_OPS = (_OP_FWD, _OP_MM, _OP_BWD, _OP_WGRAD)

_BLOCK = 32
_E4M3 = torch.float8_e4m3fn
_RCEIL = ScaleCalculationMode.RCEIL

# ---------------------------------------------------------------------------
# Test-1 tolerances: derived from the measured variability of the unfused
# lane itself, never copied from kernel-level (ao op suite) gates.
# Calibration method (rerunnable in-place after any torch/torchao numerics
# change): run THIS file's reference against (a) an alternative reduction
# order of the SAME unfused math (per-expert fp32 dequant-loop GEMMs
# consuming the reference's own BF16 z/h/dz boundary values) and (b) the
# fp32 eager MLP, at both parametrized shapes. Measured 2026-08-18 on GB200
# (torch 2.14.0a0 nightly, app clocks 2062 MHz):
#
#   output  ref-vs-alt-order (dB)   ref-vs-fp32 (dB)
#   y       inf    / 128.16          23.70 / 23.66
#   dx      112.30 / 118.86          23.68 / 23.68
#   dw13    inf    / inf             23.69 / 23.68
#   dw2     inf    / inf             23.65 / 23.66
#
# Identical RCEIL quantization boundaries make the two unfused lanes
# near-bitwise (their mutual floor is 112.3 dB) — but the COMPOSITE cannot
# reach that floor against any bf16-z reference: the cuDNN GLU/dGLU kernels
# evaluate SwiGLU/dSwiGLU from their in-kernel FP32 accumulators (h is
# quantized from f32 silu(z_f32)*up_f32; dz from f32 dh), while this
# reference — and the real unfused MXFP8 baseline it models — round z and dh
# to BF16 first. That one-boundary difference dominates every
# composite output; measured composite-vs-reference band (same host/session
# as the table above):
#
#   output  debugmodel_tail_zero_expert  asym_d_ne_f_tail
#   y            35.39 dB                    35.32 dB
#   dx           35.64 dB                    35.70 dB
#   dw13         35.68 dB                    35.64 dB
#   dw2          35.32 dB                    35.30 dB
#
# The gate sits 5.3 dB below the measured band floor (35.30) and 6.3 dB
# ABOVE the 23.65-23.70 dB "independent-but-correct lane" level (what fp32
# itself scores against this reference), so it still discriminates "shares
# every quantization boundary except the kernel-internal h/dh rounds" from
# "merely correct" — and a real dataflow/layout/offsets bug lands near 0 dB.
# Kernel-level exactness (60-160 dB) is enforced separately by the ao op
# suite against kernel-native references.
_SQNR_VS_REFERENCE_DB = 30.0
# Secondary tracking gate vs the FP32 eager MLP: measured 23.65-23.70 dB for
# every output at both shapes; 2.6 dB of seed headroom. Catches a blind spot
# shared by both MXFP8 lanes.
_SQNR_VS_FP32_DB = 21.0
# ---------------------------------------------------------------------------

# Fixture shapes: per-expert row counts are 256-multiples (the dispatcher's
# pad_multiple=256 ABI guarantee), zero-token experts are legal anywhere, and
# `tail` allocates inactive rows past offsets[-1] (A < R). D != F in the
# second case so a wrong-axis weight cast cannot cancel.
_CASES = {
    "debugmodel_tail_zero_expert": dict(
        d=256, f=256, sizes=[256, 0, 256, 512, 0, 256], tail=256, seed=0
    ),
    "asym_d_ne_f_tail": dict(d=256, f=512, sizes=[256, 0, 512], tail=256, seed=1),
}


def _blk_view(w13: torch.Tensor):
    """[E, 2F, D] 32-block order -> view [E, F//32, 2, 32, D] with the
    gate/up axis at dim 2."""
    e, two_f, d = w13.shape
    return w13.view(e, two_f // 64, 2, 32, d)


def _to_blk(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Stock [E, F, D] pairs -> 32-block [E, 2F, D]."""
    e, f, d = w1.shape
    return (
        torch.stack([w1, w3], dim=2)
        .view(e, f // 32, 32, 2, d)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, 2 * f, d)
    )


def _zsplit(z: torch.Tensor, f: int):
    """[R, 2F] in 32-block order -> (gate [R, F], up [R, F])."""
    r = z.shape[0]
    v = z.view(r, f // 32, 2, 32)
    return v[:, :, 0, :].reshape(r, f), v[:, :, 1, :].reshape(r, f)


def _zmerge(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(gate [R, F], up [R, F]) -> [R, 2F] in 32-block order."""
    r, f = gate.shape
    out = torch.empty(r, 2 * f, dtype=gate.dtype, device=gate.device)
    v = out.view(r, f // 32, 2, 32)
    v[:, :, 0, :] = gate.view(r, f // 32, 32)
    v[:, :, 1, :] = up.view(r, f // 32, 32)
    return out


def _make_case(*, d, f, sizes, tail, seed=0):
    """Dispatcher-shaped fixture: expert-major x [R, D] with per-expert row
    counts ``sizes`` (256-multiples), offsets = inclusive cumsum, and a
    strict inactive tail of ``tail`` rows. Tail rows carry large deliberate
    garbage plus NaN (a NaN-poisoning attack on the undefined inactive tail):
    producers do not define them, kernels must never let them contaminate
    active rows, and y/dx comparisons mask them because the mm op leaves
    output tail rows unwritten."""
    g = len(sizes)
    a = sum(sizes)
    r = a + tail
    torch.manual_seed(seed)
    offsets = torch.tensor(
        [sum(sizes[: i + 1]) for i in range(g)], device="cuda", dtype=torch.int32
    )
    x = torch.randn(r, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    dy = torch.randn(r, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    w1 = torch.randn(g, f, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    w3 = torch.randn(g, f, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    w13 = _to_blk(w1, w3)
    w2 = torch.randn(g, d, f, device="cuda", dtype=torch.bfloat16) / d**0.5
    if tail:
        x[a:] = 12345.0
        dy[a:] = -6789.0
        x[a : a + tail // 2] = float("nan")
        dy[a : a + tail // 2] = float("nan")
    return dict(x=x, dy=dy, w13=w13, w2=w2, offsets=offsets, sizes=sizes, a=a, r=r)


# ---------------------------------------------------------------------------
# Independent quantized-unfused reference. Standalone torchao RCEIL casts
# (``to_mx``) + raw ``torch._scaled_grouped_mm`` + eager SwiGLU
# forward/backward (the BF16 round of z precedes SwiGLU; the BF16 rounds of
# h/dz precede their quantizers); wgrads are colwise quant-dequant + fp32
# matmul per expert. Built from first principles, sharing no code with the
# module under test. Operates directly on the 32-block ``w13 [G, 2F, D]``:
# rowwise quantization is per-row (row order is irrelevant) and colwise
# 32-blocks along 2F are pure-gate or pure-up in this order, so quantization
# boundaries match the composite exactly.
# ---------------------------------------------------------------------------


def _rceil_rowwise(t):
    scale, q = to_mx(t, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, to_blocked(scale)


def _rceil_rowwise_3d(w):
    qs, sfs = zip(*(_rceil_rowwise(w[g]) for g in range(w.shape[0])))
    return torch.stack(list(qs)), torch.stack(list(sfs))


def _rceil_colwise_3d(w):
    """[G, N, K] -> qdata [G, N, K] stride (N*K, 1, N) quantized along N +
    per-group blocked scales (the ``mat2`` of a dgrad ``_scaled_grouped_mm``)."""
    qs, sfs = zip(*(_rceil_rowwise(w[g].t().contiguous()) for g in range(w.shape[0])))
    return torch.stack(list(qs)).transpose(-2, -1), torch.stack(list(sfs))


def _dequant(q, scale):
    m, k = q.shape
    return (
        q.float().view(m, k // _BLOCK, _BLOCK)
        * scale.to(torch.float32).view(m, k // _BLOCK, 1)
    ).view(m, k)


def _quant_dequant_colwise(t):
    """[m, N] bf16 -> fp32 [N, m]: RCEIL-quantize along the row axis (32x1)
    and dequantize. Per-expert slices quantize identically to the whole
    matrix because 256-multiple group sizes keep every 32-value block inside
    one group."""
    scale, q = to_mx(t.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return _dequant(q, scale)


def _wgrad_expert(a, b):
    """Normative wgrad for one expert: dequant(a_col).T @ dequant(b_col),
    fp32 accumulation, one BF16 round. a [m, N], b [m, K] -> [N, K]."""
    return (_quant_dequant_colwise(a) @ _quant_dequant_colwise(b).t()).to(
        torch.bfloat16
    )


def _reference_forward_backward(x, w13, w2, dy, offsets, sizes, a):
    """Returns (y, dx, dw13 [G, 2F, D] 32-block order, dw2 [G, D, F]). y/dx
    tail rows [a:] are defined as zero here (the real ops leave them
    unwritten; callers mask them out of every comparison)."""
    r, d = x.shape
    g, two_f = w13.shape[0], w13.shape[1]
    f = two_f // 2

    # FC1 forward, then eager SwiGLU on the BF16-rounded z. The gate/up split
    # follows the 32-block column order z inherits from the w13 row order.
    x_q, x_sf = _rceil_rowwise(x)
    w13_row_q, w13_row_sf = _rceil_rowwise_3d(w13)
    z = torch._scaled_grouped_mm(
        x_q,
        w13_row_q.transpose(-2, -1),
        x_sf.reshape(r, -1),
        w13_row_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    z[a:] = 0
    gate_bf16, up_bf16 = _zsplit(z, f)
    gate = gate_bf16.float()
    up = up_bf16.float()
    h = (F.silu(gate) * up).to(torch.bfloat16)

    # FC2 forward.
    h_q, h_sf = _rceil_rowwise(h)
    w2_row_q, w2_row_sf = _rceil_rowwise_3d(w2)
    y = torch._scaled_grouped_mm(
        h_q,
        w2_row_q.transpose(-2, -1),
        h_sf.reshape(r, -1),
        w2_row_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    y[a:] = 0

    # FC2 dgrad, then eager dSwiGLU on the BF16-rounded dh.
    dy_q, dy_sf = _rceil_rowwise(dy)
    w2_col_q, w2_col_sf = _rceil_colwise_3d(w2)
    dh = torch._scaled_grouped_mm(
        dy_q,
        w2_col_q,
        dy_sf.reshape(r, -1),
        w2_col_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    dh[a:] = 0
    sig = torch.sigmoid(gate)
    silu_g = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    dhf = dh.float()
    dgate = (dhf * up * dsilu).to(torch.bfloat16)
    dup = (dhf * silu_g).to(torch.bfloat16)
    dz = _zmerge(dgate, dup)

    # FC1 dgrad.
    dz_q, dz_sf = _rceil_rowwise(dz)
    w13_col_q, w13_col_sf = _rceil_colwise_3d(w13)
    dx = torch._scaled_grouped_mm(
        dz_q,
        w13_col_q,
        dz_sf.reshape(r, -1),
        w13_col_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    dx[a:] = 0

    # Wgrads over active rows only; zero-token experts stay all-zero.
    dw13 = torch.zeros(g, two_f, d, device=x.device, dtype=torch.bfloat16)
    dw2 = torch.zeros(g, d, f, device=x.device, dtype=torch.bfloat16)
    prev = 0
    for gi in range(g):
        end = int(offsets[gi])
        if end > prev:
            dw13[gi] = _wgrad_expert(dz[prev:end], x[prev:end])
            dw2[gi] = _wgrad_expert(dy[prev:end], h[prev:end])
        prev = end
    return y, dx, dw13, dw2


def _fp32_reference(x, w13, w2, dy, offsets, a):
    """FP32 eager autograd MLP over the active rows. Returns
    (y, dx, dw13 [G, 2F, D] 32-block order, dw2)."""
    g, two_f = w13.shape[0], w13.shape[1]
    f = two_f // 2
    x32 = x[:a].float().detach().requires_grad_(True)
    w13_32 = w13.float().detach().requires_grad_(True)
    w2_32 = w2.float().detach().requires_grad_(True)
    v = _blk_view(w13_32)
    outs, prev = [], 0
    for gi in range(g):
        end = int(offsets[gi])
        w1_g = v[gi, :, 0].reshape(f, x.shape[1])
        w3_g = v[gi, :, 1].reshape(f, x.shape[1])
        gate = x32[prev:end] @ w1_g.t()
        up = x32[prev:end] @ w3_g.t()
        h = F.silu(gate) * up
        outs.append(h @ w2_32[gi].t())
        prev = end
    y_ref = torch.cat(outs, dim=0)
    y_ref.backward(dy[:a].float())
    return y_ref, x32.grad, w13_32.grad, w2_32.grad


# ---------------------------------------------------------------------------
# 1. Composite numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_CASES))
def test_composite_matches_quantized_unfused_reference(case):
    fx = _make_case(**_CASES[case])
    a, r, d = fx["a"], fx["r"], fx["x"].shape[1]

    x = fx["x"].clone().detach().requires_grad_(True)
    w13 = fx["w13"].clone().detach().requires_grad_(True)
    w2 = fx["w2"].clone().detach().requires_grad_(True)
    y = mxfp8_fused_grouped_mlp(x, w13, w2, fx["offsets"])
    assert y.shape == (r, d)
    assert y.dtype == torch.bfloat16
    y.backward(fx["dy"])

    ref_y, ref_dx, ref_dw13, ref_dw2 = _reference_forward_backward(
        fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], fx["sizes"], a
    )
    fp32 = _fp32_reference(fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], a)

    # Zero-token experts must produce exactly-zero weight gradients through
    # the autograd path (the wgrad op writes empty-group outputs as zero).
    for gi, m in enumerate(fx["sizes"]):
        if m == 0:
            assert w13.grad[gi].abs().max().item() == 0.0
            assert w2.grad[gi].abs().max().item() == 0.0

    # y/dx are compared over active rows only: both lanes leave the inactive
    # tail [A, R) unwritten, and the garbage+NaN planted in the x/dy tails
    # must not move (or NaN-poison) any active output.
    for name, got, ref, hp in [
        ("y", y[:a], ref_y[:a], fp32[0]),
        ("dx", x.grad[:a], ref_dx[:a], fp32[1]),
        ("dw13", w13.grad, ref_dw13, fp32[2]),
        ("dw2", w2.grad, ref_dw2, fp32[3]),
    ]:
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= _SQNR_VS_REFERENCE_DB, (
            f"{name} SQNR vs quantized-unfused reference {sqnr} < "
            f"{_SQNR_VS_REFERENCE_DB}"
        )
        sqnr_hp = compute_error(hp.float(), got.float())
        assert sqnr_hp >= _SQNR_VS_FP32_DB, (
            f"{name} SQNR vs fp32 {sqnr_hp} < {_SQNR_VS_FP32_DB}"
        )


def test_composite_zero_routed_tokens():
    # A local expert set receiving zero routed tokens (R == 0, every offset
    # 0): the titan cast chain must produce empty quantized operands and the
    # ops' documented R == 0 early-outs must return empty y/dx and
    # exactly-zero weight grads without error.
    d, f, g = 256, 256, 3
    torch.manual_seed(8)
    x = torch.zeros(0, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w13 = (
        torch.randn(g, 2 * f, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    ).requires_grad_(True)
    w2 = (
        torch.randn(g, d, f, device="cuda", dtype=torch.bfloat16) / d**0.5
    ).requires_grad_(True)
    offsets = torch.zeros(g, device="cuda", dtype=torch.int32)

    y = mxfp8_fused_grouped_mlp(x, w13, w2, offsets)
    assert y.shape == (0, d)
    assert y.dtype == torch.bfloat16
    y.backward(torch.zeros_like(y))

    assert x.grad is not None and x.grad.shape == (0, d)
    assert w13.grad is not None and w13.grad.abs().max().item() == 0.0
    assert w2.grad is not None and w2.grad.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 2. Composition (config-time factory gating through the real pipeline)
# ---------------------------------------------------------------------------


def _prepare(config):
    """Mimic the trainer's pre-override step (sharding fill)."""
    config.model_spec.model.update_from_config(config=config)
    return config


def _routed_experts_nodes(config):
    return list(config.traverse(RoutedExperts.Config))


def test_composition_fired_on_named_config():
    config = deepseek_v3_debugmodel_mxfp8_grouped_mlp()
    # The activation string is part of the frozen interface.
    assert _OVERRIDE_TARGET in config.override.imports
    _prepare(config)
    # Self-contained: before the apply, the experts are STOCK (no grouped
    # converter ran on this flavor) and the dispatcher is not yet padded.
    for _fqn, cfg, _parent, _attr in _routed_experts_nodes(config):
        assert type(cfg.inner_experts) is GroupedExperts.Config
        assert not isinstance(cfg.token_dispatcher, TorchAOTokenDispatcher.Config)
    replacements = apply_overrides(config.override, config)
    assert replacements

    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        assert type(cfg.inner_experts) is MXFP8FusedGroupedExperts.Config
        # The factory installed the padded dispatcher itself.
        assert isinstance(cfg.token_dispatcher, TorchAOTokenDispatcher.Config)
        assert cfg.token_dispatcher.pad_multiple == 256

    # Module TYPE is the accepted activation evidence (never log text).
    with torch.device("meta"):
        experts = nodes[0][1].inner_experts.build()
    assert type(experts) is MXFP8FusedGroupedExperts


def test_composition_fires_on_stock_config():
    # No converter anywhere: the override alone opts the model in, dispatcher
    # swap included.
    config = _prepare(deepseek_v3_debugmodel())
    config.override.imports.append(_OVERRIDE_TARGET)
    apply_overrides(config.override, config)

    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        assert type(cfg.inner_experts) is MXFP8FusedGroupedExperts.Config
        assert isinstance(cfg.token_dispatcher, TorchAOTokenDispatcher.Config)
        assert cfg.token_dispatcher.pad_multiple == 256


def test_composition_raises_on_converter_quantized_experts():
    # The composite quantizes every grouped GEMM itself; layering it on the
    # MXFP8 grouped-experts converter's output is a config error that must
    # raise -- never a silent fallback to the converter's unfused path.
    config = _prepare(deepseek_v3_debugmodel_mxfp8())
    config.override.imports.append(_OVERRIDE_TARGET)
    with pytest.raises(ValueError, match="grouped-experts converter"):
        apply_overrides(config.override, config)


def test_composition_raises_on_non_stock_routed_experts_subclass():
    # A RoutedExperts.Config SUBCLASS must raise, not no-op (the decorator no
    # longer carries exact=True, so subclass nodes are claimed and gated).
    @dataclass(kw_only=True)
    class _DerivedRoutedExpertsConfig(RoutedExperts.Config):
        pass

    config = _prepare(deepseek_v3_debugmodel())
    node = _routed_experts_nodes(config)[0][1]
    with pytest.raises(ValueError, match="stock RoutedExperts"):
        mxfp8_grouped_experts(derive(node, _DerivedRoutedExpertsConfig))


def test_composition_raises_on_unsupported_dims():
    config = _prepare(deepseek_v3_debugmodel())
    node = _routed_experts_nodes(config)[0][1]
    node.inner_experts.hidden_dim = 100  # not a 128-multiple
    with pytest.raises(ValueError, match="is_supported"):
        mxfp8_grouped_experts(node)


def test_composition_raises_on_non_alltoall_dispatcher():
    # hybridep's padded dispatcher is not validated for the cuDNN FE 256-row
    # contract; the factory must refuse it rather than swap or accept it.
    # HybridEP's own config gate requires EP>1; satisfy it so the factory
    # refusal (not the dispatcher validation) is what's under test.
    config = deepseek_v3_debugmodel_hybridep()
    config.parallelism.expert_parallel_degree = 2
    _prepare(config)
    node = _routed_experts_nodes(config)[0][1]
    with pytest.raises(ValueError, match="TorchAO padded"):
        mxfp8_grouped_experts(node)


def test_composition_raises_when_ops_unavailable(monkeypatch):
    import torchtitan.overrides.mxfp8_grouped_mlp as override_module

    monkeypatch.setattr(override_module, "_TORCHAO_GROUPED_MLP_OPS_AVAILABLE", False)
    monkeypatch.setattr(
        override_module,
        "_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON",
        "unavailable for the test",
    )
    config = _prepare(deepseek_v3_debugmodel())
    node = _routed_experts_nodes(config)[0][1]
    with pytest.raises(ValueError, match="unavailable for the test"):
        override_module.mxfp8_grouped_experts(node)


def test_16b_ab_arms_share_trainer_knobs_and_dispatcher():
    # The fused 16b flavor is a standalone config (no grouped converter), so
    # its A/B pairing with the p256 baseline is hand-maintained; pin the
    # shared knobs and the post-override dispatcher equivalence.
    base = deepseek_v3_16b_mxfp8_p256()
    fused = deepseek_v3_16b_mxfp8_grouped_mlp()
    assert fused.training.steps == base.training.steps
    assert fused.training.disable_cuda_graphs == base.training.disable_cuda_graphs
    assert (
        fused.parallelism.expert_parallel_degree
        == base.parallelism.expert_parallel_degree
    )
    assert fused.debug.moe_force_load_balance == base.debug.moe_force_load_balance
    assert fused.hf_assets_path == base.hf_assets_path
    assert fused.dataloader.dataset == base.dataloader.dataset

    _prepare(fused)
    apply_overrides(fused.override, fused)
    base_nodes = _routed_experts_nodes(base)
    fused_nodes = _routed_experts_nodes(fused)
    assert len(base_nodes) == len(fused_nodes) > 0
    for (_bf, b, _bp, _ba), (_ff, fz, _fp, _fa) in zip(base_nodes, fused_nodes):
        assert type(fz.token_dispatcher) is type(b.token_dispatcher)
        assert fz.token_dispatcher.pad_multiple == 256
        assert b.token_dispatcher.pad_multiple == 256


# ---------------------------------------------------------------------------
# 3/5. Module-level fixtures (SelectiveAC + trace shape)
# ---------------------------------------------------------------------------

_MOD_D, _MOD_F, _MOD_E = 256, 256, 4
_MOD_SIZES = [256, 512, 256, 256]


def _build_module(seed):
    torch.manual_seed(seed)
    module = MXFP8FusedGroupedExperts.Config(
        dim=_MOD_D, hidden_dim=_MOD_F, num_experts=_MOD_E
    ).build()
    module = module.to("cuda")
    with torch.no_grad():
        # fp32 master weights: the .bfloat16() casts stay outside the Function
        # so autograd routes bf16 grads back to fp32 params.
        module.w13.normal_(0.0, _MOD_D**-0.5)
        module.w2_EDF.normal_(0.0, _MOD_D**-0.5)
    return module


def _module_inputs(seed=0):
    torch.manual_seed(seed)
    r = sum(_MOD_SIZES)
    x = torch.randn(r, _MOD_D, device="cuda", dtype=torch.bfloat16) / _MOD_D**0.5
    dy = torch.randn(r, _MOD_D, device="cuda", dtype=torch.bfloat16) / _MOD_D**0.5
    num_tokens = torch.tensor(_MOD_SIZES, device="cuda")
    return x, dy, num_tokens


def _run_module(module, x, dy, num_tokens):
    x = x.clone().detach().requires_grad_(True)
    y = module(x, num_tokens)
    y.backward(dy)
    return y.detach(), x.grad


def _torchao_op_counts(prof):
    # Count ONLY the torchao:: custom-op events: aten CPU events double-count
    # under SAC recompute, so they are never used for launch evidence.
    counts = {}
    for evt in prof.key_averages():
        if evt.key in _ALL_OPS:
            counts[evt.key] = counts.get(evt.key, 0) + evt.count
    return counts


def test_selective_ac_recompute_count_and_bitwise_grads():
    x, dy, num_tokens = _module_inputs()

    ref = _build_module(seed=1)
    y_ref, dx_ref = _run_module(ref, x, dy, num_tokens)

    acm = _build_module(seed=2)
    with torch.no_grad():
        acm.w13.copy_(ref.w13)
        acm.w2_EDF.copy_(ref.w2_EDF)
    wrapped = SelectiveAC(SelectiveAC.Config())._wrap_block(acm)

    # Warm up kernel JIT outside the profiled region.
    _run_module(wrapped, x, dy, num_tokens)
    acm.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        y_ac, dx_ac = _run_module(wrapped, x, dy, num_tokens)
    counts = _torchao_op_counts(prof)

    # The composite forward runs exactly twice under SelectiveAC (original +
    # recompute; saves come from the recompute pass), backward once. The mm
    # op runs 2x in the two forwards (FC2) + 1x in backward (FC1 dgrad).
    assert counts.get(_OP_FWD, 0) == 2, counts
    assert counts.get(_OP_MM, 0) == 3, counts
    assert counts.get(_OP_BWD, 0) == 1, counts
    assert counts.get(_OP_WGRAD, 0) == 2, counts

    # Deterministic kernels + save-from-recompute => bitwise-identical results.
    assert torch.equal(y_ac, y_ref)
    assert torch.equal(dx_ac, dx_ref)
    assert torch.equal(acm.w13.grad, ref.w13.grad)
    assert torch.equal(acm.w2_EDF.grad, ref.w2_EDF.grad)


def test_param_ref_save_survives_optimizer_step():
    # The Function saves w13/w2 by reference; the same-step backward precedes
    # the optimizer update, so step -> next fwd+bwd must work.
    x, dy, num_tokens = _module_inputs()
    module = _build_module(seed=3)
    wrapped = SelectiveAC(SelectiveAC.Config())._wrap_block(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

    _run_module(wrapped, x, dy, num_tokens)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    _, dx = _run_module(wrapped, x, dy, num_tokens)
    assert dx is not None
    assert torch.isfinite(dx).all()
    assert module.w13.grad is not None
    assert torch.isfinite(module.w13.grad).all()


# ---------------------------------------------------------------------------
# 4. State dict / param layout (32-block checkpoint hooks + init remap)
# ---------------------------------------------------------------------------


def test_state_dict_round_trips_stock_layout():
    src = _build_module(seed=4)
    sd = src.state_dict()

    # Saved in the stock GroupedExperts layout, not as the fused parameter.
    assert set(sd) == {"w1_EFD", "w3_EFD", "w2_EDF"}
    v = _blk_view(src.w13)
    f = _MOD_F
    assert torch.equal(sd["w1_EFD"], v[:, :, 0].reshape(_MOD_E, f, _MOD_D))
    assert torch.equal(sd["w3_EFD"], v[:, :, 1].reshape(_MOD_E, f, _MOD_D))

    dst = _build_module(seed=5)
    dst.load_state_dict(sd)
    assert torch.equal(dst.w13, src.w13)
    assert torch.equal(dst.w2_EDF, src.w2_EDF)


def test_state_dict_loads_stock_grouped_experts_checkpoint():
    stock = GroupedExperts.Config(
        dim=_MOD_D, hidden_dim=_MOD_F, num_experts=_MOD_E
    ).build()
    with torch.no_grad():
        for param in stock.parameters():
            param.normal_()

    fused = _build_module(seed=6).cpu()
    fused.load_state_dict(stock.state_dict())
    assert torch.equal(fused.w13, _to_blk(stock.w1_EFD, stock.w3_EFD))
    assert torch.equal(fused.w2_EDF, stock.w2_EDF)


def test_remap_round_trip_identity():
    # elem -> 32-block -> elem is the identity, and the 32-block rows are the
    # documented [gate_0..31 | up_0..31 | ...] pattern.
    e, f, d = 2, 64, 8
    w1 = torch.arange(e * f * d, dtype=torch.float32).reshape(e, f, d)
    w3 = -torch.arange(e * f * d, dtype=torch.float32).reshape(e, f, d)
    blk = _to_blk(w1, w3)
    assert torch.equal(blk[:, 0:32], w1[:, 0:32])  # first 32 gate rows
    assert torch.equal(blk[:, 32:64], w3[:, 0:32])  # then their up rows
    assert torch.equal(blk[:, 64:96], w1[:, 32:64])
    v = _blk_view(blk)
    assert torch.equal(v[:, :, 0].reshape(e, f, d), w1)
    assert torch.equal(v[:, :, 1].reshape(e, f, d), w3)


def test_param_init_remap_initializes_gate_and_up_blocks():
    t = torch.empty(2, 2 * 64, 8)
    init = _make_w13_init(
        lambda w: torch.nn.init.constant_(w, 1.0),
        lambda w: torch.nn.init.constant_(w, 2.0),
    )
    init(t)
    v = _blk_view(t)
    assert (v[:, :, 0] == 1.0).all()  # gate blocks
    assert (v[:, :, 1] == 2.0).all()  # up blocks
    # Alternating 32-row pattern in the flat layout.
    assert (t[:, 0:32] == 1.0).all()
    assert (t[:, 32:64] == 2.0).all()
    assert (t[:, 64:96] == 1.0).all()


# ---------------------------------------------------------------------------
# 5. Trace shape
# ---------------------------------------------------------------------------


def test_trace_counts_one_fwd_two_mm_one_bwd_two_wgrad():
    x, dy, num_tokens = _module_inputs()
    module = _build_module(seed=7)

    # Warm up kernel JIT outside the profiled region.
    _run_module(module, x, dy, num_tokens)
    module.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run_module(module, x, dy, num_tokens)
    counts = _torchao_op_counts(prof)

    assert counts.get(_OP_FWD, 0) == 1, counts
    assert counts.get(_OP_MM, 0) == 2, counts
    assert counts.get(_OP_BWD, 0) == 1, counts
    assert counts.get(_OP_WGRAD, 0) == 2, counts
