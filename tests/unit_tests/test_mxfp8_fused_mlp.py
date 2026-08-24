# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the self-contained MXFP8 fused-MLP overrides.

The wiring tests are config-tree transforms plus meta/CPU builds with the
composites mocked, so they run without a GPU: the factories' SM100 gate is
patched out and the grouped_gemm_swiglu plan's torchao-ops availability is
mocked. Numerics of the grouped_gemm_swiglu composite run in the class gated
on real SM100 hardware plus the torchao fused grouped-MLP ops (skipped
otherwise, e.g. in GPU-less stock-torchao CI); the swiglu composites' numerics
are validated on SM100 hardware in NVIDIA-internal CI.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from torchtitan.components.quantization import MXFP8GroupedExpertsConverter
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.llama3 import model_registry as llama3_model_registry

try:
    from torchtitan.overrides.mxfp8_fused_mlp import (
        _pack_w13_blocks,
        _TORCHAO_GROUPED_MLP_OPS_AVAILABLE,
        _TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON,
        mxfp8_fused_grouped_mlp,
        mxfp8_fused_mlp,
        MXFP8FusedGroupedMLP,
        MXFP8FusedMLP,
    )
    from torchtitan.overrides.mxfp8_fused_mlp import (
        _MXFP8GroupedGemmMLP,
    )
except ImportError as e:  # torchao (or a transitive dep) not installed
    raise unittest.SkipTest(
        f"torchao is required for the MXFP8 fused-MLP overrides: {e}"
    ) from e

# Reaching here means the override module (and thus torchao) imported; these
# torchao symbols exist in every torchao the override module accepts and are
# only exercised by the SM100-gated numerics class.
from torchao.prototype.mx_formats.config import ScaleCalculationMode  # noqa: E402
from torchao.prototype.mx_formats.mx_tensor import to_mx  # noqa: E402
from torchao.prototype.mx_formats.utils import to_blocked  # noqa: E402
from torchao.quantization.utils import compute_error  # noqa: E402

_DENSE_OVERRIDE = "torchtitan.overrides.mxfp8_fused_mlp.mxfp8_fused_mlp"
_GROUPED_OVERRIDE = "torchtitan.overrides.mxfp8_fused_mlp.mxfp8_fused_grouped_mlp"
_GROUPED_PLAN_IMPORT = (_GROUPED_OVERRIDE, {"fusion_plan": "grouped_gemm_swiglu"})

_HAS_SM100_GPU = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
)


class TestMXFP8FusedMLPOverride(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp.has_cuda_capability",
            lambda *args: True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dense_override_builds_mxfp8_fused_mlp(self):
        model_config = llama3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_DENSE_OVERRIDE]), model_config)
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedMLP)]
        self.assertTrue(fused)
        self.assertTrue(all(m.fuse_activation for m in fused))

    def test_dense_override_kwargs_configure_the_composite(self):
        model_config = llama3_model_registry("debugmodel").model
        apply_overrides(
            OverrideConfig(imports=[(_DENSE_OVERRIDE, {"fuse_activation": False})]),
            model_config,
        )
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedMLP)]
        self.assertTrue(fused)
        self.assertFalse(any(m.fuse_activation for m in fused))

    def _grouped_model_config(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_GROUPED_OVERRIDE]), model_config)
        return model_config

    def _grouped_experts_config(self, model_config):
        nodes = list(model_config.traverse(MXFP8FusedGroupedMLP.Config))
        self.assertTrue(nodes)
        return nodes[0][1]

    def test_grouped_override_builds_experts_and_padded_dispatcher(self):
        model_config = self._grouped_model_config()
        pads = [
            dispatcher_cfg.pad_multiple
            for _fqn, dispatcher_cfg, _parent, _attr in model_config.traverse(
                TorchAOTokenDispatcher.Config
            )
        ]
        self.assertTrue(pads)
        self.assertTrue(all(pad == 128 for pad in pads))
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedGroupedMLP)]
        self.assertTrue(fused)
        self.assertTrue(all(type(m) is MXFP8FusedGroupedMLP for m in fused))
        self.assertTrue(all(m.fuse_activation for m in fused))

    def test_grouped_forward_validates_and_applies_the_function(self):
        cfg = self._grouped_experts_config(self._grouped_model_config())
        module = cfg.build()
        num_tokens = torch.zeros(cfg.num_experts, dtype=torch.int64)
        num_tokens[0] = 2
        x = torch.randn(2, cfg.dim)
        sentinel = torch.zeros(2, cfg.dim, dtype=torch.bfloat16)
        with mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._validate_grouped_inputs"
        ) as validate, mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._MXFP8GroupedMLP.apply",
            return_value=sentinel,
        ) as function:
            out = module(x, num_tokens)
        function.assert_called_once()
        args = function.call_args.args
        self.assertEqual(args[0].dtype, torch.bfloat16)
        self.assertEqual(
            tuple(args[1].shape), (cfg.num_experts, cfg.hidden_dim, 2, cfg.dim)
        )
        self.assertEqual(
            tuple(args[2].shape), (cfg.num_experts, cfg.hidden_dim, cfg.dim)
        )
        self.assertEqual(args[3].dtype, torch.int32)
        self.assertEqual(args[3].tolist(), torch.cumsum(num_tokens, dim=0).tolist())
        self.assertEqual(args[4], True)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3])
        self.assertEqual(out.dtype, x.dtype)

    def test_grouped_checkpoint_keys_unchanged(self):
        stock_nodes = list(
            deepseek_v3_model_registry("debugmodel").model.traverse(
                GroupedExperts.Config
            )
        )
        self.assertTrue(stock_nodes)
        fused_cfg = self._grouped_experts_config(self._grouped_model_config())
        with torch.device("meta"):
            stock = stock_nodes[0][1].build()
            fused = fused_cfg.build()
        self.assertEqual(set(fused.state_dict().keys()), set(stock.state_dict().keys()))

    def test_fresh_init_matches_stock_bitwise(self):
        # Stock parameters in stock registration order: fresh-init draws must
        # be bitwise-identical to the corresponding stock module's.
        def seeded_state_dict(cfg):
            torch.manual_seed(42)
            module = cfg.build()
            module.init_states()
            return module.state_dict()

        dense_stock = list(
            llama3_model_registry("debugmodel").model.traverse(FeedForward.Config)
        )[0][1]
        dense_model = llama3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_DENSE_OVERRIDE]), dense_model)
        dense_fused = list(dense_model.traverse(MXFP8FusedMLP.Config))[0][1]
        grouped_stock = list(
            deepseek_v3_model_registry("debugmodel").model.traverse(
                GroupedExperts.Config
            )
        )[0][1]
        grouped_fused = self._grouped_experts_config(self._grouped_model_config())
        for stock_cfg, fused_cfg in (
            (dense_stock, dense_fused),
            (grouped_stock, grouped_fused),
        ):
            stock_sd = seeded_state_dict(stock_cfg)
            fused_sd = seeded_state_dict(fused_cfg)
            self.assertEqual(set(fused_sd), set(stock_sd))
            for key, stock_tensor in stock_sd.items():
                self.assertTrue(torch.equal(fused_sd[key], stock_tensor), key)

    def test_dense_factory_raises_on_non_stock_ffn(self):
        # A FeedForward.Config SUBCLASS (already overridden) must raise, not
        # no-op.
        gate = Linear.Config(in_features=128, out_features=256)
        cfg = MXFP8FusedMLP.Config(
            w1=gate,
            w2=Linear.Config(in_features=256, out_features=128),
            w3=gate,
        )
        with self.assertRaisesRegex(ValueError, "stock FeedForward.Config"):
            mxfp8_fused_mlp(cfg)

    def test_dense_factory_raises_on_converted_projection(self):
        from torchtitan.components.quantization.mx import MXFP8Linear

        if MXFP8Linear is None:
            self.skipTest("torchao not installed")
        gate = Linear.Config(in_features=128, out_features=256)
        cfg = FeedForward.Config(
            w1=gate,
            w2=MXFP8Linear.Config(in_features=256, out_features=128),
            w3=gate,
        )
        with self.assertRaisesRegex(ValueError, "quantization converter"):
            mxfp8_fused_mlp(cfg)


class TestGroupedGemmPlanWiring(unittest.TestCase):
    """GPU-less wiring tests for the ``grouped_gemm_swiglu`` fusion plan:
    config routing, the pad-256 dispatcher policy, the forward-time 32-block
    pack, and the factory's fail-loud raises. The SM100 gate and the torchao
    fused grouped-MLP ops' availability are mocked."""

    def setUp(self):
        for patcher in (
            mock.patch(
                "torchtitan.overrides.mxfp8_fused_mlp.has_cuda_capability",
                lambda *args: True,
            ),
            mock.patch(
                "torchtitan.overrides.mxfp8_fused_mlp."
                "_TORCHAO_GROUPED_MLP_OPS_AVAILABLE",
                True,
            ),
            mock.patch(
                "torchtitan.overrides.mxfp8_fused_mlp.is_supported",
                lambda dim, hidden_dim: dim % 128 == 0 and hidden_dim % 128 == 0,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _plan_model_config(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_GROUPED_PLAN_IMPORT]), model_config)
        return model_config

    def _plan_experts_config(self, model_config):
        nodes = list(model_config.traverse(MXFP8FusedGroupedMLP.Config))
        self.assertTrue(nodes)
        return nodes[0][1]

    def _stock_routed_experts_config(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        nodes = list(model_config.traverse(RoutedExperts.Config))
        self.assertTrue(nodes)
        return nodes[0][1]

    def test_pack_w13_blocks_matches_kernel_32_block_mapping(self):
        # Byte-verify the forward-time pack against an independent
        # re-implementation of the kernels' 32-block GLU row order: block i
        # holds gate (w1) rows [32i, 32i+32), then the SAME features' up (w3)
        # rows.
        e, f, d = 2, 96, 8
        w1 = torch.arange(e * f * d, dtype=torch.float32).reshape(e, f, d)
        w3 = -1.0 - torch.arange(e * f * d, dtype=torch.float32).reshape(e, f, d)
        packed = _pack_w13_blocks(w1, w3)
        self.assertEqual(tuple(packed.shape), (e, 2 * f, d))
        expected = torch.empty(e, 2 * f, d)
        for i in range(f // 32):
            expected[:, 64 * i : 64 * i + 32] = w1[:, 32 * i : 32 * i + 32]
            expected[:, 64 * i + 32 : 64 * i + 64] = w3[:, 32 * i : 32 * i + 32]
        self.assertTrue(torch.equal(packed, expected))
        # Unpack round-trip through the (E, F//32, 2, 32, D) view is the
        # identity.
        v = packed.view(e, f // 32, 2, 32, d)
        self.assertTrue(torch.equal(v[:, :, 0].reshape(e, f, d), w1))
        self.assertTrue(torch.equal(v[:, :, 1].reshape(e, f, d), w3))

    def test_plan_routes_to_the_grouped_gemm_function(self):
        cfg = self._plan_experts_config(self._plan_model_config())
        self.assertEqual(cfg.fusion_plan, "grouped_gemm_swiglu")
        module = cfg.build()
        torch.manual_seed(0)
        with torch.no_grad():
            for param in module.parameters():
                param.normal_()
        num_tokens = torch.zeros(cfg.num_experts, dtype=torch.int64)
        num_tokens[0] = 2
        x = torch.randn(2, cfg.dim)
        sentinel = torch.zeros(2, cfg.dim, dtype=torch.bfloat16)
        with mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._MXFP8GroupedGemmMLP.apply",
            return_value=sentinel,
        ) as fused, mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._MXFP8GroupedMLP.apply"
        ) as swiglu:
            out = module(x, num_tokens)
        fused.assert_called_once()
        swiglu.assert_not_called()
        x_arg, w13_arg, w2_arg, offs_arg = fused.call_args.args
        self.assertEqual(x_arg.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                w13_arg,
                _pack_w13_blocks(module.w1_EFD, module.w3_EFD).bfloat16(),
            )
        )
        self.assertTrue(torch.equal(w2_arg, module.w2_EDF.bfloat16()))
        self.assertEqual(offs_arg.dtype, torch.int32)
        self.assertEqual(
            offs_arg.tolist(), torch.cumsum(num_tokens, dim=0).tolist()
        )
        self.assertEqual(out.dtype, x.dtype)

    def test_plan_pads_the_dispatcher_to_256(self):
        model_config = self._plan_model_config()
        pads = [
            dispatcher_cfg.pad_multiple
            for _fqn, dispatcher_cfg, _parent, _attr in model_config.traverse(
                TorchAOTokenDispatcher.Config
            )
        ]
        self.assertTrue(pads)
        self.assertTrue(all(pad == 256 for pad in pads))
        nodes = list(model_config.traverse(MXFP8FusedGroupedMLP.Config))
        self.assertTrue(nodes)
        for _fqn, cfg, _parent, _attr in nodes:
            self.assertEqual(cfg.fusion_plan, "grouped_gemm_swiglu")

    def test_grouped_factory_raises_on_unknown_fusion_plan(self):
        cfg = self._stock_routed_experts_config()
        with self.assertRaisesRegex(ValueError, "unknown fusion_plan"):
            mxfp8_fused_grouped_mlp(cfg, fusion_plan="fully_fused")

    def test_grouped_factory_raises_on_unfused_activation_under_the_plan(self):
        cfg = self._stock_routed_experts_config()
        with self.assertRaisesRegex(ValueError, "always fuses the activation"):
            mxfp8_fused_grouped_mlp(
                cfg, fuse_activation=False, fusion_plan="grouped_gemm_swiglu"
            )

    def test_grouped_factory_raises_when_the_torchao_ops_are_unavailable(self):
        cfg = self._stock_routed_experts_config()
        with mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp."
            "_TORCHAO_GROUPED_MLP_OPS_AVAILABLE",
            False,
        ), mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp."
            "_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON",
            "unavailable for the test",
        ):
            with self.assertRaisesRegex(ValueError, "unavailable for the test"):
                mxfp8_fused_grouped_mlp(cfg, fusion_plan="grouped_gemm_swiglu")


# ---------------------------------------------------------------------------
# grouped_gemm_swiglu numerics (SM100 + torchao fused grouped-MLP ops only).
# Ported from the plan's gate-proven suite, adapted to stock parameters.
# ---------------------------------------------------------------------------

_BLOCK = 32
_E4M3 = torch.float8_e4m3fn
_RCEIL = ScaleCalculationMode.RCEIL

# Tolerances derived from the measured variability of the unfused lane itself
# (calibrated 2026-08-18 on GB200, torch 2.14.0a0): the two unfused MXFP8
# lanes score >= 112.3 dB against each other, fp32 scores 23.65-23.70 dB
# against the quantized reference, and the composite's measured band vs the
# reference is 35.30-35.70 dB (the fused GLU/dGLU kernels evaluate SwiGLU
# from their in-kernel FP32 accumulators while the reference rounds z and dh
# to BF16 first -- that one boundary dominates). The gate sits 5.3 dB below
# the measured band floor and 6.3 dB above the "independent-but-correct lane"
# level, so it still discriminates; a real dataflow/layout/offsets bug lands
# near 0 dB. Kernel-level exactness (60-160 dB) is enforced separately by the
# torchao op suite.
_SQNR_VS_REFERENCE_DB = 30.0
# Secondary tracking gate vs the FP32 eager MLP (measured 23.65-23.70 dB):
# catches a blind spot shared by both MXFP8 lanes.
_SQNR_VS_FP32_DB = 21.0

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


def _blk_view(w13):
    """[E, 2F, D] 32-block order -> view [E, F//32, 2, 32, D] with the
    gate/up axis at dim 2."""
    e, two_f, d = w13.shape
    return w13.view(e, two_f // 64, 2, 32, d)


def _to_blk(w1, w3):
    """Stock [E, F, D] pairs -> 32-block [E, 2F, D] (independent
    re-implementation; the module's pack is byte-verified against this
    mapping in the wiring tests)."""
    e, f, d = w1.shape
    return (
        torch.stack([w1, w3], dim=2)
        .view(e, f // 32, 32, 2, d)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, 2 * f, d)
    )


def _zsplit(z, f):
    """[R, 2F] in 32-block order -> (gate [R, F], up [R, F])."""
    r = z.shape[0]
    v = z.view(r, f // 32, 2, 32)
    return v[:, :, 0, :].reshape(r, f), v[:, :, 1, :].reshape(r, f)


def _zmerge(gate, up):
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


# Independent quantized-unfused reference: standalone torchao RCEIL casts
# (``to_mx``) + raw ``torch._scaled_grouped_mm`` + eager SwiGLU
# forward/backward (the BF16 round of z precedes SwiGLU; the BF16 rounds of
# h/dz precede their quantizers); wgrads are colwise quant-dequant + fp32
# matmul per expert. Built from first principles, sharing no code with the
# module under test. Operates directly on the 32-block ``w13 [G, 2F, D]``:
# rowwise quantization is per-row (row order is irrelevant) and colwise
# 32-blocks along 2F are pure-gate or pure-up in this order, so quantization
# boundaries match the composite exactly.


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


_MOD_D, _MOD_F, _MOD_E = 256, 256, 4
_MOD_SIZES = [256, 512, 256, 256]


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


@unittest.skipUnless(_HAS_SM100_GPU, "Requires CUDA SM 10.0 (Blackwell)")
@unittest.skipUnless(
    _TORCHAO_GROUPED_MLP_OPS_AVAILABLE,
    "torchao fused grouped-MLP ops unavailable: "
    f"{_TORCHAO_GROUPED_MLP_UNAVAILABLE_REASON}",
)
class TestGroupedGemmPlanNumerics(unittest.TestCase):
    """Numerics and fail-loud gating of the ``grouped_gemm_swiglu`` composite
    on real SM100 hardware with the torchao fused grouped-MLP ops."""

    def test_composite_matches_quantized_unfused_reference(self):
        for case in sorted(_CASES):
            with self.subTest(case=case):
                self._check_composite_case(**_CASES[case])

    def _check_composite_case(self, **case):
        fx = _make_case(**case)
        a, r, d = fx["a"], fx["r"], fx["x"].shape[1]

        x = fx["x"].clone().detach().requires_grad_(True)
        w13 = fx["w13"].clone().detach().requires_grad_(True)
        w2 = fx["w2"].clone().detach().requires_grad_(True)
        y = _MXFP8GroupedGemmMLP.apply(x, w13, w2, fx["offsets"])
        self.assertEqual(tuple(y.shape), (r, d))
        self.assertEqual(y.dtype, torch.bfloat16)
        y.backward(fx["dy"])

        ref_y, ref_dx, ref_dw13, ref_dw2 = _reference_forward_backward(
            fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], fx["sizes"], a
        )
        fp32 = _fp32_reference(
            fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], a
        )

        # Zero-token experts must produce exactly-zero weight gradients
        # through the autograd path (the wgrad op writes empty-group outputs
        # as zero).
        for gi, m in enumerate(fx["sizes"]):
            if m == 0:
                self.assertEqual(w13.grad[gi].abs().max().item(), 0.0)
                self.assertEqual(w2.grad[gi].abs().max().item(), 0.0)

        # y/dx are compared over active rows only: both lanes leave the
        # inactive tail [A, R) unwritten, and the garbage+NaN planted in the
        # x/dy tails (the negative control) must not move -- or NaN-poison --
        # any active output.
        for name, got, ref, hp in [
            ("y", y[:a], ref_y[:a], fp32[0]),
            ("dx", x.grad[:a], ref_dx[:a], fp32[1]),
            ("dw13", w13.grad, ref_dw13, fp32[2]),
            ("dw2", w2.grad, ref_dw2, fp32[3]),
        ]:
            self.assertTrue(
                torch.isfinite(got).all(), f"{name} contains non-finite values"
            )
            sqnr = compute_error(ref.float(), got.float())
            self.assertGreaterEqual(
                sqnr,
                _SQNR_VS_REFERENCE_DB,
                f"{name} SQNR vs quantized-unfused reference",
            )
            sqnr_hp = compute_error(hp.float(), got.float())
            self.assertGreaterEqual(sqnr_hp, _SQNR_VS_FP32_DB, f"{name} SQNR vs fp32")

    def test_composite_zero_routed_tokens(self):
        # A local expert set receiving zero routed tokens (R == 0, every
        # offset 0): the cast chain must produce empty quantized operands and
        # the ops' documented R == 0 early-outs must return empty y/dx and
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

        y = _MXFP8GroupedGemmMLP.apply(x, w13, w2, offsets)
        self.assertEqual(tuple(y.shape), (0, d))
        self.assertEqual(y.dtype, torch.bfloat16)
        y.backward(torch.zeros_like(y))

        self.assertIsNotNone(x.grad)
        self.assertEqual(tuple(x.grad.shape), (0, d))
        self.assertIsNotNone(w13.grad)
        self.assertEqual(w13.grad.abs().max().item(), 0.0)
        self.assertIsNotNone(w2.grad)
        self.assertEqual(w2.grad.abs().max().item(), 0.0)

    def _build_plan_module(self, seed):
        torch.manual_seed(seed)
        module = MXFP8FusedGroupedMLP.Config(
            dim=_MOD_D,
            hidden_dim=_MOD_F,
            num_experts=_MOD_E,
            fusion_plan="grouped_gemm_swiglu",
        ).build()
        module = module.to("cuda")
        with torch.no_grad():
            # fp32 master weights: the .bfloat16() casts stay outside the
            # Function so autograd routes bf16 grads back to fp32 params.
            module.w1_EFD.normal_(0.0, _MOD_D**-0.5)
            module.w3_EFD.normal_(0.0, _MOD_D**-0.5)
            module.w2_EDF.normal_(0.0, _MOD_D**-0.5)
        return module

    def test_torch_compile_matches_eager_bitwise(self):
        x, dy, num_tokens = _module_inputs()
        module = self._build_plan_module(seed=9)

        y_eager, dx_eager = _run_module(module, x, dy, num_tokens)
        grads_eager = [
            module.w1_EFD.grad.clone(),
            module.w3_EFD.grad.clone(),
            module.w2_EDF.grad.clone(),
        ]
        module.zero_grad(set_to_none=True)

        compiled = torch.compile(module)
        y_comp, dx_comp = _run_module(compiled, x, dy, num_tokens)

        # Deterministic kernels + compile-invariant surrounding data movement
        # (cumsum, pack, dtype casts) => bitwise-identical results.
        self.assertTrue(torch.equal(y_comp, y_eager))
        self.assertTrue(torch.equal(dx_comp, dx_eager))
        for got, ref in zip(
            [module.w1_EFD.grad, module.w3_EFD.grad, module.w2_EDF.grad],
            grads_eager,
        ):
            self.assertTrue(torch.equal(got, ref))

    def test_plan_checkpoint_keys_match_stock(self):
        stock_nodes = list(
            deepseek_v3_model_registry("debugmodel").model.traverse(
                GroupedExperts.Config
            )
        )
        self.assertTrue(stock_nodes)
        model_config = deepseek_v3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_GROUPED_PLAN_IMPORT]), model_config)
        plan_nodes = list(model_config.traverse(MXFP8FusedGroupedMLP.Config))
        self.assertTrue(plan_nodes)
        with torch.device("meta"):
            stock = stock_nodes[0][1].build()
            fused = plan_nodes[0][1].build()
        self.assertEqual(set(fused.state_dict().keys()), set(stock.state_dict().keys()))

    def test_factory_raises_on_converter_quantized_experts(self):
        # The composite quantizes every grouped GEMM itself; layering it on
        # the MXFP8 grouped-experts converter's output is a config error that
        # must raise -- never a silent fallback to the converter's unfused
        # path.
        model_config = deepseek_v3_model_registry(
            "debugmodel",
            converters=[MXFP8GroupedExpertsConverter.Config(pad_multiple=128)],
        ).model
        nodes = list(model_config.traverse(RoutedExperts.Config))
        self.assertTrue(nodes)
        with self.assertRaisesRegex(ValueError, "grouped-experts converter"):
            mxfp8_fused_grouped_mlp(nodes[0][1], fusion_plan="grouped_gemm_swiglu")

    def test_factory_raises_on_unsupported_dims(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        node = list(model_config.traverse(RoutedExperts.Config))[0][1]
        node.inner_experts.hidden_dim = 100  # not a 128-multiple
        with self.assertRaisesRegex(ValueError, "is_supported"):
            mxfp8_fused_grouped_mlp(node, fusion_plan="grouped_gemm_swiglu")

    def test_factory_raises_on_non_alltoall_dispatcher(self):
        # hybridep's padded dispatcher is not validated for the fused
        # kernels' 256-row contract; the factory must refuse it rather than
        # swap or accept it.
        model_config = deepseek_v3_model_registry(
            "debugmodel",
            moe_comm_backend="hybridep",
            non_blocking_capacity_factor=1.0,
        ).model
        node = list(model_config.traverse(RoutedExperts.Config))[0][1]
        with self.assertRaisesRegex(ValueError, "TorchAO padded"):
            mxfp8_fused_grouped_mlp(node, fusion_plan="grouped_gemm_swiglu")


if __name__ == "__main__":
    unittest.main()
