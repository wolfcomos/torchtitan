# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantization converter.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4Linear`, which keeps a bf16
weight and quantizes activations, weights, and gradients to NVFP4 on the fly via
TorchAO's ``nvfp4_training`` kernels (NVIDIA Blackwell / sm_100+, CUDA only).

Like :class:`MXFP8LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Under tensor
parallelism the block boundary keeps its stock bf16 collectives (all-gather /
reduce-scatter); NVFP4 does not move fp4 codes over the wire.
"""

import math
from dataclasses import dataclass, field, fields, replace
from typing import cast

import spmd_types as spmd
import torch

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, SpmdLayout
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

TP = MeshAxisName.TP

# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

# Fixed Random Hadamard Transform basis (the NVFP4 v1 recipe default in torchao
# and Transformer Engine). It must be identical across TP ranks -- rowwise TP
# shards the GEMM contraction dim, and the Hadamard transform only cancels
# between the two operands when both use the same sign vector. Hardcoding it
# makes every rank produce the same vector by construction (no cross-rank
# broadcast). Per-recipe dynamic sign vectors are a future extension.
_HARDCODED_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)

try:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
        nvfp4_linear,
        nvfp4_mm_triton,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        _make_rht_sign_vector,
        _rht_sign_vector_to_tuple,
        NVFP4Linear as TorchAONVFP4Linear,
    )

    # The NVFP4 GEMM is a raw autograd Function that runs on local shards inside
    # the spmd.local_map region. Mark it local-safe so SPMD type checking
    # propagates through it; the local_map boundary declares the real
    # colwise/rowwise output and input-gradient types.
    spmd.register_local_autograd_function(nvfp4_mm_triton)

    class NVFP4Linear(TorchAONVFP4Linear, Module):
        """NVFP4 Linear satisfying torchtitan's Module protocol.

        Reuses TorchAO's ``NVFP4Linear`` (weight/bias, the ``_sr_seed`` /
        ``_rht_sign_vector`` runtime buffers, RHT logic, functional forward) and
        adds torchtitan's meta-init buffer protocol and local SPMD sharding.
        ``_rht_sign_vector`` is the fixed ``_HARDCODED_SIGN_VECTOR`` (identical on
        every rank by construction) and ``_sr_seed`` is per-rank.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds NVFP4Linear."""

            def __post_init__(self) -> None:
                # NVFP4's Triton kernels need every GEMM dim to be a multiple of
                # 128. in_features / out_features are known at config-build time
                # (the TP degree is not), so reject the model-dim violations up
                # front here; the AO kernel (nvfp4_mm_triton) itself raises on the
                # per-rank local dims once TP has sharded the weight.
                for name in ("in_features", "out_features"):
                    value = getattr(self, name)
                    if value % _NVFP4_BLOCK:
                        raise ValueError(
                            f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                            f"got {name}={value}. NVFP4 cannot quantize this Linear; "
                            "exclude it from the converter fqns."
                        )

            def build(self, **kwargs):
                # sharding_config (the stock colwise/rowwise weight placement) is
                # attached by update_from_config after this Config is built, so it
                # is available here but not in __post_init__. Fold it into the
                # local_map region for the opaque nvfp4_linear op now, so base
                # Module.parallelize consumes it directly.
                # slots=True breaks zero-arg super(), so call the parent explicitly.
                instance = Linear.Config.build(self, **kwargs)
                if instance._sharding_config is not None:
                    sc = instance._sharding_config
                    weight_tp = (
                        sc.state_shardings["weight"].per_axis_spmd_types().get(TP)
                    )
                    rowwise = isinstance(weight_tp, spmd.Shard) and weight_tp.dim == 1
                    if rowwise:
                        in_layout = dense_activation_placement(
                            tp=spmd.S(-1), cp=spmd.S(0)
                        )
                        in_grad = dense_activation_placement(
                            tp=spmd.S(-1), cp=spmd.S(0)
                        )
                    else:
                        in_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
                        in_grad = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
                    instance._sharding_config = replace(
                        sc,
                        state_shardings={
                            **sc.state_shardings,
                            "_sr_seed": SpmdLayout(
                                {
                                    MeshAxisName.DP: spmd.V,
                                    MeshAxisName.CP: spmd.V,
                                    TP: spmd.V,
                                }
                            ),
                        },
                        in_src_shardings={
                            **(sc.in_src_shardings or {}),
                            "x": in_layout,
                        },
                        in_dst_shardings={
                            **(sc.in_dst_shardings or {}),
                            "x": in_layout,
                        },
                        local_map=LocalMapConfig(in_grad_placements=(in_grad,)),
                    )
                return instance

        def __init__(self, config: Linear.Config):
            TorchAONVFP4Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )
            # TorchAO created the runtime buffers on the (meta) build device.
            # Re-register them as None so ``_distribute_states`` skips them and
            # ``_init_self_buffers`` materializes them on the real device, per
            # torchtitan's buffer protocol.
            # _sr_seed is a stochastic-rounding seed drawn locally per rank with
            # no cross-rank coordination. Ranks that share an RNG stream (all but
            # the pp axis, which set_determinism seeds distinctly) draw the same
            # value, but that is fine: SR stays unbiased and NVFP4 never
            # communicates quantized values, so the seed need not differ across
            # ranks. It is non-persistent (a Philox key needs no checkpointing).
            # Re-register it None so it is not distributed and is re-drawn per
            # rank in _init_self_buffers.
            self.register_buffer("_sr_seed", None, persistent=False)
            # _rht_sign_vector is the fixed _HARDCODED_SIGN_VECTOR (see module
            # top): identical on every rank, so it is non-persistent (a
            # deterministic constant needs no checkpointing) and re-materialized
            # per rank in _init_self_buffers with no cross-rank broadcast.
            self.register_buffer("_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None

        def _local_rht_sign_vector(self) -> torch.Tensor:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            return sign_vector

        def _refresh_rht_sign_vector_tuple(self) -> None:
            sign_vector = self._local_rht_sign_vector()
            self._rht_sign_vector_tuple = (
                None if sign_vector is None else _rht_sign_vector_to_tuple(sign_vector)
            )

        def _load_from_state_dict(self, *args, **kwargs):
            super()._load_from_state_dict(*args, **kwargs)
            self._refresh_rht_sign_vector_tuple()

        @property
        def rht_sign_vector(self) -> tuple[int, ...]:
            if self._rht_sign_vector_tuple is None:
                self._refresh_rht_sign_vector_tuple()
            if self._rht_sign_vector_tuple is None:
                raise RuntimeError("rht_sign_vector is not materialized")
            return self._rht_sign_vector_tuple

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            dev = (
                buffer_device
                if buffer_device is not None
                else cast(torch.Tensor, self.weight).device
            )
            # Per-rank seed: a plain local tensor (not distributed), so each rank
            # draws its own.
            self._sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            # Static RHT basis: identical on every rank by construction, so it is
            # a plain local tensor with no cross-rank broadcast.
            self._rht_sign_vector = _make_rht_sign_vector(
                _HARDCODED_SIGN_VECTOR, device=dev
            )
            self._refresh_rht_sign_vector_tuple()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nvfp4_linear(
                x,
                self.weight,
                self.bias,
                sr_seed=self._sr_seed,
                sign_vector=self.rht_sign_vector,
            )

except ImportError:
    NVFP4Linear = None


def nvfp4_bf16_tail_fqns(num_layers: int, bf16_tail_fraction: float) -> list[str]:
    """Converter ``fqns`` selecting the leading decoder layers for NVFP4 while
    keeping the last ``ceil(num_layers * bf16_tail_fraction)`` layers in bf16.

    Each fqn has a trailing '.' so 'layers.1.' matches layer 1 only, not
    'layers.10' (NVFP4LinearConverter.convert substring-matches). Raises if the
    fraction would leave no layer to convert: an empty fqns list would instead
    convert *all* Linears (the ``not fqns`` branch in convert), the opposite of
    the intent.
    """
    num_bf16 = math.ceil(num_layers * bf16_tail_fraction)
    convert_upto = num_layers - num_bf16
    if convert_upto <= 0:
        raise ValueError(
            f"bf16_tail_fraction={bf16_tail_fraction} keeps all {num_layers} "
            "layers in bf16; nothing to convert to NVFP4."
        )
    return [f"layers.{i}." for i in range(convert_upto)]


def nvfp4_bf16_first_last_fqns(
    num_layers: int, num_start_layers_bf16: int, num_end_layers_bf16: int
) -> list[str]:
    """Converter ``fqns`` keeping the first ``num_start_layers_bf16`` and last
    ``num_end_layers_bf16`` decoder layers in bf16 (the miles NVFP4 RL recipes'
    ``--first-last-layers-bf16`` analog).

    Each fqn has a trailing '.' so 'layers.1.' matches layer 1 only, not
    'layers.10' (the converters substring-match). Raises if the window would
    leave no layer to convert: an empty fqns list would instead convert *all*
    matching modules (the ``not fqns`` branch in convert), the opposite of the
    intent.
    """
    convert_from = num_start_layers_bf16
    convert_upto = num_layers - num_end_layers_bf16
    if convert_from >= convert_upto:
        raise ValueError(
            f"num_start_layers_bf16={num_start_layers_bf16} and "
            f"num_end_layers_bf16={num_end_layers_bf16} keep all {num_layers} "
            "layers in bf16; nothing to convert to NVFP4."
        )
    return [f"layers.{i}." for i in range(convert_from, convert_upto)]


def _validate_four_over_six_knobs(
    backward_override: str | None,
    weight_block: str,
    *,
    row_scaled_activation: bool = False,
    grouped: bool = False,
) -> None:
    """Reject invalid four-over-six knob combinations at config time.

    Mirrors the TorchAO ops' runtime checks so a bad recipe fails when the
    config tree is built rather than on the first forward. Grouped GEMMs have
    no quantized backward (TransformerEngine rejects four-over-six group
    quantization), and a row-scaled four-over-six tensor has no columnwise
    form for the quantized wgrad operand.
    """
    allowed = (
        (None, "high_precision", "dequantized")
        if grouped
        else (None, "quantized", "high_precision", "dequantized")
    )
    if backward_override not in allowed:
        raise ValueError(
            f"backward_override must be one of {allowed}; "
            f"got {backward_override!r}"
        )
    if backward_override == "quantized" and row_scaled_activation:
        raise ValueError(
            "row-scaled four-over-six has no quantized backward; use "
            "'high_precision' or 'dequantized'"
        )
    if weight_block not in ("1x16", "16x16"):
        raise ValueError(
            f"weight_block must be '1x16' or '16x16', got {weight_block!r}"
        )


class NVFP4LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply NVFP4 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted -- pass explicit fqns to keep
        the LM head in bf16, which the mixed recipe leaves unquantized for stability.
        """

    def __init__(self, config: Config):
        self.config = config

        if NVFP4Linear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 training "
                "prototype. Install a torchao build with "
                "torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert NVFP4Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = NVFP4Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config


try:
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        four_over_six_linear,
        four_over_six_mm,
    )

    # Like nvfp4_mm_triton above: an opaque local autograd function, marked
    # local-safe so SPMD type checking propagates through it.
    spmd.register_local_autograd_function(four_over_six_mm)

    class NVFP4FourOverSixLinear(Linear):
        """Linear with NVFP4 four-over-six quantized GEMMs.

        Four-over-six encodes every FP4 block twice (the standard map-to-6
        scale and a 1.5x-expanded map-to-4 scale) and stores the candidate
        with the lower dequantization error. Forward GEMM operands are
        four-over-six (activations 1x16, optionally row-scaled; weights
        16x16); gradients use standard NVFP4, or bf16 GEMMs when
        ``row_scaled_activation`` is set (a row-scaled four-over-six tensor
        has no columnwise form for the wgrad operand). No RHT and no
        stochastic rounding: the recipe targets RL and post-training.
        Pure leaf swap like :class:`MXFP8Linear`; TP is not wired up.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds
            NVFP4FourOverSixLinear."""

            err_mode: str = "mae"
            """Candidate-selection error metric, 'mae' or 'mse'."""

            e4m3_scale_bound: int = 256
            """Global E4M3 scale bound; 256 leaves map-to-4 headroom."""

            row_scaled_activation: bool = False
            """One FP32 global scale per activation row instead of per
            tensor. Selects the bf16 backward."""

            backward_override: str | None = None
            """'quantized', 'high_precision', or 'dequantized' (mirrors
            TransformerEngine's NVTE_BACKWARD_OVERRIDE). None keeps the
            recipe defaults: quantized backward per tensor, high-precision
            backward when row-scaled."""

            weight_block: str = "16x16"
            """Weight tile granularity; '1x16' mirrors
            NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1."""

            def __post_init__(self) -> None:
                for name in ("in_features", "out_features"):
                    value = getattr(self, name)
                    if value % _NVFP4_BLOCK:
                        raise ValueError(
                            f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                            f"got {name}={value}. NVFP4 cannot quantize this Linear; "
                            "exclude it from the converter fqns."
                        )
                _validate_four_over_six_knobs(
                    self.backward_override,
                    self.weight_block,
                    row_scaled_activation=self.row_scaled_activation,
                )

        def __init__(self, config: Config):
            Linear.__init__(self, config)
            self.err_mode = config.err_mode
            self.e4m3_scale_bound = config.e4m3_scale_bound
            self.row_scaled_activation = config.row_scaled_activation
            self.backward_override = config.backward_override
            self.weight_block = config.weight_block

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return four_over_six_linear(
                x,
                self.weight,
                self.bias,
                self.err_mode,
                self.e4m3_scale_bound,
                self.row_scaled_activation,
                self.backward_override,
                self.weight_block,
            )

except ImportError:
    NVFP4FourOverSixLinear = None


class NVFP4FourOverSixLinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4FourOverSixLinear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply four-over-six NVFP4
        quantization to. Only Linear.Config entries whose FQN contains a match
        are converted. If empty, all Linear modules are converted -- pass
        explicit fqns to keep the LM head in bf16.
        """

        err_mode: str = "mae"
        """Candidate-selection error metric, 'mae' or 'mse'."""

        e4m3_scale_bound: int = 256
        """Global E4M3 scale bound; 256 leaves map-to-4 headroom."""

        row_scaled_activation: bool = False
        """One FP32 global scale per activation row instead of per tensor."""

        backward_override: str | None = None
        """'quantized', 'high_precision', or 'dequantized' (mirrors
        TransformerEngine's NVTE_BACKWARD_OVERRIDE). None keeps the recipe
        defaults."""

        weight_block: str = "16x16"
        """Weight tile granularity; '1x16' mirrors
        NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1."""

    def __init__(self, config: Config):
        self.config = config

        if NVFP4FourOverSixLinear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 "
                "four-over-six training prototype. Install a torchao build "
                "with torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert NVFP4FourOverSixLinear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = NVFP4FourOverSixLinear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    err_mode=self.config.err_mode,
                    e4m3_scale_bound=self.config.e4m3_scale_bound,
                    row_scaled_activation=self.config.row_scaled_activation,
                    backward_override=self.config.backward_override,
                    weight_block=self.config.weight_block,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4FourOverSixLinear")
        return model_config


try:
    from torchao.prototype.moe_training.nvfp4_training.four_over_six_grouped import (
        four_over_six_grouped_mm,
    )

except ImportError:
    four_over_six_grouped_mm = None


_four_over_six_experts_cache: dict[type, type] = {}


def _get_four_over_six_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create a four-over-six-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``), like
    ``_get_mxfp8_grouped_experts_cls``. The returned class has a proper
    ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``four_over_six_grouped_mm``. Four-over-six needs no RHT sign vector and no
    stochastic-rounding seed, so the stateless hook carries everything.
    """
    if parent_cls in _four_over_six_experts_cache:
        return _four_over_six_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class FourOverSixGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            err_mode: str = "mae"
            e4m3_scale_bound: int = 256
            row_scaled_activation: bool = False
            backward_override: str | None = None
            weight_block: str = "16x16"

            def __post_init__(self) -> None:
                # The grouped GEMM requires K % 128 == 0 and N % 128 == 0; both
                # projections (dim x hidden_dim and hidden_dim x dim) hit both
                # dims, so reject the model-dim violations up front here.
                for name in ("dim", "hidden_dim"):
                    value = getattr(self, name)
                    if value % _NVFP4_BLOCK:
                        raise ValueError(
                            f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                            f"got {name}={value}. NVFP4 cannot quantize these "
                            "grouped experts; exclude them from the converter fqns."
                        )
                _validate_four_over_six_knobs(
                    self.backward_override, self.weight_block, grouped=True
                )

        def __init__(self, config: Config):
            super().__init__(config)
            self._four_over_six_kwargs = dict(
                err_mode=config.err_mode,
                e4m3_scale_bound=config.e4m3_scale_bound,
                row_scaled_activation=config.row_scaled_activation,
                weight_block=config.weight_block,
                backward_override=config.backward_override,
            )

        def _grouped_mm(self, *, A, B_t, offs):
            # The hook receives B_t pre-transposed to (E, K, N); the torchao op
            # takes expert weights in their stored (E, N, K) layout (it makes
            # the operands contiguous itself).
            #
            # The swapped TorchAOTokenDispatcher already 128-aligns every
            # expert group but over-allocates the activation buffer past
            # offs[-1], while the torchao op requires offs[-1] == A.shape[0]
            # and the unwritten tail rows must not feed the per-group amaxes.
            # Slice to the logical rows, skip the op's own padding, and
            # zero-extend the output (pad routes zero grads to the tail;
            # the unpermute never reads tail rows).
            m_total = int(offs[-1])
            out = four_over_six_grouped_mm(
                A[:m_total],
                B_t.transpose(-2, -1),
                offs,
                pad_token_groups_for_grouped_mm=False,
                **self._four_over_six_kwargs,
            )
            return torch.nn.functional.pad(out, (0, 0, 0, A.shape[0] - m_total))

    FourOverSixGroupedExperts.__name__ = f"NVFP4FourOverSix{parent_cls.__name__}"
    FourOverSixGroupedExperts.__qualname__ = f"NVFP4FourOverSix{parent_cls.__name__}"
    _four_over_six_experts_cache[parent_cls] = FourOverSixGroupedExperts
    return FourOverSixGroupedExperts


class NVFP4FourOverSixGroupedExpertsConverter(QuantizationConverter):
    """Apply four-over-six NVFP4 quantization to MoE expert grouped GEMMs.

    The miles NVFP4 RL recipes quantize only the routed-expert projections;
    with no Linear converter alongside, this converter alone reproduces that
    allow-list (attention, dense MLP, shared experts, router, embeddings, and
    the LM head all stay bf16).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply four-over-six NVFP4
        quantization to. Only GroupedExperts.Config entries whose FQN contains
        a match are converted. If empty, all GroupedExperts are converted.
        """

        err_mode: str = "mae"
        """Candidate-selection error metric, 'mae' or 'mse'."""

        e4m3_scale_bound: int = 256
        """Global E4M3 scale bound; 256 leaves map-to-4 headroom."""

        row_scaled_activation: bool = False
        """One FP32 global scale per activation row instead of per token
        group."""

        backward_override: str | None = None
        """'high_precision' (the default when None) or 'dequantized'; grouped
        four-over-six has no quantized backward."""

        weight_block: str = "16x16"
        """Weight tile granularity; '1x16' mirrors
        NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1."""

        pad_multiple: int = 128
        """
        Pad per-expert token groups to this multiple for NVFP4 grouped GEMM
        alignment (the four-over-six grouped GEMM requires 128-row groups).
        """

    def __init__(self, config: Config):
        self.config = config

        if four_over_six_grouped_mm is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 "
                "four-over-six grouped training prototype. Install a torchao "
                "build with torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert four_over_six_grouped_mm is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            if fqns and not any(target_fqn in fqn for target_fqn in fqns):
                continue
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_four_over_six_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                err_mode=self.config.err_mode,
                e4m3_scale_bound=self.config.e4m3_scale_bound,
                row_scaled_activation=self.config.row_scaled_activation,
                backward_override=self.config.backward_override,
                weight_block=self.config.weight_block,
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic NVFP4 four-over-six "
            "quantization for grouped_mm ops"
        )
        return model_config
