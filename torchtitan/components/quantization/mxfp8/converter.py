# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib import import_module
from importlib.util import find_spec
from typing import Literal

import torch

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from ..utils import swap_token_dispatcher

_mxfp8_linear_import_error: ImportError | None = None

try:
    # Nothing about the class itself can fail here. What raises is two levels
    # down: linear.py and tensor.py import torchao's mxfp8 cast kernels at
    # module scope, and triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale
    # is newer than any torchao release. Catching it keeps
    # ``import torchtitan.components.quantization`` working for float8 and
    # nvfp4 users, and defers the error to whoever builds this converter.
    from .linear import MXFP8Linear

except ImportError as import_error:
    MXFP8Linear = None
    _mxfp8_linear_import_error = import_error


FusionPlan = Literal["none", "swiglu", "grouped_gemm_swiglu"]
_FUSION_PLANS = ("none", "swiglu", "grouped_gemm_swiglu")
# Contract of the fused plans (implemented in grouped_experts.py): every
# expert's token group is zero-padded to this many rows by the token
# dispatcher, and the expert dimensions are multiples of the kernels' tile.
_FUSION_PLAN_PAD_MULTIPLES = {"swiglu": 128, "grouped_gemm_swiglu": 256}
_FUSED_DIM_ALIGNMENT = 128
# torchao kernel module of each fused plan (under moe_training.kernels.mxfp8)
# and the PR that added it; neither is in a torchao release yet.
_FUSION_PLAN_KERNELS = {
    "swiglu": ("cutedsl_gated_act_mxfp8", "pytorch/ao#4743"),
    "grouped_gemm_swiglu": ("cudnn_grouped_mlp", "pytorch/ao#4820"),
}


def _import_fusion_plan_kernels(fusion_plan: str) -> None:
    """Import what a fused plan runs on, so a torchao that predates the plan's
    kernels or a missing runtime package fails at converter construction
    rather than at the first expert forward.
    """
    kernels, pull_request = _FUSION_PLAN_KERNELS[fusion_plan]
    try:
        import_module(f"torchao.prototype.moe_training.kernels.mxfp8.{kernels}")
        if fusion_plan == "grouped_gemm_swiglu":
            # torchao's cuDNN ops import their runtime lazily at first launch.
            import_module("cudnn")
        from . import grouped_experts  # noqa: F401
    except ImportError as import_error:
        raise ImportError(
            f"MXFP8 fusion_plan={fusion_plan!r} needs torchao's {kernels} kernels "
            f"({pull_request}) and their runtime: {import_error}"
        ) from import_error


class MXFP8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with MXFP8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted.
        """
        linears_saving_inputs_for_backward_in_mxfp8: list[str] = field(
            default_factory=list
        )
        """FQN substrings selecting linears that save inputs in MXFP8 for backward.

        A linear can save either its BF16 input or a columnwise MXFP8 input for
        the backward pass.

        Without activation checkpointing, if the preceding operation already
        saves its BF16 output for backward, as flash attention does, that tensor
        is also available as this linear's input. Saving another MXFP8
        operands would increase memory usage, so this linear should save
        BF16. If no other operation retains the BF16 input, saving MXFP8 reduces
        activation memory and avoids columnwise quantization during backward.

        With full activation checkpointing, saved tensors from the original
        forward are discarded and reconstructed during backward. Today, a
        linear selected here produces its columnwise MXFP8 input in both the
        original forward and recomputation, even though the original result is
        discarded. An ideal checkpoint-aware policy could produce it only
        during recomputation, but distinguishing those executions would add
        complexity to the linear and its autograd contract. We intentionally
        apply the same policy to both.

        More granular ``torch.remat`` policies add further save-versus-recompute
        choices, so the optimal format depends on both model activation
        ownership and the activation-checkpointing policy. BF16 is therefore
        the conservative default, and users can opt selected modules into MXFP8
        with this list.
        """

        def __post_init__(self) -> None:
            if any(not fqn for fqn in self.linears_saving_inputs_for_backward_in_mxfp8):
                raise ValueError(
                    "MXFP8 linears_saving_inputs_for_backward_in_mxfp8 cannot "
                    "contain an empty FQN selector."
                )

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "MXFP8 linear layers need torchao's 32x32 swizzled cast "
                "kernels, added in pytorch/ao#4777 and not in any release up "
                "to v0.18.0. Install a torchao that contains it."
            ) from _mxfp8_linear_import_error

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

    def convert(self, model_config):
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        targets = [
            entry
            for entry in model_config.traverse(Linear.Config)
            if not fqns or any(target_fqn in entry[0] for target_fqn in fqns)
        ]

        selectors = self.config.linears_saving_inputs_for_backward_in_mxfp8
        target_fqns = [fqn for fqn, _config, _parent, _attr in targets]
        unmatched_fqn_selectors = {
            selector
            for selector in selectors
            if not any(selector in fqn for fqn in target_fqns)
        }
        if unmatched_fqn_selectors:
            raise ValueError(
                "MXFP8 linears_saving_inputs_for_backward_in_mxfp8 selectors "
                "did not match any converted Linear.Config: "
                f"{sorted(unmatched_fqn_selectors)}."
            )

        mxfp8_fqns = {
            fqn for fqn in target_fqns if any(selector in fqn for selector in selectors)
        }
        for fqn, config, parent, attr in targets:
            new_config = MXFP8Linear.Config(
                in_features=config.in_features,
                out_features=config.out_features,
                bias=config.bias,
                param_init=config.param_init,
                input_activation_format_for_backward=(
                    "mxfp8" if fqn in mxfp8_fqns else "bf16"
                ),
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        num_mxfp8 = len(mxfp8_fqns)
        num_bf16 = len(targets) - num_mxfp8
        logger.info(
            "Converted Linear layers to MXFP8Linear with saved input activation "
            f"formats: {num_bf16} bf16, {num_mxfp8} mxfp8"
        )
        logger.debug(f"Linears saving MXFP8 input activations: {sorted(mxfp8_fqns)}")
        return model_config


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``_quantize_then_scaled_grouped_mm``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"
            fusion_plan: FusionPlan = "none"

            def __post_init__(self) -> None:
                if self.fusion_plan not in _FUSION_PLANS:
                    raise ValueError(
                        f"MXFP8 fusion_plan must be one of {_FUSION_PLANS}; got "
                        f"{self.fusion_plan!r}."
                    )
                if self.fusion_plan == "none":
                    return
                # The fused kernels quantize with RCEIL scales internally.
                if self.recipe_name != "mxfp8_rceil":
                    raise ValueError(
                        f"MXFP8 fusion_plan={self.fusion_plan!r} supports only "
                        f"recipe_name='mxfp8_rceil'; got {self.recipe_name!r}."
                    )
                # The fused composites implement the stock SwiGLU MLP; other
                # experts modules (GptOssGroupedExperts: biases, clamped SwiGLU)
                # keep the per-GEMM quantization.
                if parent_cls is not GroupedExperts:
                    raise ValueError(
                        f"MXFP8 fusion_plan={self.fusion_plan!r} supports only "
                        f"GroupedExperts, not {parent_cls.__name__}; use "
                        "fusion_plan='none' for this model."
                    )

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            self._mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            self.fusion_plan = config.fusion_plan

        def _grouped_mlp(self, *, x_RD, w1_EFD, w2_EDF, w3_EFD, offsets_E):
            if self.fusion_plan == "none":
                return super()._grouped_mlp(
                    x_RD=x_RD,
                    w1_EFD=w1_EFD,
                    w2_EDF=w2_EDF,
                    w3_EFD=w3_EFD,
                    offsets_E=offsets_E,
                )
            from . import grouped_experts

            # Casts stay outside the composites so autograd covers high-precision
            # master weights; the packed gate/up operand is a differentiable copy
            # of the stock parameters. The output takes ``x_RD``'s dtype like the
            # stock MLP.
            x_bf16_RD = x_RD.bfloat16()
            w1_EFD, w2_EDF, w3_EFD = (
                w1_EFD.bfloat16(),
                w2_EDF.bfloat16(),
                w3_EFD.bfloat16(),
            )
            grouped_experts._validate_inputs(x_bf16_RD, w1_EFD, self.fusion_plan)
            if self.fusion_plan == "swiglu":
                out_RD = grouped_experts._MXFP8SwiGLUFusionFunction.apply(
                    x_bf16_RD,
                    torch.cat([w1_EFD, w3_EFD], dim=1),
                    w2_EDF.transpose(-2, -1),
                    offsets_E,
                )
            else:
                out_RD = grouped_experts._MXFP8GroupedGemmSwiGLUFusionFunction.apply(
                    x_bf16_RD,
                    grouped_experts._pack_w13_blocks(w1_EFD, w3_EFD),
                    w2_EDF,
                    offsets_E,
                )
            return out_RD.type_as(x_RD)

        def _grouped_mm(self, *, A, weight_EOI, offs):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                A,
                weight_EOI.bfloat16().transpose(-2, -1),
                config=self._mxfp8_op_config,
                offs=offs,
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        """
        pad_multiple: int = 32
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        """
        fusion_plan: FusionPlan = "none"
        """
        Which part of the expert SwiGLU MLP runs as one fused torchao kernel.

        - none: three separately quantized grouped GEMMs with the SwiGLU in
          BF16 between them.
        - swiglu: the two grouped GEMMs stay separate; the SwiGLU and both MXFP8
          quantizations of its output (rowwise for the down GEMM, columnwise for
          the down weight gradient) run in one CuTeDSL kernel (torchao
          ``cutedsl_gated_act_mxfp8``, pytorch/ao#4743). Requires ``pad_multiple``
          a multiple of 128 and ``dim`` / ``hidden_dim`` multiples of 128.
        - grouped_gemm_swiglu: grouped GEMM, SwiGLU and MXFP8 quantization fused
          in one cuDNN kernel per direction (torchao ``cudnn_grouped_mlp`` ops,
          pytorch/ao#4820). Requires ``pad_multiple`` a multiple of 256 (the
          kernels' fixed group padding) and ``dim`` / ``hidden_dim`` multiples
          of 128.

        Both fused plans support only the stock ``GroupedExperts`` and raise at
        config time otherwise; there is no silent fallback.
        """

        def __post_init__(self) -> None:
            if self.fusion_plan not in _FUSION_PLANS:
                raise ValueError(
                    f"MXFP8 fusion_plan must be one of {_FUSION_PLANS}; got "
                    f"{self.fusion_plan!r}."
                )
            if self.fusion_plan != "none" and (
                self.pad_multiple % _FUSION_PLAN_PAD_MULTIPLES[self.fusion_plan]
            ):
                raise ValueError(
                    f"MXFP8 fusion_plan={self.fusion_plan!r} requires pad_multiple "
                    f"to be a multiple of {_FUSION_PLAN_PAD_MULTIPLES[self.fusion_plan]}"
                    f"; got {self.pad_multiple}."
                )

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if self.config.fusion_plan != "none":
            _import_fusion_plan_kernels(self.config.fusion_plan)
        if (
            self.config.fusion_plan == "grouped_gemm_swiglu"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() != (10, 0)
        ):
            # The cuDNN grouped-MLP wrappers are built for SM 10.0 only.
            raise ValueError(
                "MXFP8 fusion_plan='grouped_gemm_swiglu' runs only on SM 10.0 "
                f"GPUs; got {torch.cuda.get_device_capability()}."
            )

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            if self.config.fusion_plan != "none" and not isinstance(
                parent.token_dispatcher, AllToAllTokenDispatcher.Config
            ):
                # The fused kernels' padded-row contract is validated only
                # against TorchAOTokenDispatcher's permute_and_pad.
                raise ValueError(
                    f"MXFP8 fusion_plan={self.config.fusion_plan!r} requires the "
                    f"all-to-all token dispatcher; got "
                    f"{type(parent.token_dispatcher).__qualname__}. Use "
                    "fusion_plan='none' with this dispatcher."
                )
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
                fusion_plan=self.config.fusion_plan,
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm ops"
        )
        return model_config
