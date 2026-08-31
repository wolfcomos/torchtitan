# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields, replace
from importlib.util import find_spec
from typing import Literal

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

try:
    from torchao.prototype.moe_training.mxfp8_linear import (
        MXFP8Linear as TorchAOMXFP8Linear,
    )

    class MXFP8Linear(TorchAOMXFP8Linear, Module):
        """Inherits from Module (not Linear) to satisfy the Module protocol
        (init_states, _param_init) while avoiding MRO conflicts with
        Linear.__init__. Config still inherits from Linear.Config for
        field compatibility.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

            pass

        def __init__(self, config: Config):
            TorchAOMXFP8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )

except ImportError:
    MXFP8Linear = None


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

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
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

        logger.info("Converted Linear layers to MXFP8Linear")
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
            backward_override: str | None = None

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            self._mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            # None and "quantized" both keep the stock quantized backward, so
            # normalize before probing: "quantized" then works on installed
            # torchao builds that predate the backward_override knob, and
            # replace() only runs when an actual override is set.
            backward_override = config.backward_override
            if backward_override == "quantized":
                backward_override = None
            if backward_override is not None:
                if "backward_override" not in {
                    f.name for f in fields(MXFP8TrainingOpConfig)
                }:
                    raise ValueError(
                        "The installed torchao does not support backward_override "
                        "for MXFP8 grouped GEMMs. Install a torchao build whose "
                        "MXFP8TrainingOpConfig has the backward_override field."
                    )
                self._mxfp8_op_config = replace(
                    self._mxfp8_op_config,
                    backward_override=backward_override,
                )

        def _grouped_mm(self, *, A, B_t, offs):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                A, B_t, config=self._mxfp8_op_config, offs=offs
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only GroupedExperts.Config entries whose FQN contains a match are
        converted. If empty, all GroupedExperts are converted.
        """

        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        """
        backward_override: (
            Literal["quantized", "high_precision", "dequantized"] | None
        ) = None
        """
        Backward computation override for the grouped GEMMs. None or
        "quantized" keeps the quantized MXFP8 backward, "high_precision"
        computes gradients from the saved high-precision operands, and
        "dequantized" computes them from the dequantized forward operands.
        """
        pad_multiple: int = 32
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        allowed = (None, "quantized", "high_precision", "dequantized")
        if self.config.backward_override not in allowed:
            raise ValueError(
                f"backward_override must be one of {allowed}; "
                f"got {self.config.backward_override!r}"
            )
        # Fail fast at converter construction rather than deep inside model
        # build: an actual override needs a torchao whose MXFP8TrainingOpConfig
        # carries the knob ("quantized" normalizes to the stock backward and
        # works on any torchao).
        if self.config.backward_override not in (None, "quantized"):
            from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig

            if "backward_override" not in {
                f.name for f in fields(MXFP8TrainingOpConfig)
            }:
                raise ValueError(
                    "The installed torchao does not support backward_override "
                    "for MXFP8 grouped GEMMs. Install a torchao build whose "
                    "MXFP8TrainingOpConfig has the backward_override field."
                )

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            if fqns and not any(target_fqn in fqn for target_fqn in fqns):
                continue
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
                backward_override=self.config.backward_override,
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
