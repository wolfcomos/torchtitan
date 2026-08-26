"""Six-arm Qwen3-30B-A3B recipe-ablation configs (TitanRL + torchao).

Untracked launcher module (selected via ``--module ablation_arms``),
replicating the six-configuration recipe ablation from
lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl at reduced scale on 2x4 GB200:

1. arm_bf16       — BF16 training + BF16 rollout.
2. arm_mxfp8_e2e  — end-to-end MXFP8 experts training + MXFP8 rollout.
3. arm_mxfp8_hp   — MXFP8 rollout/forward, high-precision backward.
4. arm_mxfp8_deq  — MXFP8 rollout/forward, dequantized backward.
5. arm_nvfp4_hp   — per-token NVFP4 (4over6 row-scaled) rollout/forward,
                    high-precision backward.
6. arm_nvfp4_deq  — same forward, dequantized backward.

Shared workload across all arms (the fixed-comparison contract): GRPO on
dapo-math-17k, 8 prompts x 8 samples per step, 10 synchronous steps,
max 2048 response tokens (blog: 8192 — scaled down to fit the window),
AIME2025 greedy validation (16 samples) at start/end, AdamW lr 1e-6
warmup 2 + linear decay, seeds pinned by the launch script.

Low-precision arms additionally share (per the blog's recipe contract):
routed-expert grouped GEMMs only, last 8 of 48 layers kept BF16, and
weight decay 0 on the quantized expert weights (0.1 elsewhere).
"""

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.quantization import (
    MXFP8GroupedExpertsConverter,
    NVFP4FourOverSixGroupedExpertsConverter,
)
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_first_last_fqns
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import Controller, ValidationConfig
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_30b_a3b_varlen,
)
from torchtitan.experiments.rl.examples.dapo_math.data import AIME2025Dataset
from torchtitan.experiments.rl.examples.dapo_math.rollouter import DapoMathRollouter
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.models.qwen3 import model_registry

_N_LAYERS = 48  # Qwen3-30B-A3B decoder layers
_LAST_BF16 = 8  # blog: last 15% of layers kept BF16 -> 8 of 48
_MAX_RESPONSE_TOKENS = 2048
_MAX_TOTAL_TOKENS = 4096
_NUM_VALIDATION_SAMPLES = 16

# Allow-list of quantized layers (trailing dot so "layers.1." won't match
# "layers.10"); shared by the converters of every low-precision arm.
_QUANTIZED_LAYER_FQNS = nvfp4_bf16_first_last_fqns(_N_LAYERS, 0, _LAST_BF16)

# Same layer window as a param-FQN regex: layers 0-39 routed-expert weights
# (w1_EFD/w2_EDF/w3_EFD live at layers.N.moe.routed_experts.inner_experts,
# verified via a meta-device build of the 30B-A3B spec). ParamGroupConfig
# uses re.search on RAW named_parameters() names, which may carry wrapper
# segments (e.g. _checkpoint_wrapped_module), so allow anything between the
# layer index and routed_experts; router.gate stays in the default group.
_QUANTIZED_EXPERT_PARAM_REGEX = r"layers\.([0-9]|[1-3][0-9])\..*routed_experts\."

_ADAMW_COMMON = {"lr": 1e-6, "betas": (0.9, 0.95), "eps": 1e-8}


def _lp_optimizer() -> OptimizersContainer.Config:
    """AdamW with weight decay 0 on quantized expert weights, 0.1 elsewhere."""
    return OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=_QUANTIZED_EXPERT_PARAM_REGEX,
                optimizer_name="AdamW",
                optimizer_kwargs={**_ADAMW_COMMON, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**_ADAMW_COMMON, "weight_decay": 0.1},
            ),
        ]
    )


def _ablation_base(converters: list | None) -> Controller.Config:
    """Shared workload; arms differ ONLY in converters (+ LP weight decay)."""
    config = rl_grpo_qwen3_30b_a3b_varlen()
    config.model_spec = model_registry(
        "30B-A3B",
        attn_backend="varlen",
        converters=converters or [],
    )
    config.rollouter = DapoMathRollouter.Config(
        validation_dataset=AIME2025Dataset.Config(
            num_samples=_NUM_VALIDATION_SAMPLES,
        ),
        token_env=TokenEnv.Config(
            max_rollout_tokens=_MAX_TOTAL_TOKENS,
            max_num_turns=1,
        ),
        advantage=AdvantageEstimator.Config(should_std_normalize=True),
    )
    config.async_loop.num_training_steps = 10
    config.async_loop.num_prompts_per_train_step = 8
    config.async_loop.num_samples_per_prompt = 8
    config.async_loop.target_offpolicy_steps = 0
    config.async_loop.window_fraction = None
    # Keep zero-variance groups (their advantages are 0 via the +eps guard):
    # DAPO-style dynamic resampling would otherwise make per-step rollout
    # counts — and arm wall time — unpredictable across the six arms.
    config.async_loop.training_sample_builder.drop_zero_std_reward_groups = False
    config.async_loop.validation = ValidationConfig(
        num_samples=_NUM_VALIDATION_SAMPLES,
    )
    config.async_loop.batcher = Batcher.Config(
        batch=BatchConfig(local_batch_size=1, seq_len=_MAX_TOTAL_TOKENS),
    )
    config.generator.sampling.max_tokens = _MAX_RESPONSE_TOKENS
    config.trainer.lr_scheduler.total_steps = 10
    # Save updated weights at the final step for follow-up runs, but weights
    # only: the base config's full-state save (optimizer moments included)
    # would not fit 6 arms on this scratch volume.
    config.trainer.checkpoint.last_save_model_only = True
    if converters:
        config.trainer.optimizer = _lp_optimizer()
    return config


def _nvfp4_converter(backward_override: str):
    return NVFP4FourOverSixGroupedExpertsConverter.Config(
        fqns=_QUANTIZED_LAYER_FQNS,
        row_scaled_activation=True,
        err_mode="mse",
        e4m3_scale_bound=256,
        weight_block="1x16",
        backward_override=backward_override,
        pad_multiple=128,
    )


def _mxfp8_converter(backward_override: str | None):
    return MXFP8GroupedExpertsConverter.Config(
        fqns=_QUANTIZED_LAYER_FQNS,
        recipe_name="mxfp8_rceil",
        backward_override=backward_override,
        pad_multiple=128,
    )


def arm_bf16() -> Controller.Config:
    return _ablation_base(None)


def arm_mxfp8_e2e() -> Controller.Config:
    return _ablation_base([_mxfp8_converter(None)])


def arm_mxfp8_hp() -> Controller.Config:
    return _ablation_base([_mxfp8_converter("high_precision")])


def arm_mxfp8_deq() -> Controller.Config:
    return _ablation_base([_mxfp8_converter("dequantized")])


def arm_nvfp4_hp() -> Controller.Config:
    return _ablation_base([_nvfp4_converter("high_precision")])


def arm_nvfp4_deq() -> Controller.Config:
    return _ablation_base([_nvfp4_converter("dequantized")])
