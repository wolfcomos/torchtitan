# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
    NVFP4GroupedExpertsConverter,
)
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_first_last_fqns
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.trainer import Trainer

from . import model_registry


def enable_fused_swiglu(config: Trainer.Config) -> None:
    # fused_swiglu.py registers two overrides (dense FeedForward + MoE grouped
    # experts); activate both by naming each factory.
    for override in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert override not in config.override.imports
        config.override.imports.append(override)


def deepseek_v3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mtp() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", num_mtp_layers=1)
    config.loss = MTPLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    return config


def deepseek_v3_debugmodel_mxfp8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def _deepseek_v3_debugmodel_nvfp4_four_over_six(
    backward_override: str, fqns: list[str] | None = None
) -> Trainer.Config:
    # Quantize only the routed-expert grouped GEMMs with four-over-six NVFP4,
    # mirroring the miles NVFP4 RL recipes: row-scaled activations, MSE
    # candidate selection, E4M3 bound 256, and 1x16 weight blocks
    # (NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1). With no Linear converter,
    # attention, the dense-layer feed-forward, shared experts, the MoE router
    # gate, embeddings, and the lm_head all stay bf16 -- the recipe's
    # experts-only allow-list.
    config = deepseek_v3_debugmodel()
    # The quantized grouped GEMM reads the expert group offsets on the host
    # (and the row-scaled path loops dense GEMMs per group), which CUDA-graph
    # capture forbids; miles likewise runs its quantized recipes with CUDA
    # graphs off.
    config.training.disable_cuda_graphs = True
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns or [],
                row_scaled_activation=True,
                err_mode="mse",
                e4m3_scale_bound=256,
                weight_block="1x16",
                backward_override=backward_override,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_nvfp4_four_over_six() -> Trainer.Config:
    # The miles NVFP4 RL base recipe point: high-precision backward
    # (NVTE_BACKWARD_OVERRIDE=high_precision) on every routed-expert layer.
    return _deepseek_v3_debugmodel_nvfp4_four_over_six("high_precision")


def deepseek_v3_debugmodel_nvfp4_four_over_six_dequantized() -> Trainer.Config:
    # The advanced miles recipe variant (the GLM-5.2 NVFP4 e2e analog): the
    # dequantized backward (NVTE_BACKWARD_OVERRIDE=dequantized) backpropagates
    # through bf16 GEMMs on the dequantized fprop operands, and the first and
    # last decoder layers stay bf16 (--first-last-layers-bf16 with one layer
    # at each end). The debugmodel's layer 0 is dense (no routed experts), so
    # the leading-layer exclusion is vacuous there but keeps the recipe shape.
    # The debugmodel has 6 layers; layers 1..4 quantize.
    return _deepseek_v3_debugmodel_nvfp4_four_over_six(
        "dequantized", fqns=nvfp4_bf16_first_last_fqns(6, 1, 1)
    )


def deepseek_v3_debugmodel_hybridep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_debugmodel_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    model_spec = model_registry("16B", attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_16b_hybridep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_16b_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_671b() -> Trainer.Config:
    model_spec = model_registry(
        "671B",
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b_float8() -> Trainer.Config:
    config = deepseek_v3_671b()
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=model_compile_enabled,
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled
            ),
        ],
    )
    return config
