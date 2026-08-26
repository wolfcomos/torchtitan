#!/bin/bash
# Launch one Qwen3-30B-A3B recipe-ablation arm in the vLLM 26.08 container.
# Usage: run_ablation_arm.sh <arm_name> [timeout_s]
# <arm_name> must be a config in the ablation_arms module (arm_bf16,
# arm_mxfp8_e2e, arm_mxfp8_hp, arm_mxfp8_deq, arm_nvfp4_hp, arm_nvfp4_deq).
# Workload knobs (dataset, batch, sampling, steps, optimizer) live in
# ablation_arms.py so every arm shares them; this script sets only
# topology (trainer tp2/ep2 GPUs 0,1 + one generator tp2/ep2 GPUs 2,3),
# paths, seeds, and metrics sinks.
set -eu
ARM=${1:?arm name}
TIMEOUT=${2:-7200}

IMAGE=gitlab-master.nvidia.com:5005/dl/dgx/vllm:26.08-py3.63853605-devel-arm64
SCRATCH=/home/scratch.hanlinb_ent_1
ABL=$SCRATCH/agent_scratch/nvfp4_4over6/ablation
RL=$SCRATCH/agent_scratch/nvfp4_4over6/rl_smoke
CKPT=/scratch/agent_scratch/nvfp4_4over6/miles_smoke/models/Qwen3-30B-A3B
DUMP=/scratch/agent_scratch/nvfp4_4over6/ablation/outputs/$ARM
PYPATH=/scratch/repos/titan-nvfp4:/scratch/repos/ao-nvfp4:/scratch/agent_scratch/nvfp4_4over6/rl_smoke/pydeps_rl:/scratch/agent_scratch/nvfp4_4over6/pydeps

if [ -d "$ABL/outputs/$ARM" ]; then
  echo "REFUSING: $ABL/outputs/$ARM exists (reuse would resume from its checkpoint, not HF init)"; exit 2
fi
mkdir -p $ABL/outputs/$ARM $ABL/logs $RL/home/.cache/cute_dsl

# In-container CUDA preflight: the 2026-08-25 arm_bf16 relaunch died with
# "No CUDA GPUs are available" inside a fresh container while host
# nvidia-smi showed 4 idle GPUs, so verify CUDA init before the real run.
for attempt in 1 2 3; do
  if docker run --rm --gpus all \
      $(for c in /dev/nvidia-caps-imex-channels/channel*; do printf -- "--device=%s " $c; done) \
      -u $(id -u):$(id -g) $IMAGE \
      python3 -c "import torch; n=torch.cuda.device_count(); assert n==4, n; torch.zeros(1, device='cuda'); print('CUDA preflight OK: 4 GPUs')"; then
    PREFLIGHT_OK=1; break
  fi
  echo "CUDA preflight attempt $attempt failed; retrying in 30s"; sleep 30
done
[ "${PREFLIGHT_OK:-0}" = 1 ] || { echo "ABORT: CUDA preflight failed 3x on $(hostname)"; exit 3; }

# Host-side GPU memory sampler (the RL loop has no peak-memory metric).
# Timestamped so a relaunch cannot overwrite an earlier attempt's samples.
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 \
  > $ABL/logs/${ARM}_gpumem_$(date +%Y%m%d-%H%M%S).csv 2>/dev/null &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT

timeout $TIMEOUT docker run --rm --name abl_$ARM --gpus all --ipc=host --network host \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  $(for c in /dev/nvidia-caps-imex-channels/channel*; do printf -- "--device=%s " $c; done) \
  -u $(id -u):$(id -g) \
  -v $SCRATCH:/scratch \
  -v $RL/fa_overlay/cute:/usr/local/lib/python3.12/dist-packages/flash_attn/cute:ro \
  -v $RL/fa_overlay/_fa4.py:/usr/local/lib/python3.12/dist-packages/torch/nn/attention/_fa4.py:ro \
  -v $RL/fa_overlay/nvidia_cutlass_dsl_462:/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl:ro \
  -w /scratch/repos/titan-nvfp4 \
  -e HOME=/scratch/agent_scratch/nvfp4_4over6/rl_smoke/home \
  -e CUTE_DSL_CACHE_DIR=/scratch/agent_scratch/nvfp4_4over6/rl_smoke/home/.cache/cute_dsl \
  -e PYTHONPATH=$PYPATH \
  -e HF_HOME=/scratch/agent_scratch/nvfp4_4over6/rl_smoke/hf_home \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FOUR_OVER_SIX_GROUPED_ROW_SCALED_FUSED_BF16_OUT=1 \
  `# fused NVFP4 row-scaled fwd (loop path: 41 tok/s gen, validate 2908s vs 323s bf16); read only by the 4over6 grouped op` \
  -e WANDB_MODE=offline \
  $IMAGE bash -c "python3 -m torchtitan.experiments.rl.train \
    --module ablation_arms --config $ARM \
    --hf_assets_path $CKPT \
    --dump-folder $DUMP \
    --trainer.parallelism.data-parallel-shard-degree 1 \
    --trainer.parallelism.data-parallel-replicate-degree 1 \
    --trainer.parallelism.tensor-parallel-degree 2 \
    --trainer.parallelism.expert-parallel-degree 2 \
    --generator.parallelism.data-parallel-degree 1 \
    --generator.parallelism.tensor-parallel-degree 2 \
    --generator.parallelism.expert-parallel-degree 2 \
    --num_generators 1 \
    --trainer.debug.seed 42 \
    --generator.debug.seed 42 \
    --metrics.no-enable-wandb \
    --metrics.enable-tensorboard"
