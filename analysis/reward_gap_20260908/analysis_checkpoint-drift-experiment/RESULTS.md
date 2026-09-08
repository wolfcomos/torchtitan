# Checkpoint drift probe: HF base (w0) vs step-300 DCP weights, BF16 run vs NVFP4 run

Scripts/outputs in this directory: dump_metadata.py -> metadata_<run>.json; drift_probe.py -> drift_results.json, drift_probe_full.log;
overlap_probe.py -> overlap_results.json, overlap_probe.log; early_reward.py -> early_reward.json.
Container: gitlab-master.nvidia.com:5005/dl/dgx/pytorch:jbernloehr--26.08-devel-py313 (torch 2.14.0a0, safetensors 0.8.0), 1 GPU, inputs mounted :ro.
DCP load: torch.distributed.checkpoint.load(state_dict={key: torch.empty(shape,dtype)}, checkpoint_id=dir), no process group (works; 0.1-2 s/tensor).

## Name mapping (titan-nvfp4/torchtitan/models/qwen3/state_dict_adapter.py:33-50)
HF model.layers.L.mlp.experts.e.gate_proj.weight [768,2048] stacked over e -> layers.L.moe.routed_experts.inner_experts.w1_EFD [128,768,2048]
up_proj -> w3_EFD; down_proj [2048,768] -> w2_EDF [128,2048,768]; self_attn.o_proj -> layers.L.attention.wo.weight; q_proj -> attention.qkv_linear.wq.weight;
mlp.gate.weight -> layers.L.moe.router.gate.weight; input_layernorm -> attention_norm; embed_tokens -> tok_embeddings; lm_head -> lm_head.
Mapping check: BF16 run w300 vs w0 cosine = 1.000000 and >= 99.2% of elements bitwise identical for every tensor (a wrong mapping would give ~0 cosine).

## Checkpoint dtype (metadata_*.json)  -- REFUTES the task's "fp32 model + optimizer" premise
All 1545 model params AND AdamW exp_avg/exp_avg_sq are torch.bfloat16 in BOTH runs; only *.step and layers.N.moe.expert_bias_E are fp32.
Cause: rl_grpo_qwen3_30b_a3b_varlen sets training=TrainingConfig(dtype="bfloat16") (titan-nvfp4/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py:759),
inherited by both arms via _ablation_base (ablation_arms.py:84); configs.py:87-91 documents dtype=bfloat16 as "all parameters, gradients, and optimizer states in
bfloat16, without an extra copy of fp32 weights"; model built under set_default_dtype(bf16) at experiments/rl/actors/trainer.py:298.
(controller.py:386-393 and config_registry.py:862 comments claim "fp32 master weights" -- that is only true for the default dtype="float32", not this recipe.)

## Drift w0 -> w300 (drift_results.json; layers 0,20,39 = NVFP4-quantized in the NVFP4 arm, layer 44 = bf16 in both arms)
tensor                      | BF16 run: frac changed / rel-Fro / abs max | NVFP4 run: frac changed / rel-Fro / abs max
L0  experts w1/w2/w3        | 0.75% 0.71% 0.78% / 1.2e-4 1.1e-4 1.3e-4 / 2.5e-4 | 0.67% 0.64% 0.70% / 9.0e-5 8.3e-5 9.6e-5 / 2.0-2.2e-4
L20 experts w1/w2/w3        | 0.67% 0.68% 0.69% / 1.0e-4 1.1e-4 1.1e-4 / 2.6e-4 | 0.63% 0.64% 0.65% / 7.6e-5 8.0e-5 8.4e-5 / 2.3-2.4e-4
L39 experts w1/w2/w3        | 0.63% 0.64% 0.63% / 8.4e-5 8.8e-5 8.5e-5 / 2.5e-4 | 0.59% 0.60% 0.58% / 6.1e-5 6.5e-5 6.2e-5 / 2.1-2.3e-4
L44 experts w1/w2/w3 (bf16) | 0.51% 0.51% 0.48% / 6.2e-5 6.4e-5 5.8e-5 / 2.5e-4 | 0.48% 0.47% 0.44% / 4.9e-5 4.9e-5 4.5e-5 / 2.1-2.3e-4
attention.wo L0/20/39/44    | 1.25% 1.06% 0.93% 0.96% / 2.1e-4 1.3e-4 1.1e-4 1.0e-4 | 1.22% 1.00% 0.88% 0.91% / 1.9e-4 1.1e-4 7.7e-5 8.4e-5
router.gate L0/20/39/44     | 0.41% 1.31% 2.00% 2.29% / 2.7e-5 1.2e-4 2.0e-4 2.6e-4 | 0.39% 1.25% 1.95% 2.21% / 2.6e-5 9.1e-5 1.6e-4 2.3e-4
lm_head / tok_embeddings    | 0.058% / 8e-6 ; 0.052% / 1.3e-5                       | 0.049% / 7e-6 ; 0.044% / 1.1e-5
RMSNorm weights (|w|~1)     | 0 elements changed (L20 attn/ffn norm, L39 ffn, L0 q_norm, final norm); L0 attention_norm (|w|~1e-3) 14.0% changed | same: 0 / 13.4%
Cosine(w0,w300) = 1.000000 everywhere. Relative drift p99 = 0 for every expert tensor (>99% of elements bitwise unchanged).
bf16 dead zone: NO element with |w0| >= 2^-11 = 4.88e-4 changed in ANY tensor of EITHER run (w0_abs_max_changed = 0.000488 in all 11 tensors probed in overlap_results.json);
changed elements have |w0| p50 ~1.0e-4, p99 ~2.4e-4 (experts). Median expert |w| = 0.0156 (bf16 ulp 1.22e-4) vs nominal AdamW step lr*m/(sqrt(v)+eps): p50 1.2e-7, p99 5.9e-7, max 9.5e-7
(L20 w2_EDF moments at step 300). Only 0.18% (BF16 run) / 0.17% (NVFP4 run) of L20 w2 elements have a nominal update >= half an ulp of their bf16 value.
Max abs drift 2.2-2.6e-4 ~= 300 steps x lr 1e-6 (an AdamW step is ~lr) -> the few movable (near-zero) weights moved by the full budget; everything else is frozen by RTNE.

## NVFP4 4over6 view of the expert drift (real quantizer: ao-nvfp4 four_over_six.py::four_over_six_quantize pure-torch body, cutedsl path disabled,
## via four_over_six_grouped._quantize_expert_weights, arm settings weight_block=1x16, err_mode=mse, e4m3_scale_bound=256 from ablation_arms.py:126-135;
## per-expert amax = weight.abs().amax(dim=(1,2)) as in four_over_six_grouped.py:466-474)
For all 12 expert tensors x both runs: FP4 codes changed 0.017%-0.043% of elements; E4M3 block scales changed 0.8e-7 to 8e-7 (1-10 blocks of 12.6M);
per-expert global amax changed in 0/128 experts everywhere; weight quantization error rel-RMS = 0.0869 (0.0861 for L0 w2) at BOTH w0 and w300 to 4 digits.
=> quant(w300) is 99.96-99.98% code-identical to quant(w0) for the BF16 run's weights AND the NVFP4 run's weights. The drift (rel-Fro ~1e-4) is ~3 orders of
magnitude below the per-forward NVFP4 quantization noise (rel-RMS 8.7e-2) and below the FP4 grid step (block_amax/6 ~ 1e-2..1e-3), so in the NVFP4 arm the
expert weights the forward/rollout actually consume are effectively the step-0 weights for all 300 steps.

## Cross-run comparison (overlap_results.json)
NVFP4-run drift is systematically smaller than BF16-run drift: rel-Fro ratio 0.74-0.80 on experts (e.g. L20 w2 8.0e-5 vs 1.05e-4), 6-8% fewer changed elements;
exp_avg |mean| 3.04e-8 vs 3.92e-8 (-22%), exp_avg_sq mean 7.1e-14 vs 1.07e-13 (-34%) -- consistent with the logged grad-norm gap 0.12 vs 0.17.
The two runs changed largely the SAME elements (Jaccard of changed sets 0.85 experts, 0.80 wo, 0.80 router; eligibility is set by |w0|, not by the gradient),
but with different values: identical bf16 value on only 3.3-3.5% of commonly-changed elements, sign agreement 58-59% (experts, router) / 71% (attention),
cosine(drift_bf16, drift_nvfp4) = 0.66-0.72 (experts/attn/router), 0.17 lm_head, 0.40 embeddings. |w300_bf16 - w300_nvfp4| rel-Fro 7-9e-5 ~ same size as either drift.

## Reward gap timing (early_reward.json, tfevents, later-file-wins)
rollout_reward/_mean steps 1-5: 0.069 vs 0.034; 1-10: 0.081 vs 0.067; 1-20: 0.091 vs 0.059; 1-50: 0.115 vs 0.076; 1-100: 0.122 vs 0.077 (BF16 vs NVFP4).
zero-std group frac 1-20: 0.675 vs 0.7625. The gap is present in the first 20 steps, when cumulative drift is <= 20 x lr = 2e-5 on the <1% eligible elements.
