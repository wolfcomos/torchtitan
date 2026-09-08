# Why the NVFP4 (4over6, deq-bwd) reward curve did not track BF16 — analysis of steps 1-300 (2026-09-08)

Runs: `arm_bf16_4x4_long` vs `arm_nvfp4_deq_4x4_long` (Qwen3-30B-A3B, GRPO/DAPO-math, 8 prompts x 8 samples per step, same seeded
prompt stream -> identical prompts at every step, 4+4 GB200 cross-node, warmup 2 then constant lr 1e-6). Five independent evidence
angles (agent_reports.json; scripts/data in analysis_*/). All numbers below were computed from tfevents, rollout_samples.jsonl,
structured logs, the parked step-300 DCP checkpoints and the code checkouts the runs used.

## 1. Neither run trains most of its weights (shared root cause, decisive)
- Both arms train in pure bf16: `TrainingConfig(dtype="bfloat16")` in the shared RL base config; the RL configs docstring says all
  parameters, gradients and optimizer states are bf16 with no fp32 copy. Independent check: the step-300 checkpoint is 183.2 GB for
  30.5 B parameters = 6.01 bytes/param = bf16 weight + bf16 exp_avg + bf16 exp_avg_sq (fp32 masters would be >= 12 B/param).
- At lr 1e-6 an Adam-normalised update is <= ~1e-6, below half a bf16 ulp for every element with |w| >= 2^-12 = 2.4e-4 (median expert
  |w| = 0.015, half-ulp 6e-5). Round-to-nearest discards it every step, with no residual.
- Measured on the parked step-300 checkpoints vs the HF init (layers 0/20/39/44 experts w1/w2/w3, attention, router, norms, embeddings,
  lm_head): > 99.2% of elements bitwise identical in every matrix in BOTH runs, cosine(w0, w300) = 1.000000, relative Frobenius drift
  5e-5 .. 2.6e-4; no element with |w0| >= 2^-11 changed anywhere; Adam moments are 100% non-zero (gradients arrived and were rounded away).
  The only learning channel in either arm is the ~0.7-1.3% of near-zero elements.

## 2. The NVFP4 arm's expert forward is frozen at the init weights (NVFP4-specific)
- The 4over6 forward quantises expert weights per expert (global amax) in 1x16 E2M1 blocks with E4M3 scales; the zero bin is
  |w| < block_amax/24 ~ 2e-3. 100% of the movable (tiny) elements quantise to code 0, so their drift is invisible.
- quant(w300) vs quant(w0) with the run's quantizer settings: < 0.05% of fp4 codes changed (one probe: 12 of 12.6 M), no expert amax
  changed, identical 8.69% rel-RMS quantisation error. Layers 0-39 experts (79% of parameters, ~45% of active FLOPs/token) therefore
  contributed nothing to learning in the NVFP4 arm, for the trainer forward AND for vLLM (same converted model_spec, dynamic
  requantisation every forward, bf16 weights synced uncast).
- The BF16 arm's 0.70% moved expert elements (~170 M) do enter its bf16 GEMMs. Changed elements visible to the forward: BF16 ~206 M vs
  NVFP4 ~34 M (attention, router, embeddings, tail-layer experts only; each class also 4-15% smaller than BF16's).
- Correction of the naive "grid coarseness" story: with fp32 masters, 300 consistent steps of 1e-6 (3e-4 drift) would flip ~12% of fp4
  codes, so the fp4 forward would have moved. The freeze is the bf16-master dead zone (both arms) plus the fp4 zero bin (NVFP4).

## 3. Anatomy of the reward gap (paired on identical prompts)
- Paired same-prompt reward, BF16 vs NVFP4, 50-step blocks: 0.115/0.076, 0.129/0.078, 0.168/0.092, 0.201/0.096, 0.230/0.086,
  0.273/0.098 (diff -0.038 -> -0.175, bootstrap CIs exclude 0 from block 1; Wilcoxon z = -22.6 over 1-300).
- Small inference-side penalty before any learning: at step 1 (untouched HF weights) four independent NVFP4-deq replicates scored
  2/4/3/2 of 64 vs BF16 6/64 and 5/64 (pooled 0.043 vs 0.086, Fisher p = 0.073); over steps 1-10 all four NVFP4 replicates lie below
  both BF16 replicates (ratio 0.72); steps 11-30 paired diff -0.05 (p < 0.001) while cumulative weight drift is <= 2e-5 on < 1% of
  elements. Conditional on prompt difficulty NVFP4 solves ~0.6x as often as BF16 in steps 1-100 (weight quantisation error 8.7%
  rel-RMS + row-scaled activation quantisation, on every forward of trainer and generator).
- Learning gap on two axes. BF16 improves math accuracy (right-value rate 0.51 -> 0.57) AND answer-format compliance
  (P(reward | right value) 0.55 -> 0.80; \boxed rate among unrewarded groups 7% -> 23%); NVFP4 improves neither (0.34 -> 0.33;
  0.53 -> 0.50; 7% -> 11%). The two factors contribute about equally to the 2.8x gap at 251-300 (0.570 x 0.795 = 0.45 vs
  0.330 x 0.498 = 0.16). Format matters because the rubric extracts only LaTeX/\boxed answers (math_verify,
  try_extract_without_anchor=False): 18% of NVFP4's failures on BF16-solved prompts end with the RIGHT value in a plain "Answer: N".
- Amplifier: all-zero groups (no advantage) 70% vs 53% over 1-300 and 66% vs 42% by 251-300; 18 NVFP4 steps with zero gradient vs 2.
  Grad norm 0.12 vs 0.17 and 20-40% smaller Adam moments follow from that. Downstream of the accuracy gap, but it compounds.
- Trainer/generator mismatch: log-ratio mean -0.0029 vs -0.0007 (4.3x), 2.05% vs 0.33% of tokens clipped (zero gradient), max
  spikes up to 36. Present from step 1, flat over 300 steps, identical in the 08-25 code (-0.0028 / 2.0%), strictly on-policy
  (policy_age 0) so purely numerical; per-step uncorrelated with reward. Contribution undecided.

## 4. Refuted
Off-policy staleness (policy_age 0 in both), different prompts (bit-identical prompt_length per step), resume/handoff artefacts (11
handoffs within the sampling-noise floor), generation collapse (truncation 0.336 vs 0.339, repetition 0.5%, entropy 0.28 both,
length within 1%), optimizer differences (only weight decay 0 vs 0.1 on routed experts; negligible at lr*wd = 1e-7/step), node/hardware
placement, and "something changed since the 08-25 ablation": the 10-step ablation already had NVFP4-deq lowest of six arms (0.064 vs
0.098, -35%) with the same mismatch magnitudes; its write-up called that "no precision ordering" for lack of horizon.

## 5. What to do
1. Fix the recipe before drawing precision conclusions: fp32 master weights (fp32 params with bf16 compute, or an optimizer with an
   fp32 master copy / Kahan summation / stochastic-rounding bf16 updates), and/or a larger lr. Verify with the drift probe
   (analysis_checkpoint-drift-experiment/drift_probe.py) after ~20 steps: the changed-element fraction should be far above 1%.
2. Separate inference penalty from training penalty: NVFP4 rollouts + BF16 trainer and BF16 rollouts + NVFP4 trainer; or an offline
   eval of one checkpoint under BF16 vs NVFP4 inference on the same prompts.
3. Reward extraction: allow plain "Answer: N" (ExprExtractionConfig / try_extract_without_anchor) or shape the prompt to require
   \boxed; this removes a format-learning confound and cuts zero-signal groups in both arms.
4. Optional: drop_zero_std_reward_groups=True to stop normalising the loss over zero-advantage tokens.
