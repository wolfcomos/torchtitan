import pickle, numpy as np, csv
out=pickle.load(open("data.pkl","rb"))
B="arm_bf16_4x4_long"; N="arm_nvfp4_deq_4x4_long"
tags=sorted(out[B]['data'])
# per-step wide CSV (later file wins)
with open("per_step_all_tags.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["step"]+[f"bf16|{t}" for t in tags]+[f"nvfp4|{t}" for t in tags])
    for s in range(1,301):
        w.writerow([s]+[out[B]['data'][t].get(s,(float('nan'),))[0] for t in tags]+[out[N]['data'][t].get(s,(float('nan'),))[0] for t in tags])
# flagged tags with classification
cls={
 'rollout_reward/_mean':('CONSEQUENCE/OUTCOME','the target quantity; NVFP4 below BF16 from steps 1-20 (0.059 vs 0.091) and flat (slope +0.0076/100 steps, t=1.8) vs BF16 +0.063/100 (t=9.7)'),
 'rollout_reward/_sum':('CONSEQUENCE','= 64*mean, identical information'),
 'rollout_reward/component/RewardMathVerify/mean':('CONSEQUENCE','identical to _mean (single reward component)'),
 'rollout_reward/_std':('CONSEQUENCE','binary reward: std = sqrt(p(1-p)), follows mean'),
 'rollout_reward/group_zero_std_frac/mean':('MECHANISM (amplifier) + CONSEQUENCE','0.70 vs 0.49 in 201-300: NVFP4 trains on 2.4 signal groups/step vs 4.1; itself a function of per-prompt solve rate; 18 NVFP4 steps had zero signal groups (grad_norm=0) vs 2 for BF16'),
 'advantage/_std':('CONSEQUENCE','std-normalised advantages are 0 on zero-std groups; fewer signal groups -> lower batch advantage std'),
 'advantage/_min':('CONSEQUENCE','same mechanism as advantage/_std (most-negative advantage occurs in groups with 1 wrong of many right, which NVFP4 rarely produces)'),
 'loss/mean':('CONSEQUENCE','sum of -ratio*adv over tokens; scales with number of signal groups and reward variance'),
 'trainer/grad_norm/mean':('CONSEQUENCE','r(grad_norm, zero_std)=-0.84/-0.86 within run: grad norm is set by how many groups carry signal'),
 'bit_wise/logprob_diff/mean':('CANDIDATE CAUSE (present from step 1, flat)','trainer-vs-generator k1 KL estimate -0.0029 vs -0.0007 nats/token; slope ~0 in both runs; NOT correlated with per-step reward within NVFP4 (r=-0.005)'),
 'bit_wise/logprob_diff/max':('CANDIDATE CAUSE (present from step 1, flat)','per-step worst token 3.7 vs 1.1 nats; peaks 36.0 (step 249), 15.1 (202); uncorrelated with reward (r=-0.005..0.04)'),
 'loss/ratio_clipped_frac':('CANDIDATE CAUSE (present from step 1, flat)','6.4x BF16; policy_age=0 so ALL clipping is trainer/generator numerical mismatch, not staleness; 2.05% of tokens get zero gradient'),
 'generator/decode_time_ms/mean':('UNRELATED to reward (perf)','NVFP4 rollout kernels 1.63x slower; same tokens generated'),
 'generator/decode_time_ms/max':('UNRELATED (perf)',''),
 'generator/inter_token_latency_ms/mean':('UNRELATED (perf)','1.64x'),
 'generator/inter_token_latency_ms/max':('UNRELATED (perf)',''),
 'generator/prefill_time_ms/mean':('UNRELATED (perf)','1.52x'),
 'generator/prefill_time_ms/max':('UNRELATED (perf)',''),
 'generator/time_to_first_token_ms/mean':('UNRELATED (perf)',''),
 'generator/time_to_first_token_ms/max':('UNRELATED (perf)',''),
 'generator/queue_time_ms/mean':('UNRELATED (perf, ms-scale noise)',''),
 'generator/queue_time_ms/max':('UNRELATED (perf, ms-scale noise)',''),
 'timing/step/forward_backward/mean':('UNRELATED (perf)','38 vs 23 s'),
 'timing/step/total/mean':('UNRELATED (perf)','642 vs 388 s; 94% of it is wait_for_training_batch = rollout generation'),
 'timing/step/wait_for_training_batch/mean':('UNRELATED (perf)','604 vs 364 s'),
 'perf/trainer/tokens_per_second_full_step':('UNRELATED (perf)',''),
 'perf/trainer/tokens_per_second_fwd_bwd':('UNRELATED (perf)',''),
 'perf/trainer/step_time_ratio/blocking_generator_pull_model_state_dict':('UNRELATED (1e-8 ratios, timing noise)',''),
 'perf/trainer/step_time_ratio/blocking_trainer_push_model_state_dict':('UNRELATED (1e-8 ratios, timing noise)',''),
 'perf/trainer/step_time_ratio/unaccounted':('UNRELATED (1e-6 ratios, timing noise)',''),
}
rows=list(csv.DictReader(open("all_tags_block_means.csv")))
with open("flagged_tags_gt15pct_201-300.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["tag","bf16_201-300","nvfp4_201-300","ratio_nvfp4_over_bf16","bf16_b1->b2->b3","nvfp4_b1->b2->b3","classification","note"])
    for r in rows:
        if abs(float(r['reldiff_b3']))>0.15:
            c,n=cls.get(r['tag'],('?',''))
            w.writerow([r['tag'],r['bf16_b3'],r['nvfp4_b3'],r['ratio_b3'],r['bf16_shape'],r['nvfp4_shape'],c,n])
print(open("flagged_tags_gt15pct_201-300.csv").read()); import os; print(os.listdir('.'))
