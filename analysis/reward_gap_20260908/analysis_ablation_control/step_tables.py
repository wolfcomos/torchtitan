import json, numpy as np
OUT="/tmp/claude-147915/-home-hanlinb/87c5c385-2451-4aa9-92a6-6e561d466781/scratchpad/analysis_ablation_control"
D=json.load(open(f"{OUT}/scalars.json"))
runs=["arm_bf16","arm_nvfp4_deq","arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]
short={"arm_bf16":"ABL_bf16","arm_nvfp4_deq":"ABL_nvfp4","arm_bf16_4x4_long":"LONG_bf16","arm_nvfp4_deq_4x4_long":"LONG_nvfp4"}
tags=["trainer/lr","rollout_reward/_mean","rollout_reward/group_zero_std_frac/mean","loss/ratio_clipped_frac","bit_wise/logprob_diff/mean","bit_wise/logprob_diff/max","trainer/grad_norm/mean","trainer/entropy/mean","rollout/response_length/mean","rollout/truncation_rate/mean","loss/mean","train_batch/num_training_samples","trainer/policy_version","validation_reward/_mean","rollout/prompt_length/mean"]
def g(r,t,s):
    v=D[r].get(t,{}).get(str(s)); return v
for t in tags:
    print(f"\n### {t}")
    print("step | "+" | ".join(short[r] for r in runs))
    for s in range(0,11):
        row=[]
        for r in runs:
            v=g(r,t,s); row.append("-" if v is None else f"{v:.5g}")
        print(f"{s:4d} | "+" | ".join(row))
# block means over steps 1-10 and paired differences
print("\n### means over steps 1-10")
for t in tags:
    row=[]
    for r in runs:
        vals=[g(r,t,s) for s in range(1,11)]; vals=[v for v in vals if v is not None]
        row.append(f"{np.mean(vals):.5g}" if vals else "-")
    print(f"{t:45s} "+" | ".join(row))
# reward blocks of the long runs early
print("\n### long-run reward block means")
for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,50),(1,30),(31,60),(61,100),(1,100)]:
    row=[]
    for r in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
        vals=[g(r,"rollout_reward/_mean",s) for s in range(lo,hi+1)]
        row.append(f"{np.mean(vals):.4f}")
    print(f"{lo}-{hi}: BF16 {row[0]} NVFP4 {row[1]}")
# paired per-step diffs steps 1-10
print("\n### per-step reward diff (bf16-nvfp4) ablation vs long, steps 1-10")
da=[g("arm_bf16","rollout_reward/_mean",s)-g("arm_nvfp4_deq","rollout_reward/_mean",s) for s in range(1,11)]
dl=[g("arm_bf16_4x4_long","rollout_reward/_mean",s)-g("arm_nvfp4_deq_4x4_long","rollout_reward/_mean",s) for s in range(1,11)]
print("ablation diffs:",[f"{x:+.4f}" for x in da],"mean",f"{np.mean(da):+.4f}","sd",f"{np.std(da,ddof=1):.4f}")
print("long diffs    :",[f"{x:+.4f}" for x in dl],"mean",f"{np.mean(dl):+.4f}","sd",f"{np.std(dl,ddof=1):.4f}")
# same-arm cross-run comparison (ablation vs long) steps 1-10
print("\n### same arm: ablation vs long, reward per step")
for a,b in [("arm_bf16","arm_bf16_4x4_long"),("arm_nvfp4_deq","arm_nvfp4_deq_4x4_long")]:
    d=[g(a,"rollout_reward/_mean",s)-g(b,"rollout_reward/_mean",s) for s in range(1,11)]
    print(a, "minus long:", [f"{x:+.4f}" for x in d], "mean", f"{np.mean(d):+.4f}")
    same=sum(1 for s in range(1,11) if abs(g(a,"rollout_reward/_mean",s)-g(b,"rollout_reward/_mean",s))<1e-9)
    print("  identical steps:",same)
