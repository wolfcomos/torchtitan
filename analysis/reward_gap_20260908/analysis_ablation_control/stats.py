import glob, os, json, numpy as np, math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
dirs=["arm_bf16","arm_bf16_4x4_long","arm_nvfp4_deq","arm_nvfp4_deq_4x4","arm_nvfp4_deq_4x4_100","arm_nvfp4_deq_4x4_long","arm_nvfp4_hp","arm_mxfp8_deq","arm_mxfp8_e2e","arm_mxfp8_hp","arm_nvfp4_deq_cache_smoke"]
PL={}; RW={}
for d in dirs:
    pl={}; rw={}
    for f in sorted(glob.glob(f"{BASE}/{d}/events.out.tfevents.*")):
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        T=ea.Tags()['scalars']
        if "rollout/prompt_length/mean" in T:
            for ev in ea.Scalars("rollout/prompt_length/mean"):
                if ev.step<=10: pl[ev.step]=ev.value
        if "rollout_reward/_mean" in T:
            for ev in ea.Scalars("rollout_reward/_mean"):
                if ev.step<=10: rw[ev.step]=ev.value
    PL[d]=pl; RW[d]=rw
ref=PL["arm_bf16"]
print("### prompt_length/mean steps 1-10 identical to arm_bf16?")
for d in dirs:
    same=[s for s in range(1,11) if s in PL[d] and abs(PL[d][s]-ref[s])<1e-6]
    print(f"{d:32s} identical steps: {len(same)}/{len([s for s in range(1,11) if s in PL[d]])}")
print("\n### step-1 reward (policy version 0 = untouched HF weights; rollout precision is the ONLY difference), counts of 64")
for d in dirs:
    if 1 in RW[d]: print(f"{d:32s} {RW[d][1]:.4f} = {round(RW[d][1]*64)}/64")
def hyper_p(a,n1,b,n2):
    # one-sided Fisher: P(X>=a) for X ~ Hypergeom(N=n1+n2, K=a+b, n=n1)
    N=n1+n2; K=a+b
    p=0.0
    for x in range(a, min(K,n1)+1):
        p+=math.comb(K,x)*math.comb(N-K,n1-x)/math.comb(N,n1)
    return p
bf=[RW["arm_bf16"][1],RW["arm_bf16_4x4_long"][1]]
nv=[RW[d][1] for d in ["arm_nvfp4_deq","arm_nvfp4_deq_4x4","arm_nvfp4_deq_4x4_100","arm_nvfp4_deq_4x4_long"]]
a=round(sum(bf)*64); b=round(sum(nv)*64)
print(f"pooled step-1: BF16 {a}/{64*len(bf)} = {a/(64*len(bf)):.4f}; NVFP4-deq {b}/{64*len(nv)} = {b/(64*len(nv)):.4f}; one-sided Fisher p = {hyper_p(a,64*len(bf),b,64*len(nv)):.3f}")
print("\n### steps 1-10 mean reward per run (all on identical prompts)")
m={d:np.mean([RW[d][s] for s in range(1,11)]) for d in dirs if all(s in RW[d] for s in range(1,11))}
for d,v in sorted(m.items(), key=lambda x:-x[1]): print(f"{d:32s} {v:.4f}")
bfm=[m["arm_bf16"],m["arm_bf16_4x4_long"]]; nvm=[m[d] for d in ["arm_nvfp4_deq","arm_nvfp4_deq_4x4","arm_nvfp4_deq_4x4_100","arm_nvfp4_deq_4x4_long"]]
print(f"BF16 replicates mean {np.mean(bfm):.4f} (min {min(bfm):.4f}); NVFP4-deq replicates mean {np.mean(nvm):.4f} (max {max(nvm):.4f}); ratio {np.mean(nvm)/np.mean(bfm):.2f}")
print(f"all 4 NVFP4-deq below both BF16: {max(nvm)<min(bfm)}; P under exchangeability of 2 BF16 ranking top-2 of 6 = {2*math.factorial(4)*math.factorial(2)/math.factorial(6)/2:.3f}")
lp=[m[d] for d in dirs if d in m and d not in ("arm_bf16","arm_bf16_4x4_long","arm_nvfp4_deq_cache_smoke")]
print(f"all {len(lp)} low-precision arms/replicates below both BF16: {max(lp)<min(bfm)}; P(2 BF16 rank top-2 of {len(lp)+2}) = {math.factorial(len(lp))*math.factorial(2)/math.factorial(len(lp)+2):.4f}")
print("\n### paired sign counts steps 1-10 (BF16 - NVFP4deq)")
for bfr in ["arm_bf16","arm_bf16_4x4_long"]:
    for nvr in ["arm_nvfp4_deq","arm_nvfp4_deq_4x4_long"]:
        d=[RW[bfr][s]-RW[nvr][s] for s in range(1,11)]
        print(f"{bfr} vs {nvr}: + {sum(x>1e-9 for x in d)}  0 {sum(abs(x)<=1e-9 for x in d)}  - {sum(x<-1e-9 for x in d)}  mean {np.mean(d):+.4f}")
# per-step SD across the 4 NVFP4 replicates vs across the 2 BF16 -> prompt effect
print("\n### per-step reward across replicates (prompt-difficulty signature)")
for s in range(1,11):
    print(f"step {s:2d}: BF16 {[f'{RW[d][s]:.3f}' for d in ['arm_bf16','arm_bf16_4x4_long']]}  NVFP4deq {[f'{RW[d][s]:.3f}' for d in ['arm_nvfp4_deq','arm_nvfp4_deq_4x4','arm_nvfp4_deq_4x4_100','arm_nvfp4_deq_4x4_long']]}")
