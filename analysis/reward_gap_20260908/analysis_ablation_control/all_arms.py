import glob, os, json, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
OUT="/tmp/claude-147915/-home-hanlinb/87c5c385-2451-4aa9-92a6-6e561d466781/scratchpad/analysis_ablation_control"
tags=["rollout_reward/_mean","trainer/lr","rollout_reward/group_zero_std_frac/mean","bit_wise/logprob_diff/mean","loss/ratio_clipped_frac"]
res={}
for d in sorted(os.listdir(BASE)):
    files=sorted(glob.glob(f"{BASE}/{d}/events.out.tfevents.*"))
    if not files: continue
    data={}
    for f in files:
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        for t in tags:
            if t in ea.Tags()['scalars']:
                for ev in ea.Scalars(t):
                    if ev.step<=30: data.setdefault(t,{})[ev.step]=ev.value
    if not data: print(f"{d}: tfevents but no steps"); continue
    steps=sorted(data.get("rollout_reward/_mean",{}))
    res[d]=data
    r=data.get("rollout_reward/_mean",{}); lr=data.get("trainer/lr",{})
    print(f"\n== {d}: files={len(files)} steps<=30: {steps[0] if steps else None}-{steps[-1] if steps else None} n={len(steps)}")
    print("  reward 1-10:", [f"{r[s]:.4f}" if s in r else "-" for s in range(1,11)], "mean:", f"{np.mean([r[s] for s in range(1,11) if s in r]):.4f}" if any(s in r for s in range(1,11)) else "-")
    print("  lr 1-10    :", [f"{lr[s]:.2e}" if s in lr else "-" for s in range(1,11)])
    z=data.get("rollout_reward/group_zero_std_frac/mean",{})
    print("  zstd 1-10  :", [f"{z[s]:.3f}" if s in z else "-" for s in range(1,11)])
json.dump({d:{t:{str(k):v for k,v in dd[t].items()} for t in dd} for d,dd in res.items()}, open(f"{OUT}/all_arms_first30.json","w"))
