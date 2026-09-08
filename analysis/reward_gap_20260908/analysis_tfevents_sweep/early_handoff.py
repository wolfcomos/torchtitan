import pickle, numpy as np, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out=pickle.load(open("data.pkl","rb"))
B="arm_bf16_4x4_long"; N="arm_nvfp4_deq_4x4_long"
def arr(run,tag,lo=1,hi=300):
    d=out[run]['data'][tag]; return np.array([d[s][0] for s in range(lo,hi+1)])
rb=arr(B,'rollout_reward/_mean'); rn=arr(N,'rollout_reward/_mean')
zb=arr(B,'rollout_reward/group_zero_std_frac/mean'); zn=arr(N,'rollout_reward/group_zero_std_frac/mean')
print("=== steps 1-12 per-step: reward(bf16/nvfp4) zero_std(bf16/nvfp4) lr entropy resp_len")
for s in range(1,13):
    print(f"  step {s:2d}: rew {rb[s-1]:.4f}/{rn[s-1]:.4f}  zstd {zb[s-1]:.3f}/{zn[s-1]:.3f}  lr {out[B]['data']['trainer/lr'][s][0]:.2e}/{out[N]['data']['trainer/lr'][s][0]:.2e}  ent {out[B]['data']['trainer/entropy/mean'][s][0]:.3f}/{out[N]['data']['trainer/entropy/mean'][s][0]:.3f}  len {out[B]['data']['rollout/response_length/mean'][s][0]:.0f}/{out[N]['data']['rollout/response_length/mean'][s][0]:.0f}")
# paired tests on per-step reward difference (same prompts)
def paired(lo,hi):
    d=rb[lo-1:hi]-rn[lo-1:hi]; n=len(d); t=d.mean()/(d.std(ddof=1)/np.sqrt(n))
    pos=(d>0).sum(); neg=(d<0).sum(); tie=(d==0).sum()
    # correct-sample counts (64 samples/step)
    cb=int(round(rb[lo-1:hi].sum()*64)); cn=int(round(rn[lo-1:hi].sum()*64))
    return f"steps {lo}-{hi}: mean diff {d.mean():+.4f}, paired t={t:+.2f} (n={n}), bf16>nvfp4 on {pos} steps, < on {neg}, tie {tie}; correct samples {cb} vs {cn} of {64*n}"
for lo,hi in [(1,10),(1,20),(1,30),(1,50),(1,100),(101,200),(201,300),(1,300)]: print(" ",paired(lo,hi))
# HANDOFFS: replayed steps, compare file A vs file B values on the same step
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
tags_chk=['rollout_reward/_mean','rollout_reward/group_zero_std_frac/mean','bit_wise/logprob_diff/mean','bit_wise/logprob_diff/max','loss/ratio_clipped_frac','trainer/entropy/mean','rollout/response_length/mean','trainer/grad_norm/mean','loss/mean','trainer/lr','rollout/prompt_length/mean','trainer/policy_version']
for run in [B,N]:
    print(f"\n=== {run}: replayed steps (same step in two files) -- earlier-file value vs later-file value")
    files=sorted(glob.glob(f"{BASE}/{run}/events.out.tfevents.*"))
    per={}
    for fi,f in enumerate(files):
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        for t in tags_chk:
            if t in ea.Tags()['scalars']:
                for e in ea.Scalars(t):
                    if e.step<=300: per.setdefault(t,{}).setdefault(e.step,[]).append((fi,e.value))
    dup_steps=sorted({s for s in per['rollout_reward/_mean'] if len(per['rollout_reward/_mean'][s])>1})
    print("  replayed steps:",dup_steps)
    for s in dup_steps:
        line=f"  step {s}:"
        for t in tags_chk:
            vals=per[t].get(s,[])
            if len(vals)>1: line+=f" {t.split('/')[-2] if t.endswith('mean') or t.endswith('max') else t.split('/')[-1]}=" + "|".join(f"{v:.4g}" for _,v in vals)
        print(line)
# local means around handoffs (later-file-wins series): 5 steps before vs 5 after
def around(run,tag,h):
    a=arr(run,tag); pre=a[max(0,h-6):h-1]; post=a[h-1:h+4]   # steps h-5..h-1 vs h..h+4
    return pre.mean(),post.mean()
for run,hs in [(N,[30,70,106,148,189,227,263]),(B,[67,126,185,248])]:
    print(f"\n=== {run}: 5-step means before/after handoff (first step of new launch = h)")
    for h in hs:
        line=f"  h={h:3d}:"
        for t,lab in [('rollout_reward/_mean','rew'),('bit_wise/logprob_diff/mean','lpd'),('bit_wise/logprob_diff/max','lpdmax'),('loss/ratio_clipped_frac','clip'),('trainer/entropy/mean','ent'),('trainer/grad_norm/mean','gn'),('rollout_reward/group_zero_std_frac/mean','zstd')]:
            p,q=around(run,t,h); line+=f" {lab} {p:.4g}->{q:.4g}"
        print(line)
# generic: is the mean over 'first 3 steps of each launch' different from the rest?
for run,hs in [(N,[30,70,106,148,189,227,263]),(B,[67,126,185,248])]:
    for t in ['rollout_reward/_mean','bit_wise/logprob_diff/mean','bit_wise/logprob_diff/max','loss/ratio_clipped_frac']:
        a=arr(run,t); first=np.concatenate([a[h-1:h+2] for h in hs]); mask=np.ones(300,bool)
        for h in hs: mask[h-1:h+2]=False
        print(f"  {run} {t}: first-3-steps-after-resume mean {first.mean():.4g} (n={len(first)}) vs other steps {a[mask].mean():.4g}")
