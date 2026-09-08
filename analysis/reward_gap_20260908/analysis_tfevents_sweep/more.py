import pickle, numpy as np
out=pickle.load(open("data.pkl","rb"))
B="arm_bf16_4x4_long"; N="arm_nvfp4_deq_4x4_long"
def arr(run,tag,lo=1,hi=300):
    d=out[run]['data'][tag]; return np.array([d[s][0] for s in range(lo,hi+1)])
def fidx(run,tag,lo=1,hi=300):
    d=out[run]['data'][tag]; return np.array([d[s][1] for s in range(lo,hi+1)])
print("=== per-launch (tfevents file) means: steps, reward, zstd, lpd_mean, lpd_max, clip, decode_ms, ITL_ms, fwd_bwd_s, step_total_s")
for run in [B,N]:
    print(run)
    fi=fidx(run,'rollout_reward/_mean'); files=out[run]['files']
    for f in sorted(set(fi)):
        m=fi==f; steps=np.arange(1,301)[m]
        g=lambda t: arr(run,t)[m].mean()
        print(f"  file{f} {files[f][0].split('.')[3]:26s} steps {steps.min():3d}-{steps.max():3d} n={m.sum():3d} rew {g('rollout_reward/_mean'):.3f} zstd {g('rollout_reward/group_zero_std_frac/mean'):.3f} lpd {g('bit_wise/logprob_diff/mean'):+.5f} lpdmax {g('bit_wise/logprob_diff/max'):.2f} clip {g('loss/ratio_clipped_frac'):.4f} decode {g('generator/decode_time_ms/mean')/1e3:.0f}s ITL {g('generator/inter_token_latency_ms/mean'):.0f}ms fwdbwd {g('timing/step/forward_backward/mean'):.1f}s total {g('timing/step/total/mean'):.0f}s")
rb=arr(B,'rollout_reward/_mean'); rn=arr(N,'rollout_reward/_mean')
print("\n=== NVFP4 reward conditional on BF16 reward (prompt-difficulty proxy), steps 201-300 and 1-100")
for lo,hi in [(1,100),(201,300)]:
    b=rb[lo-1:hi]; n=rn[lo-1:hi]
    for blo,bhi in [(0,0.0001),(0.0001,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,1.01)]:
        m=(b>=blo)&(b<bhi)
        if m.sum(): print(f"  [{lo}-{hi}] bf16 reward in [{blo:.2f},{bhi:.2f}): n={m.sum():3d} steps, bf16 mean {b[m].mean():.3f}, nvfp4 mean {n[m].mean():.3f}, ratio {n[m].mean()/max(b[m].mean(),1e-9):.2f}")
    # OLS slope of nvfp4 on bf16
    A=np.vstack([b,np.ones_like(b)]).T; coef=np.linalg.lstsq(A,n,rcond=None)[0]
    print(f"  [{lo}-{hi}] OLS nvfp4 = {coef[0]:.3f}*bf16 + {coef[1]:.4f}")
print("\n=== linear trend slope per 100 steps (OLS on steps 1-300) with SE")
for tag in ['rollout_reward/_mean','rollout_reward/group_zero_std_frac/mean','trainer/grad_norm/mean','loss/mean','bit_wise/logprob_diff/mean','bit_wise/logprob_diff/max','loss/ratio_clipped_frac','trainer/entropy/mean','rollout/response_length/mean','rollout/truncation_rate/mean','advantage/_std','rollout_reward/_std']:
    line=f"  {tag:46s}"
    for run in [B,N]:
        y=arr(run,tag); x=np.arange(1,301)/100.0
        A=np.vstack([x,np.ones_like(x)]).T; coef,res,_,_=np.linalg.lstsq(A,y,rcond=None)
        yhat=A@coef; s2=((y-yhat)**2).sum()/(300-2); se=np.sqrt(s2/((x-x.mean())**2).sum())
        line+=f" | {run[4:9]} slope {coef[0]:+.4g}/100steps (t={coef[0]/se:+.1f})"
    print(line)
print("\n=== sampling-noise floor from replayed steps (same weights, same prompts, two launches)")
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
for run in [B,N]:
    files=sorted(glob.glob(f"{BASE}/{run}/events.out.tfevents.*")); per={}
    for fi,f in enumerate(files):
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        if 'rollout_reward/_mean' in ea.Tags()['scalars']:
            for e in ea.Scalars('rollout_reward/_mean'):
                if e.step<=300: per.setdefault(e.step,[]).append(e.value)
    d=np.array([v[1]-v[0] for s,v in per.items() if len(v)>1]); a=np.array([v[0] for s,v in per.items() if len(v)>1]); b=np.array([v[1] for s,v in per.items() if len(v)>1])
    print(f"  {run}: n={len(d)} replayed steps; mean|replay diff| {np.abs(d).mean():.4f}, std of diff {d.std(ddof=1):.4f}, corr earlier-vs-later {np.corrcoef(a,b)[0,1]:.3f}, mean reward earlier {a.mean():.3f} later {b.mean():.3f}")
# for comparison: cross-run per-step |diff| and std in same range
d=rb-rn; print(f"  cross-run per-step diff (all 300): mean {d.mean():+.4f}, std {d.std(ddof=1):.4f}; steps 1-30 mean {d[:30].mean():+.4f} std {d[:30].std(ddof=1):.4f}")
# signal-carrying groups per step
for run in [B,N]:
    z=arr(run,'rollout_reward/group_zero_std_frac/mean'); print(f"  {run}: signal groups/step (8*(1-zstd)) b1 {8*(1-z[:100].mean()):.2f} b2 {8*(1-z[100:200].mean()):.2f} b3 {8*(1-z[200:].mean()):.2f}; steps with 0 signal groups: {(z==1).sum()}")
