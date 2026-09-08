import pickle, numpy as np
out=pickle.load(open("data.pkl","rb"))
B="arm_bf16_4x4_long"; N="arm_nvfp4_deq_4x4_long"
def arr(run,tag,lo=1,hi=300):
    d=out[run]['data'][tag]; return np.array([d[s][0] for s in range(lo,hi+1)])
def pear(x,y): return float(np.corrcoef(x,y)[0,1])
print("=== Pearson correlations per-step BF16 vs NVFP4 (same prompts)")
for tag in ['rollout_reward/_mean','rollout_reward/_std','rollout_reward/group_zero_std_frac/mean','rollout_reward/_max','trainer/entropy/mean','rollout/response_length/mean','rollout/truncation_rate/mean','rollout/prompt_length/mean','bit_wise/logprob_diff/mean','bit_wise/logprob_diff/max','loss/ratio_clipped_frac','trainer/grad_norm/mean','loss/mean','advantage/_std','rollout/output_tokens/std']:
    line=f"{tag:48s}"
    for lo,hi in [(1,300),(1,100),(101,200),(201,300),(1,30)]:
        line+=f" [{lo}-{hi}] r={pear(arr(B,tag,lo,hi),arr(N,tag,lo,hi)):+.3f}"
    print(line)
# control: correlation with a lag (shifted prompts) to show prompt-difficulty is the driver
rb=arr(B,'rollout_reward/_mean'); rn=arr(N,'rollout_reward/_mean')
print("\nreward corr lag control: lag0", round(pear(rb,rn),3), "lag+1", round(pear(rb[1:],rn[:-1]),3), "lag-1", round(pear(rb[:-1],rn[1:]),3), "lag+5",round(pear(rb[5:],rn[:-5]),3))
# detrended (remove block means) correlation
def detrend(x):
    y=x.copy()
    for lo,hi in [(0,100),(100,200),(200,300)]: y[lo:hi]-=y[lo:hi].mean()
    return y
print("reward corr detrended (block means removed):", round(pear(detrend(rb),detrend(rn)),3))
# within-run correlations
print("\n=== within-run correlations (steps 1-300)")
for run in [B,N]:
    r=arr(run,'rollout_reward/_mean'); z=arr(run,'rollout_reward/group_zero_std_frac/mean'); e=arr(run,'trainer/entropy/mean'); L=arr(run,'rollout/response_length/mean'); tr=arr(run,'rollout/truncation_rate/mean'); ld=arr(run,'bit_wise/logprob_diff/mean'); ldm=arr(run,'bit_wise/logprob_diff/max'); cf=arr(run,'loss/ratio_clipped_frac'); gn=arr(run,'trainer/grad_norm/mean')
    print(run)
    print(f"  reward~zero_std {pear(r,z):+.3f}  reward~trunc {pear(r,tr):+.3f}  reward~resp_len {pear(r,L):+.3f}  reward~entropy {pear(r,e):+.3f}  reward~logprobdiff_mean {pear(r,ld):+.3f}  reward~clipfrac {pear(r,cf):+.3f}  gradnorm~zero_std {pear(gn,z):+.3f}  clipfrac~logprobdiff_max {pear(cf,ldm):+.3f}  logprobdiff_mean~resp_len {pear(ld,L):+.3f} clipfrac~trunc {pear(cf,tr):+.3f}")
# step-by-step reward: first 30 steps and when they separate (cumulative mean)
print("\n=== cumulative mean reward by step")
for s in [10,20,30,40,50,60,80,100,150,200,250,300]:
    print(f"  step<= {s:3d}: bf16 {rb[:s].mean():.4f}  nvfp4 {rn[:s].mean():.4f}  diff {rb[:s].mean()-rn[:s].mean():+.4f}")
# 20-step block means
print("\n=== 20-step block means reward / zero_std / clipfrac / logprobdiff_mean / logprobdiff_max / gradnorm")
zb=arr(B,'rollout_reward/group_zero_std_frac/mean'); zn=arr(N,'rollout_reward/group_zero_std_frac/mean')
cb=arr(B,'loss/ratio_clipped_frac'); cn=arr(N,'loss/ratio_clipped_frac')
lb=arr(B,'bit_wise/logprob_diff/mean'); ln_=arr(N,'bit_wise/logprob_diff/mean')
mb=arr(B,'bit_wise/logprob_diff/max'); mn=arr(N,'bit_wise/logprob_diff/max')
gb=arr(B,'trainer/grad_norm/mean'); gn_=arr(N,'trainer/grad_norm/mean')
xb=arr(B,'rollout_reward/_max'); xn=arr(N,'rollout_reward/_max')
for lo in range(0,300,20):
    sl=slice(lo,lo+20)
    print(f"  {lo+1:3d}-{lo+20:3d}: rew {rb[sl].mean():.3f}/{rn[sl].mean():.3f}  zstd {zb[sl].mean():.3f}/{zn[sl].mean():.3f}  clip {cb[sl].mean():.4f}/{cn[sl].mean():.4f}  lpd {lb[sl].mean():+.5f}/{ln_[sl].mean():+.5f}  lpdmax {mb[sl].mean():.2f}/{mn[sl].mean():.2f}  gn {gb[sl].mean():.3f}/{gn_[sl].mean():.3f} rmax {xb[sl].mean():.2f}/{xn[sl].mean():.2f}")
# steps with reward_max == 0 (no correct answer at all in 64 samples)
print("\nsteps with max reward 0: bf16", int((xb==0).sum()), "nvfp4", int((xn==0).sum()))
print("steps with reward mean==0: bf16", int((rb==0).sum()), "nvfp4", int((rn==0).sum()))
# reward histograms
print("reward per-step quantiles bf16 b3:", np.percentile(rb[200:],[10,25,50,75,90]).round(3), " nvfp4 b3:", np.percentile(rn[200:],[10,25,50,75,90]).round(3))
# logprob diff max: peaks
print("\nlogprobdiff max top steps bf16:", sorted([(round(float(v),2),i+1) for i,v in enumerate(mb)],reverse=True)[:6])
print("logprobdiff max top steps nvfp4:", sorted([(round(float(v),2),i+1) for i,v in enumerate(mn)],reverse=True)[:6])
print("logprobdiff max quantiles bf16:", np.percentile(mb,[10,50,90,99]).round(2), "nvfp4:", np.percentile(mn,[10,50,90,99]).round(2))
print("logprobdiff mean quantiles bf16:", np.percentile(lb,[1,10,50,90,99]).round(5), "nvfp4:", np.percentile(ln_,[1,10,50,90,99]).round(5))
print("clipfrac quantiles bf16:", np.percentile(cb,[1,10,50,90,99]).round(4), "nvfp4:", np.percentile(cn,[1,10,50,90,99]).round(4))
print("ratio_mean quantiles bf16:", np.percentile(arr(B,'loss/ratio_mean'),[1,10,50,90,99]).round(5), "nvfp4:", np.percentile(arr(N,'loss/ratio_mean'),[1,10,50,90,99]).round(5))
print("ratio_tokens_different:", arr(B,'bit_wise/ratio_tokens_different/mean').mean().round(4), arr(N,'bit_wise/ratio_tokens_different/mean').mean().round(4))
