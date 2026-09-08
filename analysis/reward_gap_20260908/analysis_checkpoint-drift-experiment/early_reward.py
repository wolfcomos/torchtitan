import glob, os, sys, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
base="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
out={}
for run in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
    files=sorted(glob.glob(f"{base}/{run}/events.out.tfevents.*"))
    series={}
    for f in files:  # later file wins on duplicate steps
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        for tag in ["rollout_reward/_mean","rollout_reward/group_zero_std_frac/mean","trainer/grad_norm/mean"]:
            if tag not in ea.Tags()['scalars']: continue
            d=series.setdefault(tag,{})
            for e in ea.Scalars(tag):
                if e.step<=300: d[e.step]=e.value
    r={}
    for tag,d in series.items():
        steps=sorted(d)
        v=np.array([d[s] for s in steps]); st=np.array(steps)
        r[tag]={f"mean_steps_1-{n}":float(v[(st>=1)&(st<=n)].mean()) for n in [5,10,20,50,100]}
        r[tag]["n_steps"]=len(steps); r[tag]["first_steps"]=[(int(s),round(float(d[s]),4)) for s in steps[:12]]
    out[run]=r
    print(run,json.dumps(r,indent=0))
json.dump(out,open("early_reward.json","w"),indent=1)
