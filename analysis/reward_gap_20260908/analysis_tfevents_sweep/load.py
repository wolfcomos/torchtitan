import os, glob, json, pickle, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
RUNS=["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]
out={}
for run in RUNS:
    files=sorted(glob.glob(f"{BASE}/{run}/events.out.tfevents.*"))
    data={}   # tag -> {step: (value, fileidx, walltime)}
    fileinfo=[]
    for fi,f in enumerate(files):
        ea=EventAccumulator(f, size_guidance={'scalars':0}); ea.Reload()
        tags=ea.Tags()['scalars']
        steps=set()
        for t in tags:
            for e in ea.Scalars(t):
                if e.step>300: continue
                data.setdefault(t,{})[e.step]=(e.value,fi,e.wall_time)  # later file overwrites
                steps.add(e.step)
        fileinfo.append((os.path.basename(f),len(tags),min(steps) if steps else None,max(steps) if steps else None,len(steps)))
    out[run]={'data':data,'files':fileinfo}
    print(run)
    for x in fileinfo: print("  ",x)
    print("  ntags",len(data))
pickle.dump(out,open("data.pkl","wb"))
tags_b=set(out[RUNS[0]]['data']); tags_n=set(out[RUNS[1]]['data'])
print("tags only bf16:",tags_b-tags_n); print("tags only nvfp4:",tags_n-tags_b)
print(sorted(tags_b|tags_n))
