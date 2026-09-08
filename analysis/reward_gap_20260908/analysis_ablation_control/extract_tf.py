import glob, os, sys, json, csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
OUT="/tmp/claude-147915/-home-hanlinb/87c5c385-2451-4aa9-92a6-6e561d466781/scratchpad/analysis_ablation_control"
runs=["arm_bf16","arm_nvfp4_deq","arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]
MAXSTEP=300
data={}
for r in runs:
    files=sorted(glob.glob(f"{BASE}/{r}/events.out.tfevents.*"))
    d={}  # tag -> {step: value}; later file wins
    for f in files:
        ea=EventAccumulator(f, size_guidance={'scalars':0}); ea.Reload()
        tags=ea.Tags()['scalars']
        for t in tags:
            for ev in ea.Scalars(t):
                if ev.step<=MAXSTEP:
                    d.setdefault(t,{})[ev.step]=ev.value
    data[r]=d
    print(r, "files:",len(files), "tags:",len(d))
    steps=sorted(set(s for t in d for s in d[t]))
    print("  step range:", steps[0] if steps else None, steps[-1] if steps else None, "n=",len(steps))
json.dump(data, open(f"{OUT}/scalars.json","w"))
# tags list
alltags=sorted(set(t for r in data for t in data[r]))
open(f"{OUT}/tags.txt","w").write("\n".join(alltags))
print("\n".join(alltags))
