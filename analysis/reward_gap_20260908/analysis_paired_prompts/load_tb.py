import glob, os, json, collections
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
res={}
for run in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
    files=sorted(glob.glob(f"{OUT}/{run}/events.out.tfevents.*"))
    series=collections.defaultdict(dict)  # tag -> step -> value (later file wins)
    for f in files:
        ea=EventAccumulator(f, size_guidance={'scalars':0}); ea.Reload()
        for tag in ea.Tags()['scalars']:
            for e in ea.Scalars(tag):
                if e.step<=300: series[tag][e.step]=e.value
    res[run]={t:{str(k):v for k,v in d.items()} for t,d in series.items()}
    print(run, "tags:",len(series))
    print(sorted(series.keys()))
json.dump(res, open("tb_series.json","w"))
