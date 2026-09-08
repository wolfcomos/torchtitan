import torch, json, sys
from torch.distributed.checkpoint import FileSystemReader
for run in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
    d=f"/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/{run}/parked/step-300"
    md=FileSystemReader(d).read_metadata()
    sd=md.state_dict_metadata
    out={}
    for k,v in sd.items():
        if hasattr(v,'size'):
            out[k]={"shape":list(v.size),"dtype":str(v.properties.dtype),"nchunks":len(v.chunks)}
        else:
            out[k]={"type":type(v).__name__,"val":str(v)[:200]}
    json.dump(out,open(f"/out/analysis_checkpoint-drift-experiment/metadata_{run}.json","w"),indent=1)
    print(run,len(sd))
    keys=[k for k in sd if k.startswith("model.")]
    print("model keys",len(keys))
    for k in keys:
        if ".layers." in k and not (".layers.0." in k): continue
        print(" ",k,out[k])
    others=[k for k in sd if not k.startswith("model.")]
    print("non-model keys sample",others[:15])
    # planner info
    print("planner_data type", type(md.planner_data), str(md.planner_data)[:300] if md.planner_data else None)
    print("storage_data sample", list(md.storage_data.items())[:2])
