import glob, numpy as np, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TAGS=['loss/ratio_mean','bit_wise/ratio_tokens_different/mean','loss/generator_logprob_nan_frac','advantage/_std','advantage/_mean','train_batch/num_training_samples','train_batch/num_rollout_groups','train_batch/num_metric_only_groups','rollout/truncation_rate/mean','training_sample_builder/num_training_samples/sum','loss/ratio_clipped_frac','bit_wise/logprob_diff/mean','bit_wise/logprob_diff/max','trainer/grad_norm/mean','rollout_reward/_mean','rollout_reward/group_zero_std_frac/mean']
out={}
for run in ['arm_bf16_4x4_long','arm_nvfp4_deq_4x4_long']:
    d=f'/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/{run}'
    files=sorted(glob.glob(d+'/events.out.tfevents.*'))
    series={t:{} for t in TAGS}
    for f in files:
        ea=EventAccumulator(f, size_guidance={'scalars':0}); ea.Reload()
        for t in TAGS:
            if t in ea.Tags()['scalars']:
                for e in ea.Scalars(t):
                    if e.step<=300: series[t][e.step]=e.value
    out[run]={}
    for t in TAGS:
        s=series[t]; 
        if not s: out[run][t]=None; continue
        st=np.array(sorted(s)); v=np.array([s[k] for k in st])
        blk=lambda a,b: float(v[(st>=a)&(st<=b)].mean())
        out[run][t]={'mean_1_300':float(v.mean()),'b1':blk(1,100),'b2':blk(101,200),'b3':blk(201,300),'max':float(v.max()),'n':int(len(st))}
    # distribution of per-step logprob_diff max for nvfp4
    s=series['bit_wise/logprob_diff/max']; v=np.array([s[k] for k in sorted(s)])
    out[run]['logprob_diff_max_quantiles']={q:float(np.quantile(v,q)) for q in [0.5,0.9,0.99,1.0]}
    out[run]['logprob_diff_max_steps_gt5']=int((v>5).sum())
json.dump(out,open('tf_aux.json','w'),indent=1)
for run in out:
    print('==',run)
    for t,v in out[run].items(): print(f'  {t}: {v}')
