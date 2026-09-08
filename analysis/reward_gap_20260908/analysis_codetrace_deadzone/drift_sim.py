import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fo6_np import *
from hf_read import load as hf_load, bf16_to_f32 as b2f
L=12; experts=list(range(8))
w=b2f(np.stack([hf_load(f'model.layers.{L}.mlp.experts.{e}.gate_proj.weight') for e in experts]))
c0,s0,d0,p0,a0=quantize_expert_stack(w); db0=f32_to_bf16_bits(d0); wb0=f32_to_bf16_bits(w)
out={'n':int(w.size),'median_absw':float(np.median(np.abs(w))),'frac_pick4':float(p0.mean()),'frac_code_zero':float((c0==0).mean()),
     'frac_code_abs':{str(k):float((np.abs(c0)==k).mean()) for k in range(8)}}
TINY=2.0**-12
tiny=np.abs(w)<TINY
out['frac_tiny']=float(tiny.mean()); out['tiny_frac_fp4_nonzero']=float((c0[tiny]!=0).mean())
vals=[0.5,1,1.5,2,3,4,6]; out['e2m1_rel_spacing_pct']={f'{vals[i]}->{vals[i+1]}':round(100*(vals[i+1]-vals[i])/vals[i],1) for i in range(len(vals)-1)}
out['e4m3_rel_spacing_pct']=12.5; out['bf16_rel_spacing_pct']='0.39-0.78'
blk_amax=np.abs(w.reshape(-1,16)).max(axis=1)
out['block_amax_q50']=float(np.median(blk_amax)); out['expert_amax']=[float(x) for x in a0]
# dequantization error of the recipe itself on these weights
out['fp4_rel_rmse']=float(np.sqrt(np.mean((d0-w)**2))/np.sqrt(np.mean(w**2)))
out['rel']={}
rng=np.random.default_rng(0)
for d in [0.001,0.01,0.03,0.1]:
    w2=(w*(1+d)).astype(np.float32); c1,s1,d1,p1,a1=quantize_expert_stack(w2)
    out['rel'][str(d)]={'fp4_codes_changed':float((c1!=c0).mean()),'fp4_deq_bf16_changed':float((f32_to_bf16_bits(d1)!=db0).mean()),
                        'fp4_scales_changed':float((s1!=s0).mean()),'bf16_changed':float((f32_to_bf16_bits(w2)!=wb0).mean())}
    w3=(w*(1+d*rng.choice([-1,1],size=w.shape))).astype(np.float32); c3,s3,d3,p3,a3=quantize_expert_stack(w3)
    out['rel'][str(d)].update({'randsign_fp4_codes_changed':float((c3!=c0).mean()),'randsign_bf16_changed':float((f32_to_bf16_bits(w3)!=wb0).mean())})
out['abs']={}
for a in [1e-6,1e-5,1e-4,3e-4,1e-3]:
    w2=(w+a*rng.choice([-1,1],size=w.shape)).astype(np.float32); c1,s1,d1,p1,a1=quantize_expert_stack(w2)
    wb2=f32_to_bf16_bits(w2); wq=b2f(wb2); c2,s2,d2,p2,a2=quantize_expert_stack(wq)
    out['abs'][str(a)]={'fp4_codes_changed_fp32master':float((c1!=c0).mean()),'bf16_changed':float((wb2!=wb0).mean()),
                        'fp4_codes_changed_after_bf16master':float((c2!=c0).mean()),'fp4_deq_changed_after_bf16master':float((f32_to_bf16_bits(d2)!=db0).mean())}
json.dump(out,open('drift_sim_results.json','w'),indent=1); print(json.dumps(out,indent=1))
