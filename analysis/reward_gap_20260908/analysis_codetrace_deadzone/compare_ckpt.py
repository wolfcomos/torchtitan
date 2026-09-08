import numpy as np, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fo6_np import *
from dcp_read import read_chunk, chunks
from hf_read import load as hf_load, bf16_to_f32 as b2f
OUT='/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs'
ARMS={'bf16':f'{OUT}/arm_bf16_4x4_long/parked/step-300','nvfp4':f'{OUT}/arm_nvfp4_deq_4x4_long/parked/step-300'}
LAYERS=[0,12,24,36,39,44]
TINY=2.0**-12   # |w| below this: bf16 half-ulp < 1e-6 (ulp(w)=2^(floor(log2|w|)-7); half-ulp<1e-6 <=> |w|<2^-12=2.44e-4)
res={}
def ulp_bf16(w):
    a=np.abs(w.astype(np.float32)); m,e=np.frexp(a); return np.exp2(e-1-7).astype(np.float32)
def cmp(init_u16, ck_u16):
    init_u16=init_u16.ravel(); ck_u16=ck_u16.ravel()
    ch=init_u16!=ck_u16; n=ch.size; nc=int(ch.sum())
    wi=b2f(init_u16); wc=b2f(ck_u16)
    tiny=np.abs(wi)<TINY
    r={'n':n,'frac_changed':nc/n,'frac_tiny_init':float(tiny.mean()),'frac_tiny_changed':float(ch[tiny].mean()) if tiny.any() else None,
       'frac_nontiny_changed':float(ch[~tiny].mean())}
    if nc:
        d=np.abs(wc[ch]-wi[ch]); wi_c=np.abs(wi[ch]); u=ulp_bf16(wi[ch])
        r.update({'changed_absw_init_q50':float(np.median(wi_c)),'changed_absw_init_q90':float(np.quantile(wi_c,.9)),'changed_absw_init_max':float(wi_c.max()),
                  'changed_frac_tiny':float((wi_c<TINY).mean()),'absdelta_q50':float(np.median(d)),'absdelta_q90':float(np.quantile(d,.9)),'absdelta_max':float(d.max()),
                  'frac_changed_multi_ulp':float((d>1.5*u).mean()),'ulps_moved_q50':float(np.median(d/u)),'ulps_moved_max':float((d/u).max())})
    return r
def fp4_cmp(w_init, w_ck, tag):
    c0,s0,d0,p0,a0=quantize_expert_stack(w_init); c1,s1,d1,p1,a1=quantize_expert_stack(w_ck)
    db0=f32_to_bf16_bits(d0); db1=f32_to_bf16_bits(d1)
    return {'fp4_frac_codes_changed':float((c0!=c1).mean()),'fp4_frac_deq_bf16_changed':float((db0!=db1).mean()),'fp4_frac_scales_changed':float((s0!=s1).mean()),
            'fp4_n_expert_amax_changed':int((a0!=a1).sum()),'fp4_frac_pick4':float(p0.mean()),'fp4_frac_code_zero_init':float((c0==0).mean()),
            'fp4_frac_bf16elems_changed':float((f32_to_bf16_bits(w_init)!=f32_to_bf16_bits(w_ck)).mean())}
def hf_stack(layer, proj, experts):
    return np.stack([hf_load(f'model.layers.{layer}.mlp.experts.{e}.{proj}.weight') for e in experts])
t0=time.time()
NE_FP4=8
for L in LAYERS:
    for tname,proj in [('w1_EFD','gate_proj'),('w3_EFD','up_proj'),('w2_EDF','down_proj')]:
        fqn=f'layers.{L}.moe.routed_experts.inner_experts.{tname}'
        ch,_=chunks(ARMS['bf16'],fqn); off=ch[0][1]; sz=ch[0][2]; experts=list(range(off[0],off[0]+sz[0]))
        hf=hf_stack(L,proj,experts)
        for arm,d in ARMS.items():
            ck=read_chunk(d,fqn,0)
            key=f'L{L}.{tname}'; res.setdefault(key,{})[arm]=cmp(hf,ck)
            if L<=39:
                res[key][arm].update(fp4_cmp(b2f(hf[:NE_FP4]), b2f(ck[:NE_FP4]), key))
            print(f'[{time.time()-t0:.0f}s] {key} {arm} frac_changed={res[key][arm]["frac_changed"]:.5f} ', {k:v for k,v in res[key][arm].items() if k.startswith("fp4_frac_codes") or k.startswith("fp4_frac_deq")}, flush=True)
    for fqn,hfname in [(f'layers.{L}.attention.qkv_linear.wq.weight',f'model.layers.{L}.self_attn.q_proj.weight'),
                       (f'layers.{L}.attention.wo.weight',f'model.layers.{L}.self_attn.o_proj.weight'),
                       (f'layers.{L}.moe.router.gate.weight',f'model.layers.{L}.mlp.gate.weight'),
                       (f'layers.{L}.attention_norm.weight',f'model.layers.{L}.input_layernorm.weight')]:
        hf=hf_load(hfname); ch,_=chunks(ARMS['bf16'],fqn); off=ch[0][1]; sz=ch[0][2]
        sl=tuple(slice(o,o+s) for o,s in zip(off,sz)); hfc=hf[sl]
        for arm,d in ARMS.items():
            ck=read_chunk(d,fqn,0); res.setdefault(fqn,{})[arm]=cmp(hfc,ck)
            print(f'[{time.time()-t0:.0f}s] {fqn} {arm} frac_changed={res[fqn][arm]["frac_changed"]:.5f} multi_ulp={res[fqn][arm].get("frac_changed_multi_ulp")}', flush=True)
for fqn,hfname in [('tok_embeddings.weight','model.embed_tokens.weight'),('lm_head.weight','lm_head.weight'),('norm.weight','model.norm.weight')]:
    hf=hf_load(hfname); ch,_=chunks(ARMS['bf16'],fqn); off=ch[0][1]; sz=ch[0][2]; sl=tuple(slice(o,o+s) for o,s in zip(off,sz)); hfc=hf[sl]
    for arm,d in ARMS.items():
        ck=read_chunk(d,fqn,0); res.setdefault(fqn,{})[arm]=cmp(hfc,ck)
        print(f'[{time.time()-t0:.0f}s] {fqn} {arm} frac_changed={res[fqn][arm]["frac_changed"]:.5f}', flush=True)
for fqn in ['layers.12.moe.routed_experts.inner_experts.w1_EFD','layers.12.attention.qkv_linear.wqkv.weight','layers.44.moe.routed_experts.inner_experts.w1_EFD']:
    for arm,d in ARMS.items():
        r={}
        for st in ['exp_avg','exp_avg_sq']:
            a=read_chunk(d,f'optimizer.state.{fqn}.{st}',0); r[f'{st}_frac_nonzero']=float((a!=0).mean())
            v=np.abs(b2f(a)); nz=v[v>0]; r[f'{st}_abs_median_nonzero']=float(np.median(nz)) if nz.size else None
        res.setdefault('optstate.'+fqn,{})[arm]=r; print(f'[{time.time()-t0:.0f}s] optstate {fqn} {arm} {r}', flush=True)
json.dump(res,open('compare_ckpt_results.json','w'),indent=1)
print('DONE', time.time()-t0)
