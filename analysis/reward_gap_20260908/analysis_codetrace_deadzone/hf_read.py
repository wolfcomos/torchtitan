import json, struct, numpy as np, os
D='/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/miles_smoke/models/Qwen3-30B-A3B'
idx=json.load(open(f'{D}/model.safetensors.index.json'))['weight_map']
_hdr={}
def header(f):
    if f in _hdr: return _hdr[f]
    with open(f,'rb') as fh:
        n=struct.unpack('<Q', fh.read(8))[0]; h=json.loads(fh.read(n)); 
    _hdr[f]=(h,8+n); return _hdr[f]
def bf16_to_f32(u16):
    return (u16.astype(np.uint32)<<16).view(np.float32)
def load(name):
    f=f'{D}/{idx[name]}'; h,base=header(f); e=h[name]; assert e['dtype']=='BF16', e
    s,t=e['data_offsets']
    a=np.fromfile(f, dtype=np.uint16, count=(t-s)//2, offset=base+s).reshape(e['shape'])
    return a
if __name__=='__main__':
    import sys
    for name in sys.argv[1:]:
        a=bf16_to_f32(load(name)); ab=np.abs(a)
        q=np.quantile(ab,[0.01,0.1,0.5,0.9,0.99,0.999,1.0])
        print(name, a.shape, 'median|w| %.3e  q01 %.2e q10 %.2e q90 %.2e q99 %.2e q999 %.2e max %.2e  frac<2.5e-4: %.4f' % (q[2],q[0],q[1],q[3],q[4],q[5],q[6], (ab<2.5e-4).mean()))
