import pickle, sys, zipfile, io
class Stub:
    def __init__(self,*a,**k): self.args=a; self.kw=k
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st,dict) else {'state':st})
    def __repr__(self): return f"{type(self).__name__}({self.__dict__})"
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('torch'): return type(name,(Stub,),{'__module__':mod})
        return super().find_class(mod,name)
    def persistent_load(self, pid): return pid
d=sys.argv[1]
m=U(open(d+'/.metadata','rb')).load()
smd=m.state_dict_metadata
for k in ['layers.0.moe.routed_experts.inner_experts.w1_EFD','layers.0.attention.wo.weight','tok_embeddings.weight','optimizer.state.layers.0.moe.routed_experts.inner_experts.w1_EFD.exp_avg','optimizer.state.layers.0.moe.routed_experts.inner_experts.w1_EFD.step']:
    v=smd.get(k); 
    if v is None: print(k,'MISSING'); continue
    print(k, {kk:(vv.__dict__ if hasattr(vv,'__dict__') else vv) for kk,vv in v.__dict__.items() if kk!='chunks'})
    print('   chunks:', [c.__dict__ for c in v.__dict__['chunks']] if v.__dict__.get('chunks') else None)
sd=m.storage_data
ks=[k for k in sd if getattr(k,'__dict__',{}).get('fqn')=='layers.0.moe.routed_experts.inner_experts.w1_EFD']
for k in ks:
    print('storage', k.__dict__, sd[k].__dict__)
# peek at bytes
k=ks[0]; info=sd[k].__dict__
p=f"{d}/{info['relative_path']}"; off=info['offset']; ln=info['length']
with open(p,'rb') as f:
    f.seek(off); b=f.read(min(ln,4096))
print('head bytes', b[:64])
print('is zip', b[:2]==b'PK', 'len', ln, 'numel*2', 32*768*2048*2)
# try zip parse of the full item
with open(p,'rb') as f:
    f.seek(off); blob=f.read(ln)
try:
    z=zipfile.ZipFile(io.BytesIO(blob)); print([ (i.filename,i.file_size) for i in z.infolist()])
except Exception as e: print('zip fail', e)
