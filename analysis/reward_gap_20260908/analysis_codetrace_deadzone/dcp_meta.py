import pickle, sys, io, collections
class Stub:
    def __init__(self,*a,**k): self.args=a; self.kw=k
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st,dict) else {'state':st})
    def __repr__(self): return f"{type(self).__name__}({self.__dict__})"
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('torch'):
            if name=='dtype' or (mod=='torch' and name in ('bfloat16','float32','float16')):
                pass
            return type(name,(Stub,),{'__module__':mod})
        return super().find_class(mod,name)
    def persistent_load(self, pid): return pid
p=sys.argv[1]
m=U(open(p,'rb')).load()
print(type(m).__name__, list(m.__dict__.keys()))
smd=m.state_dict_metadata
print('n keys', len(smd))
keys=sorted(smd)
import re
seen=set(); 
for k in keys:
    kk=re.sub(r'layers\.\d+\.', 'layers.N.', k)
    if kk in seen: continue
    seen.add(kk)
    v=smd[k]
    d=v.__dict__
    props=d.get('properties'); size=d.get('size'); chunks=d.get('chunks')
    dt = props.__dict__.get('dtype') if props is not None else None
    print(kk, '| size', size, '| dtype', dt, '| nchunks', len(chunks) if chunks else None, '| chunk0', (chunks[0].__dict__ if chunks else None))
sd=m.storage_data
print('storage_data', len(sd))
for i,(k,v) in enumerate(sd.items()):
    if i<3: print(k.__dict__ if hasattr(k,'__dict__') else k, v.__dict__)
