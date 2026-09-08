import pickle, zipfile, io, numpy as np, functools
class Stub:
    def __init__(self,*a,**k): self.args=a; self.kw=k
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st,dict) else {'state':st})
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('torch'): return type(name,(Stub,),{'__module__':mod})
        return super().find_class(mod,name)
    def persistent_load(self, pid): return pid
@functools.lru_cache(None)
def meta(d): return U(open(d+'/.metadata','rb')).load()
def chunks(d,fqn):
    m=meta(d); v=m.state_dict_metadata[fqn]
    out=[]
    for i,c in enumerate(v.__dict__['chunks']):
        off=tuple(c.__dict__['offsets'].args[0]); sz=tuple(c.__dict__['sizes'].args[0])
        out.append((i,off,sz))
    return out, tuple(v.__dict__['size'].args[0])
def read_chunk(d,fqn,idx,dtype=np.uint16):
    m=meta(d)
    for k,v in m.storage_data.items():
        if k.__dict__['fqn']==fqn and k.__dict__['index']==idx:
            info=v.__dict__; break
    else: raise KeyError((fqn,idx))
    with open(f"{d}/{info['relative_path']}",'rb') as f:
        f.seek(info['offset']); blob=f.read(info['length'])
    z=zipfile.ZipFile(io.BytesIO(blob)); name=[n for n in z.namelist() if n.endswith('data/0')][0]
    raw=z.read(name)
    ch,_=chunks(d,fqn); sz=[c for c in ch if c[0]==idx][0][2]
    a=np.frombuffer(raw,dtype=dtype)
    assert a.size==int(np.prod(sz)) if sz else 1, (a.size,sz)
    return a.reshape(sz) if sz else a
