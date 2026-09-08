import pickle, numpy as np, csv, math
out=pickle.load(open("data.pkl","rb"))
B="arm_bf16_4x4_long"; N="arm_nvfp4_deq_4x4_long"
tags=sorted(out[B]['data'])
def series(run,tag):
    d=out[run]['data'][tag]; return d
def blk(d,lo,hi):
    v=[d[s][0] for s in range(lo,hi+1) if s in d]
    return (float(np.mean(v)) if v else float('nan'), len(v))
rows=[]
for t in tags:
    db=series(B,t); dn=series(N,t)
    r={'tag':t,'n_bf16':len([s for s in db if 1<=s<=300]),'n_nvfp4':len([s for s in dn if 1<=s<=300])}
    for lo,hi,name in [(1,100,'b1'),(101,200,'b2'),(201,300,'b3'),(1,300,'all')]:
        r[f'bf16_{name}'],_=blk(db,lo,hi); r[f'nvfp4_{name}'],_=blk(dn,lo,hi)
    b3=r['bf16_b3']; n3=r['nvfp4_b3']
    r['ratio_b3']= n3/b3 if b3 not in (0,) and not math.isnan(b3) else float('nan')
    r['reldiff_b3']= (n3-b3)/max(abs(b3),abs(n3)) if max(abs(b3),abs(n3))>0 else 0.0
    # trajectory shape: slope over blocks
    r['bf16_shape']=f"{r['bf16_b1']:.4g}->{r['bf16_b2']:.4g}->{r['bf16_b3']:.4g}"
    r['nvfp4_shape']=f"{r['nvfp4_b1']:.4g}->{r['nvfp4_b2']:.4g}->{r['nvfp4_b3']:.4g}"
    # relative trend b3/b1
    r['bf16_trend']= r['bf16_b3']/r['bf16_b1'] if r['bf16_b1'] else float('nan')
    r['nvfp4_trend']= r['nvfp4_b3']/r['nvfp4_b1'] if r['nvfp4_b1'] else float('nan')
    # per-step std in b3
    vb=np.array([db[s][0] for s in range(201,301) if s in db]); vn=np.array([dn[s][0] for s in range(201,301) if s in dn])
    r['bf16_std_b3']=float(vb.std()) if len(vb) else float('nan'); r['nvfp4_std_b3']=float(vn.std()) if len(vn) else float('nan')
    rows.append(r)
cols=list(rows[0].keys())
with open("all_tags_block_means.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
# coverage
print("COVERAGE (tags with <300 steps in 1..300):")
for r in rows:
    if r['n_bf16']!=300 or r['n_nvfp4']!=300: print(f"  {r['tag']:60s} bf16 n={r['n_bf16']} nvfp4 n={r['n_nvfp4']}")
print()
print(f"{'tag':58s} {'BF16 b1->b2->b3':38s} {'NVFP4 b1->b2->b3':38s} {'ratio_b3':>9s} {'reldiff':>8s}")
for r in rows:
    flag = '*' if abs(r['reldiff_b3'])>0.15 else ' '
    print(f"{flag}{r['tag']:57s} {r['bf16_shape']:38s} {r['nvfp4_shape']:38s} {r['ratio_b3']:9.3f} {r['reldiff_b3']:8.3f}")
