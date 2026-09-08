import json, collections, re
import numpy as np
exec(open("format.py").read().split("print(f\"{'block':>9}")[0])
LATEX_ENV=re.compile(r"\\boxed\s*\{|answer:\s*\$|answer:\s*\\\(|answer:\s*\\\[", re.I)
print("=== (A) boxed rate among k=0 groups' recorded samples (reward-independent selection) ===")
for lo,hi in [(1,5),(1,10),(11,30)]+BLOCKS+[(1,300)]:
    out=[]
    for key,kk in (("B","bk"),("N","nk")):
        rs=[r for r in rows if lo<=r["step"]<=hi and r[kk]==0]
        s=[x for r in rs for x in (r[key]["min"],r[key]["max"])]
        out.append((len(s), np.mean([bool(BOX.search(resp(x))) for x in s]), np.mean([x["status"]=="truncated_length" for x in s])))
    print(f" {lo:>3}-{hi:<3} BF16 n={out[0][0]:>4} boxed={out[0][1]:.1%} trunc={out[0][2]:.1%} | NVFP4 n={out[1][0]:>4} boxed={out[1][1]:.1%} trunc={out[1][2]:.1%}")
print("\n=== (B) decomposition on known-truth prompts, completed samples with an Answer line ===")
print(f"{'block':>9} {'run':>5} {'n':>5} {'right-value':>11} {'boxed/latex':>11} {'right&boxed':>11} {'right&unboxed':>13} {'wrong&boxed':>11}  reward1-rate-among-these")
for lo,hi in BLOCKS+[(1,300)]:
    for name,key in (("BF16","B"),("NVFP4","N")):
        rs=[r for r in rows if lo<=r["step"]<=hi and r["truth"]]
        s=[(r,x) for r in rs for x in (r[key]["min"],r[key]["max"]) if x["status"]=="completed"]
        s=[(r,x,norm(ans_val(resp(x)))) for r,x in s]; s=[(r,x,v) for r,x,v in s if v is not None]
        n=len(s); right=[v==r["truth"] for r,x,v in s]; boxed=[bool(LATEX_ENV.search(resp(x))) for r,x,v in s]
        rb=sum(a and b for a,b in zip(right,boxed)); ru=sum(a and not b for a,b in zip(right,boxed)); wb=sum((not a) and b for a,b in zip(right,boxed))
        r1=np.mean([x["reward"] for r,x,v in s])
        print(f"{lo:>4}-{hi:<4} {name:>5} {n:>5} {np.mean(right):>11.1%} {np.mean(boxed):>11.1%} {rb/n:>11.1%} {ru/n:>13.1%} {wb/n:>11.1%}  {r1:.3f}")
# unbiased version: min-samples only when k<8 are always reward 0 -> biased toward wrong. Use k=0 groups: both samples reward 0.
print("\n=== (C) same decomposition restricted to k=0 groups (both recorded samples reward 0) on known-truth prompts ===")
for lo,hi in [(1,300)]:
    for name,key,kk in (("BF16","B","bk"),("NVFP4","N","nk")):
        rs=[r for r in rows if lo<=r["step"]<=hi and r["truth"] and r[kk]==0]
        s=[(r,x) for r in rs for x in (r[key]["min"],r[key]["max"]) if x["status"]=="completed"]
        s=[(r,x,norm(ans_val(resp(x)))) for r,x in s]; s=[(r,x,v) for r,x,v in s if v is not None]
        n=len(s); right=sum(v==r["truth"] for r,x,v in s); boxed=sum(bool(LATEX_ENV.search(resp(x))) for r,x,v in s)
        print(f" {name}: n={n} right-value(unrewarded)={right} ({right/n:.1%}) boxed/latex={boxed} ({boxed/n:.1%})")
