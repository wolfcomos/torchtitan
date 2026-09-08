import json, collections, re, hashlib
import numpy as np
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
RUNS={"BF16":"arm_bf16_4x4_long","NVFP4":"arm_nvfp4_deq_4x4_long"}
exec(open("paired.py").read().split("# ---- validate")[0])  # reuse canonical() -> C
BOX=re.compile(r"\\boxed\s*\{"); ANSL=re.compile(r"(?im)^\s*(?:#+\s*)?\**\s*(?:final\s+)?answer\**\s*:\s*\**\s*(.+?)\s*\**\s*$")
LATEX_ANS=re.compile(r"(?i)answer:\s*\$")
def ans_val(t):
    m=ANSL.findall(t); 
    if not m: return None
    v=m[-1].strip().strip("$").replace("\\boxed{","").rstrip("}").strip().rstrip(".")
    return v
def norm(v): return re.sub(r"\s+","",v.replace("\\dfrac","\\frac").replace("\\left","").replace("\\right","")) if v else v
BLOCKS=[(1,50),(51,100),(101,150),(151,200),(201,250),(251,300)]
rows=[]
for v in sorted(set(C["BF16"])&set(C["NVFP4"])):
    step=v+1
    if step>300: continue
    for p in C["BF16"][v]:
        b=C["BF16"][v][p]; n=C["NVFP4"][v][p]
        truth=None
        for x in (b["max"],n["max"]):
            if x["reward"]==1: truth=norm(ans_val(resp(x))); break
        rows.append(dict(step=step,truth=truth,B=b,N=n,bk=b["k"],nk=n["k"]))
print(f"{'block':>9} {'run':>5} {'boxed% all':>10} {'boxed% max-samp':>15} {'reward1 n':>9} {'boxed|r=1':>9} {'latex$|r=1':>10} {'neither|r=1':>11} | known-truth r=0 completed: {'n':>4} {'rightval-unboxed':>16} {'rightval-boxed':>14} {'wrongval':>8} {'noans':>6}")
for lo,hi in BLOCKS+[(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi]
    for name,key in [("BF16","B"),("NVFP4","N")]:
        samples=[x for r in rs for x in (r[key]["min"],r[key]["max"])]
        boxed_all=np.mean([bool(BOX.search(resp(x))) for x in samples])
        boxed_max=np.mean([bool(BOX.search(resp(r[key]["max"]))) for r in rs])
        r1=[x for x in samples if x["reward"]==1]
        b1=sum(bool(BOX.search(resp(x))) for x in r1); l1=sum((not BOX.search(resp(x))) and bool(LATEX_ANS.search(resp(x))) for x in r1); n1=len(r1)-b1-l1
        # known truth, reward 0, completed
        kt=[(r,x) for r in rs if r["truth"] for x in (r[key]["min"],r[key]["max"]) if x["reward"]==0 and x["status"]=="completed"]
        ru=rb=wv=na=0
        for r,x in kt:
            t=resp(x); val=norm(ans_val(t))
            if val is None: na+=1
            elif val==r["truth"]:
                if BOX.search(t): rb+=1
                else: ru+=1
            else: wv+=1
        print(f"{lo:>4}-{hi:<4} {name:>5} {boxed_all:>9.1%} {boxed_max:>14.1%} {len(r1):>9} {b1/len(r1):>9.1%} {l1/len(r1):>10.1%} {n1/len(r1):>11.1%} | {len(kt):>28} {ru:>8} {ru/len(kt):>6.1%} {rb:>6} {rb/len(kt):>6.1%} {wv:>8} {na:>6}")
# discordant prompts: B>=1 & N=0 -> NVFP4 reward-0 completed samples: right value unboxed?
print("\n=== on prompts B>=1 & N=0 (truth known from BF16): NVFP4's two recorded samples ===")
for lo,hi in BLOCKS+[(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi and r["bk"]>=1 and r["nk"]==0 and r["truth"]]
    c=collections.Counter()
    for r in rs:
        for x in (r["N"]["min"],r["N"]["max"]):
            t=resp(x)
            if x["status"]!="completed": c["truncated"]+=1; continue
            val=norm(ans_val(t))
            if val is None: c["no Answer line"]+=1
            elif val==r["truth"]: c["RIGHT value, unextractable (no boxed/latex)"]+=1
            else: c["wrong value"]+=1
    tot=sum(c.values())
    print(f" {lo}-{hi}: n_samples={tot} "+", ".join(f"{k}={v} ({v/tot:.0%})" for k,v in sorted(c.items(),key=lambda x:-x[1])))
print("\n=== on prompts N>=1 & B=0 (truth from NVFP4): BF16's two recorded samples ===")
for lo,hi in [(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi and r["nk"]>=1 and r["bk"]==0 and r["truth"]]
    c=collections.Counter()
    for r in rs:
        for x in (r["B"]["min"],r["B"]["max"]):
            t=resp(x)
            if x["status"]!="completed": c["truncated"]+=1; continue
            val=norm(ans_val(t))
            if val is None: c["no Answer line"]+=1
            elif val==r["truth"]: c["RIGHT value, unextractable (no boxed/latex)"]+=1
            else: c["wrong value"]+=1
    tot=sum(c.values())
    print(f" {lo}-{hi}: n_samples={tot} "+", ".join(f"{k}={v} ({v/tot:.0%})" for k,v in sorted(c.items(),key=lambda x:-x[1])))
# boxed rate over time per 25 steps for the trend
print("\n=== boxed rate of recorded samples per 25-step window ===")
for lo in range(1,301,25):
    hi=lo+24; rs=[r for r in rows if lo<=r["step"]<=hi]
    out=[]
    for key in ("B","N"):
        samples=[x for r in rs for x in (r[key]["min"],r[key]["max"])]
        out.append(np.mean([bool(BOX.search(resp(x))) for x in samples]))
    print(f" {lo:>3}-{hi:<3} BF16 {out[0]:.1%}  NVFP4 {out[1]:.1%}")
