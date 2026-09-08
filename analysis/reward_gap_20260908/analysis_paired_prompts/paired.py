import json, collections, math, re, zlib, hashlib, csv, random
import numpy as np
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
RUNS={"BF16":"arm_bf16_4x4_long","NVFP4":"arm_nvfp4_deq_4x4_long"}
MAXSTEP=300
def pv(r): return r["turns"][0]["max_policy_version"]
def prompt(r): return r["turns"][0]["prompt_messages"][-1]["content"]
def resp(r): return (r["turns"][0]["completion_message"] or {}).get("content") or ""
def k_from(reward, adv):
    best=None
    for k in range(9):
        p=k/8; sd=math.sqrt(p*(1-p)); a=(reward-p)/(sd+1e-6); d=abs(a-adv)
        if best is None or d<best[0]: best=(d,k)
    assert best[0]<1e-3, (reward,adv,best)
    return best[1]

def canonical(run):
    """Return dict pv -> dict prompt -> {'k':int,'recs':[min_rec,max_rec]} using the LAST complete block per pv."""
    recs=[]
    for i,l in enumerate(open(f"{OUT}/{run}/rollout_samples.jsonl")):
        r=json.loads(l); r["_line"]=i; recs.append(r)
    bypv=collections.defaultdict(list)
    for r in recs: bypv[pv(r)].append(r)
    out={}; dropped=collections.Counter()
    for v,rs in bypv.items():
        # split into contiguous line blocks
        blocks=[[rs[0]]]
        for a,b in zip(rs,rs[1:]):
            if b["_line"]-a["_line"]>1: blocks.append([b])
            else: blocks[-1].append(b)
        complete=[b for b in blocks if len(set(r["group_id"] for r in b))==8 and len(b)==16]
        if not complete:
            dropped[v]=len(rs); continue
        if len(blocks)>1: dropped[("replayed",v)]=len(blocks)
        b=complete[-1]
        groups=collections.defaultdict(list)
        for r in b: groups[r["group_id"]].append(r)
        d={}
        for g,rr in groups.items():
            assert len(rr)==2
            ks={k_from(x["reward"],x["advantage"]) for x in rr}
            assert len(ks)==1, (v,g,ks)
            k=ks.pop()
            rr=sorted(rr,key=lambda x:x["reward"])
            p=prompt(rr[0]); assert prompt(rr[1])==p
            d[p]={"k":k,"min":rr[0],"max":rr[1],"gid":g}
        out[v]=d
    return out, dropped

C={}
for name,run in RUNS.items():
    C[name],dropped=canonical(run)
    print(f"{name}: canonical pvs={len(C[name])} min={min(C[name])} max={max(C[name])}; pvs with multiple blocks={sum(1 for k in dropped if isinstance(k,tuple))}; pvs with no complete block={[k for k in dropped if not isinstance(k,tuple)]}")

# ---- validate against tfevents: reward mean per step
tb=json.load(open("tb_series.json"))
for name,run in RUNS.items():
    s=tb[run]["rollout_reward/_mean"]; z=tb[run]["rollout_reward/group_zero_std_frac/mean"]
    for off in (0,1,-1):
        diffs=[]; zd=[]
        for v,d in C[name].items():
            st=v+off
            if str(st) in s and 1<=st<=MAXSTEP:
                diffs.append(abs(sum(x["k"] for x in d.values())/64 - s[str(st)]))
                zf=sum(1 for x in d.values() if x["k"] in (0,8))/8
                zd.append(abs(zf-z[str(st)]))
        print(f"  {name} offset step=pv+{off}: n={len(diffs)} reward mean-abs-diff={np.mean(diffs):.5f} max={np.max(diffs):.5f}; zero-std mean-abs-diff={np.mean(zd):.5f} max={np.max(zd):.5f}")

OFF=1  # set after inspecting the above; step = pv + OFF
# ---- paired table
rows=[]
for v in sorted(set(C["BF16"])&set(C["NVFP4"])):
    step=v+OFF
    if step>MAXSTEP or step<1: continue
    b=C["BF16"][v]; n=C["NVFP4"][v]
    common=set(b)&set(n)
    assert len(common)==8, (v,len(b),len(n),len(common))
    for p in sorted(common):
        rows.append(dict(step=step, pv=v, prompt_sha1=hashlib.sha1(p.encode()).hexdigest()[:12],
            prompt_head=p.split("\n\n",1)[1][:80].replace("\n"," ") if "\n\n" in p else p[:80],
            bf16_correct=b[p]["k"], nvfp4_correct=n[p]["k"], n=8,
            bf16_min_status=b[p]["min"]["status"], bf16_max_status=b[p]["max"]["status"],
            nvfp4_min_status=n[p]["min"]["status"], nvfp4_max_status=n[p]["max"]["status"],
            bf16_min_len=len(resp(b[p]["min"])), nvfp4_min_len=len(resp(n[p]["min"]))))
print("paired rows:",len(rows),"steps:",len(set(r["step"] for r in rows)), "step range", min(r["step"] for r in rows), max(r["step"] for r in rows))
with open("paired_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- statistics helpers
def sign_test(d):
    pos=sum(1 for x in d if x>0); neg=sum(1 for x in d if x<0); n=pos+neg
    if n==0: return pos,neg,1.0
    k=min(pos,neg); p=sum(math.comb(n,i) for i in range(0,k+1))*2/2**n
    return pos,neg,min(1.0,p)
def wilcoxon(d):
    d=np.array([x for x in d if x!=0],dtype=float)
    n=len(d)
    if n<10: return n,float('nan'),float('nan')
    absd=np.abs(d); order=absd.argsort(); ranks=np.empty(n)
    # average ranks for ties
    sorted_abs=absd[order]; i=0; r=np.empty(n)
    while i<n:
        j=i
        while j+1<n and sorted_abs[j+1]==sorted_abs[i]: j+=1
        r[i:j+1]=(i+j)/2+1; i=j+1
    ranks[order]=r
    Wp=ranks[d>0].sum(); Wm=ranks[d<0].sum(); W=min(Wp,Wm)
    # tie correction
    _,counts=np.unique(absd,return_counts=True)
    tie=sum(c**3-c for c in counts)
    mu=n*(n+1)/4; sig=math.sqrt(n*(n+1)*(2*n+1)/24 - tie/48)
    z=(W-mu)/sig; p=math.erfc(abs(z)/math.sqrt(2))
    return n,z,p
def boot_ci(d,B=20000,seed=0):
    rng=np.random.default_rng(seed); d=np.array(d,dtype=float)
    m=rng.choice(d,(B,len(d)),replace=True).mean(1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)

BLOCKS=[(1,50),(51,100),(101,150),(151,200),(201,250),(251,300)]
EARLY=[(1,10),(1,30),(1,300)]
print("\n=== (1) paired accuracy by block (reward = k/8 on the SAME prompts) ===")
print(f"{'block':>9} {'nprompts':>8} {'BF16':>7} {'NVFP4':>7} {'diff':>7} {'boot95CI':>18} {'sign +/-':>10} {'sign p':>8} {'wilcoxon n,z,p':>24}")
for lo,hi in BLOCKS+EARLY:
    rs=[r for r in rows if lo<=r["step"]<=hi]
    b=np.array([r["bf16_correct"] for r in rs])/8; n=np.array([r["nvfp4_correct"] for r in rs])/8
    d=n-b; m,l,u=boot_ci(d); pos,neg,sp=sign_test(d); wn,wz,wp=wilcoxon(d)
    print(f"{lo:>4}-{hi:<4} {len(rs):>8} {b.mean():>7.4f} {n.mean():>7.4f} {m:>+7.4f} [{l:>+7.4f},{u:>+7.4f}] {pos:>4}/{neg:<4} {sp:>8.2e} {wn:>6} {wz:>+7.2f} {wp:>8.2e}")

print("\n=== (2) discordance: BF16>=1 correct & NVFP4=0  vs  NVFP4>=1 & BF16=0 ; both>=1 ; both=0 ===")
print(f"{'block':>9} {'n':>5} {'B>0&N=0':>9} {'N>0&B=0':>9} {'both>0':>8} {'both=0':>8} {'| among both>0: B>N':>20} {'N>B':>5} {'tie':>5}")
for lo,hi in BLOCKS+EARLY:
    rs=[r for r in rows if lo<=r["step"]<=hi]; n=len(rs)
    bn=sum(1 for r in rs if r["bf16_correct"]>0 and r["nvfp4_correct"]==0)
    nb=sum(1 for r in rs if r["nvfp4_correct"]>0 and r["bf16_correct"]==0)
    both=[r for r in rs if r["bf16_correct"]>0 and r["nvfp4_correct"]>0]
    zero=sum(1 for r in rs if r["bf16_correct"]==0 and r["nvfp4_correct"]==0)
    bgt=sum(1 for r in both if r["bf16_correct"]>r["nvfp4_correct"]); ngt=sum(1 for r in both if r["nvfp4_correct"]>r["bf16_correct"]); tie=len(both)-bgt-ngt
    print(f"{lo:>4}-{hi:<4} {n:>5} {bn:>5} {bn/n:>3.0%} {nb:>5} {nb/n:>3.0%} {len(both):>4} {len(both)/n:>3.0%} {zero:>4} {zero/n:>3.0%} | {bgt:>18} {ngt:>5} {tie:>5}")

print("\n=== (3) earliest steps: per-step paired reward (steps 1-30) ===")
for st in range(1,31):
    rs=[r for r in rows if r["step"]==st]
    if not rs: continue
    b=sum(r["bf16_correct"] for r in rs); n=sum(r["nvfp4_correct"] for r in rs)
    print(f" step {st:>3}: BF16 {b:>2}/64={b/64:.3f}  NVFP4 {n:>2}/64={n/64:.3f}  diff {n-b:+3d}   per-prompt k pairs (B,N): {[(r['bf16_correct'],r['nvfp4_correct']) for r in rs]}")
for lo,hi in [(1,5),(1,10),(11,20),(21,30),(1,30)]:
    rs=[r for r in rows if lo<=r["step"]<=hi]
    b=sum(r["bf16_correct"] for r in rs); n=sum(r["nvfp4_correct"] for r in rs); tot=8*len(rs)
    d=np.array([r["nvfp4_correct"]-r["bf16_correct"] for r in rs])/8; m,l,u=boot_ci(d); pos,neg,sp=sign_test(d)
    print(f" steps {lo}-{hi}: BF16 {b}/{tot}={b/tot:.4f} NVFP4 {n}/{tot}={n/tot:.4f} diff {m:+.4f} CI[{l:+.4f},{u:+.4f}] sign +{pos}/-{neg} p={sp:.3f}")

# ---- (5) k histograms per block
print("\n=== (5) per-group correct-count histogram (k/8) per block ===")
hdr=" ".join(f"k={k}" for k in range(9))
print(f"{'block':>9} {'run':>5} {hdr}   zero_std=(k0+k8)/n  k0frac  k8frac")
for lo,hi in BLOCKS+[(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi]; n=len(rs)
    for name,key in [("BF16","bf16_correct"),("NVFP4","nvfp4_correct")]:
        h=collections.Counter(r[key] for r in rs)
        row=" ".join(f"{h[k]:>3}" for k in range(9))
        print(f"{lo:>4}-{hi:<4} {name:>5} {row}   {(h[0]+h[8])/n:.3f}   {h[0]/n:.3f}  {h[8]/n:.3f}")

# ---- (6) staleness
print("\n=== (6) policy-version staleness ===")
for name,run in RUNS.items():
    t=tb[run]
    mpv=t["rollout/max_policy_version/max"]; minpv=t["rollout/min_policy_version/min"]; tpv=t["trainer/policy_version"]
    age=t["train_batch/policy_age/mean"]; agemax=t["train_batch/policy_age_max"]
    steps=[s for s in range(1,MAXSTEP+1) if str(s) in mpv]
    d1=collections.Counter(int(mpv[str(s)])-s for s in steps)
    d2=collections.Counter(int(mpv[str(s)])-int(minpv[str(s)]) for s in steps if str(s) in minpv)
    d3=collections.Counter(int(tpv[str(s)])-s for s in steps if str(s) in tpv)
    print(f" {name}: tfevents rollout/max_policy_version/max - step: {sorted(d1.items())}; max-min pv: {sorted(d2.items())}; trainer/policy_version - step: {sorted(d3.items())}")
    print(f"        train_batch/policy_age/mean: distinct={sorted(collections.Counter(round(age[str(s)],3) for s in steps if str(s) in age).items())[:6]}  policy_age_max: {sorted(collections.Counter(agemax[str(s)] for s in steps if str(s) in agemax).items())[:6]}")
    # jsonl: min==max for all
    recs=[json.loads(l) for l in open(f"{OUT}/{run}/rollout_samples.jsonl")]
    print(f"        jsonl: records with min_policy_version!=max_policy_version: {sum(1 for r in recs if r['turns'][0]['min_policy_version']!=r['turns'][0]['max_policy_version'])}/{len(recs)}")

# ---- (4) failure modes
ANS=re.compile(r"(?m)^\s*\**Answer\**\s*:\s*\S"); BOX=re.compile(r"\\boxed\s*\{")
def ngram_dup_frac(text,n=12):
    w=text.split()
    if len(w)<n+1: return 0.0
    grams=[" ".join(w[i:i+n]) for i in range(len(w)-n+1)]
    c=collections.Counter(grams)
    return sum(v for v in c.values() if v>1)/len(grams)
def compress_ratio(text):
    b=text.encode(); 
    return len(zlib.compress(b,9))/max(1,len(b))
def classify(rec):
    t=resp(rec); st=rec["status"]
    f=dict(truncated=(st=="truncated_length"), has_answer=bool(ANS.search(t)), has_boxed=bool(BOX.search(t)),
           dup12=ngram_dup_frac(t), cr=compress_ratio(t), nchar=len(t))
    if f["truncated"]:
        mode="truncated@limit"+("+repetitive" if f["dup12"]>0.3 else "")
    elif not (f["has_answer"] or f["has_boxed"]):
        mode="completed_no_final_answer"
    else:
        mode="wrong_well_formed"
    f["mode"]=mode
    return f
def modes_of(samples,label):
    fs=[classify(r) for r in samples]
    c=collections.Counter(f["mode"] for f in fs)
    print(f"  {label}: n={len(fs)} modes={dict(sorted(c.items(), key=lambda x:-x[1]))}")
    if fs:
        print(f"     truncated={sum(f['truncated'] for f in fs)/len(fs):.2%} dup12>0.3={sum(f['dup12']>0.3 for f in fs)/len(fs):.2%} dup12>0.1={sum(f['dup12']>0.1 for f in fs)/len(fs):.2%} median dup12={np.median([f['dup12'] for f in fs]):.3f} median compress_ratio={np.median([f['cr'] for f in fs]):.3f} median nchar={np.median([f['nchar'] for f in fs]):.0f}")
    return fs
print("\n=== (4) failure modes (reward-0 recorded samples; each group logs its min- and max-reward sample) ===")
for lo,hi in BLOCKS+[(1,300)]:
    print(f"-- block {lo}-{hi}")
    vs=[r["pv"] for r in rows if lo<=r["step"]<=hi]
    # target: prompts BF16 solves (k>=1) and NVFP4 k=0 -> both NVFP4 recs are reward 0
    tgtN=[]; tgtB_zero=[]; allN0=[]; allB0=[]; revB=[]
    for r in rows:
        if not (lo<=r["step"]<=hi): continue
        v=r["pv"]; p=[q for q in C["BF16"][v] if hashlib.sha1(q.encode()).hexdigest()[:12]==r["prompt_sha1"]][0]
        bn=C["BF16"][v][p]; nn=C["NVFP4"][v][p]
        if r["bf16_correct"]>=1 and r["nvfp4_correct"]==0:
            tgtN += [nn["min"],nn["max"]]
            if bn["min"]["reward"]==0: tgtB_zero.append(bn["min"])
        if r["nvfp4_correct"]>=1 and r["bf16_correct"]==0:
            revB += [bn["min"],bn["max"]]
        for x in (nn["min"],nn["max"]):
            if x["reward"]==0: allN0.append(x)
        for x in (bn["min"],bn["max"]):
            if x["reward"]==0: allB0.append(x)
    modes_of(tgtN,"NVFP4 reward-0 samples on prompts BF16 solved (B>=1,N=0)")
    modes_of(tgtB_zero,"BF16 reward-0 (min) sample on those SAME prompts")
    modes_of(revB,"BF16 reward-0 samples on prompts NVFP4 solved (N>=1,B=0)")
    modes_of(allN0,"ALL NVFP4 reward-0 recorded samples")
    modes_of(allB0,"ALL BF16 reward-0 recorded samples")
# status/truncation overall from canonical recs
print("\n=== truncation rate from tfevents rollout/truncation_rate/mean by block ===")
for name,run in RUNS.items():
    tr=tb[run]["rollout/truncation_rate/mean"]; rl=tb[run]["rollout/response_length/mean"]
    print(" ",name, " ".join(f"{lo}-{hi}:{np.mean([tr[str(s)] for s in range(lo,hi+1) if str(s) in tr]):.3f}" for lo,hi in BLOCKS), "| resp_len", " ".join(f"{np.mean([rl[str(s)] for s in range(lo,hi+1) if str(s) in rl]):.0f}" for lo,hi in BLOCKS))

# save excerpts for (4)
ex=[]
for r in rows:
    if r["bf16_correct"]>=1 and r["nvfp4_correct"]==0:
        v=r["pv"]; p=[q for q in C["BF16"][v] if hashlib.sha1(q.encode()).hexdigest()[:12]==r["prompt_sha1"]][0]
        nn=C["NVFP4"][v][p]; bn=C["BF16"][v][p]
        for x in (nn["min"],nn["max"]):
            f=classify(x)
            ex.append(dict(step=r["step"],prompt_sha1=r["prompt_sha1"],mode=f["mode"],dup12=round(f["dup12"],3),cr=round(f["cr"],3),status=x["status"],bk=r["bf16_correct"],
                           prompt=p.split("\n\n",1)[1][:300] if "\n\n" in p else p[:300], tail=resp(x)[-400:], bf16_ok_tail=resp(bn["max"])[-200:]))
json.dump(ex,open("nvfp4_failures_on_bf16_solved.json","w"),indent=1)
print("saved",len(ex),"excerpt records")
