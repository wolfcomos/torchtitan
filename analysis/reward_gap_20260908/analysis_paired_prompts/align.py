import json, collections, math
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
def load(run):
    recs=[]
    for i,l in enumerate(open(f"{OUT}/{run}/rollout_samples.jsonl")):
        r=json.loads(l); r["_line"]=i; recs.append(r)
    return recs
B=load("arm_bf16_4x4_long"); N=load("arm_nvfp4_deq_4x4_long")
def pv(r): return r["turns"][0]["max_policy_version"]
def prompt(r): return r["turns"][0]["prompt_messages"][-1]["content"]
# advantage -> k table
def k_from(reward, adv):
    best=None
    for k in range(9):
        p=k/8; sd=math.sqrt(p*(1-p)); a=(reward-p)/(sd+1e-6)
        d=abs(a-adv)
        if best is None or d<best[0]: best=(d,k)
    return best
for name,R in [("BF16",B),("NVFP4",N)]:
    errs=collections.Counter()
    for r in R:
        d,k=k_from(r["reward"],r["advantage"])
        errs[round(d,4)]+=1
    print(name,"adv->k residuals:",sorted(errs.items())[:6], "max", max(errs))
# replays: pvs with >16 records; check whether the two blocks have the same prompts
for name,R in [("BF16",B),("NVFP4",N)]:
    bypv=collections.defaultdict(list)
    for r in R: bypv[pv(r)].append(r)
    print("==",name)
    for v in sorted(bypv):
        rs=bypv[v]
        if len(rs)!=16:
            # split into blocks by line contiguity / group_id resets
            lines=[r["_line"] for r in rs]
            gaps=[j for j in range(1,len(lines)) if lines[j]-lines[j-1]>1]
            blocks=[]; start=0
            for g in gaps+[len(lines)]:
                blocks.append(rs[start:g]); start=g
            ps=[set(prompt(r) for r in b) for b in blocks]
            same = all(p==ps[0] for p in ps)
            print(f"  pv={v} n={len(rs)} blocks={[len(b) for b in blocks]} lines={[ (b[0]['_line'],b[-1]['_line']) for b in blocks]} same_prompts_across_blocks={same} prompts_per_block={[len(p) for p in ps]} gids={[sorted(set(r['group_id'] for r in b)) for b in blocks]}")
# Cross-run prompt alignment by pv (using all records)
bp=collections.defaultdict(set); np_=collections.defaultdict(set)
for r in B: bp[pv(r)].add(prompt(r))
for r in N: np_[pv(r)].add(prompt(r))
match=0; partial=0; nomatch=0; detail=[]
for v in range(0,301):
    b=bp.get(v,set()); n=np_.get(v,set())
    inter=len(b&n)
    if inter==len(b)==len(n)==8: match+=1
    elif inter>0: partial+=1; detail.append((v,len(b),len(n),inter))
    else: nomatch+=1; detail.append((v,len(b),len(n),inter))
print("pv-aligned prompt sets 0..300: full match",match,"partial",partial,"none",nomatch)
print("details:",detail[:40])
# global prompt overlap
allb=set(prompt(r) for r in B); alln=set(prompt(r) for r in N)
print("distinct prompts B",len(allb),"N",len(alln),"intersection",len(allb&alln))
# for prompts in both, pv offset
pb={}; pn={}
for r in B: pb[prompt(r)]=pv(r)
for r in N: pn[prompt(r)]=pv(r)
off=collections.Counter(pn[p]-pb[p] for p in allb&alln)
print("pv offset (N-B) for common prompts:",sorted(off.items())[:20])
