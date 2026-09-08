import json, hashlib, collections
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
OUT="/tmp/claude-147915/-home-hanlinb/87c5c385-2451-4aa9-92a6-6e561d466781/scratchpad/analysis_ablation_control"
runs=["arm_bf16","arm_nvfp4_deq","arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]
MAXPV=12
res={}
for r in runs:
    pv2prompts=collections.defaultdict(set); pv2n=collections.Counter(); pv2rewards=collections.defaultdict(list); nval=0; n=0
    for line in open(f"{BASE}/{r}/rollout_samples.jsonl"):
        try: o=json.loads(line)
        except Exception: continue
        n+=1
        if o.get("is_validation"): nval+=1; continue
        t=o["turns"][0]; pv=t["min_policy_version"]
        if pv>MAXPV: continue
        pm=t["prompt_messages"]; txt=json.dumps(pm,sort_keys=True)
        h=hashlib.md5(txt.encode()).hexdigest()[:10]
        pv2prompts[pv].add(h); pv2n[pv]+=1; pv2rewards[pv].append(o["reward"])
    res[r]=(pv2prompts,pv2n,pv2rewards)
    print(f"== {r}: lines={n} validation={nval}; train samples per pv (pv<= {MAXPV}):", dict(sorted(pv2n.items())))
    print("   distinct prompts per pv:", {k:len(v) for k,v in sorted(pv2prompts.items())})
print("\n### prompt-set overlap by policy version (pv = step-1 presumably: pv0 rollouts train step 1)")
print("pv | ABLbf16∩ABLnvfp4 | LONGbf16∩LONGnvfp4 | ABLbf16∩LONGbf16 | ABLnvfp4∩LONGnvfp4 | sizes(ab,an,lb,ln)")
for pv in range(0,MAXPV+1):
    a=res["arm_bf16"][0].get(pv,set()); b=res["arm_nvfp4_deq"][0].get(pv,set()); c=res["arm_bf16_4x4_long"][0].get(pv,set()); d=res["arm_nvfp4_deq_4x4_long"][0].get(pv,set())
    if not (a or b or c or d): continue
    print(f"{pv:2d} | {len(a&b):2d} | {len(c&d):2d} | {len(a&c):2d} | {len(b&d):2d} | ({len(a)},{len(b)},{len(c)},{len(d)})")
# do ablation prompts at pv k appear in long run at any pv?
print("\n### ablation prompt -> which pv in long run (union of long runs)")
longmap={}
for r in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
    for pv,hs in res[r][0].items():
        for h in hs: longmap.setdefault(h,set()).add((r[4:8],pv))
for r in ["arm_bf16","arm_nvfp4_deq"]:
    for pv in sorted(res[r][0]):
        hits=[sorted(longmap.get(h,set())) for h in res[r][0][pv]]
        found=sum(1 for x in hits if x)
        pvs=sorted(set(p for x in hits for (_,p) in x))
        print(f"{r} pv{pv}: {found}/{len(res[r][0][pv])} prompts found in long runs at pvs {pvs}")
