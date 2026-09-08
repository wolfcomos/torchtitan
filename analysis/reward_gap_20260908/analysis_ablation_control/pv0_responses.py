import json, hashlib, collections
BASE="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
runs=["arm_bf16","arm_nvfp4_deq","arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long","arm_nvfp4_deq_4x4_100","arm_nvfp4_hp","arm_mxfp8_deq"]
def load(r, pvmax=2):
    out=collections.defaultdict(list)
    try: f=open(f"{BASE}/{r}/rollout_samples.jsonl")
    except FileNotFoundError: return out
    for line in f:
        try:o=json.loads(line)
        except Exception: continue
        if o.get("is_validation"): continue
        t=o["turns"][0]; pv=t["min_policy_version"]
        if pv>pvmax: continue
        ph=hashlib.md5(json.dumps(t["prompt_messages"],sort_keys=True).encode()).hexdigest()[:8]
        cm=t.get("completion_message"); resp=json.dumps(cm,sort_keys=True)
        rh=hashlib.md5(resp.encode()).hexdigest()[:8]
        out[pv].append((ph,rh,o["reward"],o["rollout_id"],o["group_id"],len(resp)))
    return out
D={r:load(r) for r in runs}
for r in runs:
    for pv in sorted(D[r]):
        recs=D[r][pv]
        print(f"{r} pv{pv}: n={len(recs)} prompts={len(set(x[0] for x in recs))} reward_sum={sum(x[2] for x in recs):.0f} rollout_ids={sorted(x[3] for x in recs)[:6]}...")
print("\n### pv0 response identity across runs (same prompt+response hash pairs)")
def pairs(r,pv=0): return set((x[0],x[1]) for x in D[r][pv])
for a,b in [("arm_bf16","arm_bf16_4x4_long"),("arm_nvfp4_deq","arm_nvfp4_deq_4x4_long"),("arm_nvfp4_deq","arm_nvfp4_deq_4x4_100"),("arm_nvfp4_deq_4x4_long","arm_nvfp4_deq_4x4_100"),("arm_bf16","arm_nvfp4_deq")]:
    if D[a] and D[b]:
        print(f"{a} vs {b}: shared (prompt,response) pairs at pv0 = {len(pairs(a)&pairs(b))} of {len(pairs(a))}/{len(pairs(b))}; same rollout_id sets: {sorted(x[3] for x in D[a][0])==sorted(x[3] for x in D[b][0])}")
# which subset is logged? rollout_ids
print("\n### rewards of the logged subset at pv0 per run")
for r in runs:
    if D[r]: print(r, [x[2] for x in sorted(D[r][0],key=lambda x:x[3])])
