import json, collections, sys
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
for run in ["arm_bf16_4x4_long","arm_nvfp4_deq_4x4_long"]:
    recs=[json.loads(l) for l in open(f"{OUT}/{run}/rollout_samples.jsonl")]
    print("=====",run,"n=",len(recs))
    print("status:",collections.Counter(r["status"] for r in recs))
    print("is_validation:",collections.Counter(r["is_validation"] for r in recs))
    print("nturns:",collections.Counter(len(r["turns"]) for r in recs))
    print("n prompt msgs:",collections.Counter(len(r["turns"][0]["prompt_messages"]) for r in recs))
    print("roles:",collections.Counter(tuple(m["role"] for m in r["turns"][0]["prompt_messages"]) for r in recs))
    print("reward vals:",collections.Counter(r["reward"] for r in recs))
    print("reward_breakdown keys:",collections.Counter(tuple(r["reward_breakdown"].keys()) for r in recs))
    pv=collections.Counter(r["turns"][0]["max_policy_version"] for r in recs)
    mn_ne_mx=sum(1 for r in recs if r["turns"][0]["min_policy_version"]!=r["turns"][0]["max_policy_version"])
    print("min!=max policy version:",mn_ne_mx)
    print("policy versions: min",min(pv),"max",max(pv),"distinct",len(pv))
    print("records per pv hist:",sorted(collections.Counter(pv.values()).items()))
    # missing pvs
    allpv=set(range(min(pv),max(pv)+1)); missing=sorted(allpv-set(pv))
    print("missing pvs:",missing[:50], "count",len(missing))
    # per pv: distinct group ids, prompts
    print("first 12 pvs:")
    for v in sorted(pv)[:12]:
        rs=[r for r in recs if r["turns"][0]["max_policy_version"]==v]
        gids=collections.Counter(r["group_id"] for r in rs)
        prompts=collections.Counter(r["turns"][0]["prompt_messages"][-1]["content"] for r in rs)
        rids=sorted(collections.Counter(r["rollout_id"] for r in rs).items())
        print(f"  pv={v} n={len(rs)} groups={dict(sorted(gids.items()))} nprompts={len(prompts)} rollout_ids={rids}")
    # group ids overall
    print("group_id range:",min(r["group_id"] for r in recs),max(r["group_id"] for r in recs))
    print("rollout_id range:",min(r["rollout_id"] for r in recs),max(r["rollout_id"] for r in recs))
    # duplicates: (pv, group_id, rollout_id)
    keys=collections.Counter((r["turns"][0]["max_policy_version"],r["group_id"],r["rollout_id"]) for r in recs)
    dups={k:c for k,c in keys.items() if c>1}
    print("duplicate (pv,gid,rid) keys:",len(dups), list(dups.items())[:10])
    # exact duplicate records (same prompt+response)?
    full=collections.Counter((r["turns"][0]["max_policy_version"],r["group_id"],r["rollout_id"],r["turns"][0]["completion_message"]["content"][:200]) for r in recs)
    print("exact dup records:",sum(c-1 for c in full.values() if c>1))
    # per pv distinct groups, records per group
    per_pv_groups=collections.Counter(); per_pv_rec_per_group=collections.Counter()
    for v in pv:
        rs=[r for r in recs if r["turns"][0]["max_policy_version"]==v]
        g=collections.Counter(r["group_id"] for r in rs)
        per_pv_groups[len(g)]+=1
        for c in g.values(): per_pv_rec_per_group[c]+=1
    print("distinct groups per pv hist:",sorted(per_pv_groups.items()))
    print("records per (pv,group) hist:",sorted(per_pv_rec_per_group.items()))
    # does the same prompt appear at multiple pvs?
    p2pv=collections.defaultdict(set)
    for r in recs: p2pv[r["turns"][0]["prompt_messages"][-1]["content"]].add(r["turns"][0]["max_policy_version"])
    print("distinct prompts:",len(p2pv)," prompts seen at >1 pv:",sum(1 for s in p2pv.values() if len(s)>1))
    multi=[(p[:80],sorted(s)) for p,s in p2pv.items() if len(s)>1][:8]
    for m in multi: print("   ",m)
    # advantage stats
    import statistics
    adv=[r["advantage"] for r in recs]
    print("advantage nonzero frac:",sum(1 for a in adv if a!=0)/len(adv))
