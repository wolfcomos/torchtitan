import json, collections, re, hashlib, random
import numpy as np
OUT="/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs"
RUNS={"BF16":"arm_bf16_4x4_long","NVFP4":"arm_nvfp4_deq_4x4_long"}
def pv(r): return r["turns"][0]["max_policy_version"]
def prompt(r): return r["turns"][0]["prompt_messages"][-1]["content"]
def resp(r): return (r["turns"][0]["completion_message"] or {}).get("content") or ""
R={}
for n,run in RUNS.items():
    R[n]=[json.loads(l) for l in open(f"{OUT}/{run}/rollout_samples.jsonl")]
# pv=0..3 identity by (prompt, rollout_id)
print("=== response identity across runs at early pvs, keyed (prompt, rollout_id) ===")
for v in range(0,4):
    b={(prompt(r),r["rollout_id"]):resp(r) for r in R["BF16"] if pv(r)==v}
    n={(prompt(r),r["rollout_id"]):resp(r) for r in R["NVFP4"] if pv(r)==v}
    common=set(b)&set(n)
    same=sum(1 for k in common if b[k]==n[k])
    # common prefix length
    pref=[len(__import__('os').path.commonprefix([b[k],n[k]])) for k in common]
    print(f" pv={v}: common keys={len(common)} byte-identical={same} median common-prefix chars={np.median(pref) if pref else None} max={max(pref) if pref else None}")
# in-group answer consistency for k=0 groups: do the two recorded (both wrong) samples give the same final answer?
ANS=re.compile(r"(?m)^\s*\**Answer\**\s*:\s*(.+?)\s*$")
def final(t):
    m=ANS.findall(t)
    return m[-1].strip().strip("$").strip() if m else None
print("\n=== in-group wrong-answer agreement for k=0 groups (both recorded samples reward 0, both have an Answer: line) ===")
for n,recs in R.items():
    groups=collections.defaultdict(list)
    for r in recs: groups[(pv(r),r["group_id"],r["_line"]//10**9 if "_line" in r else 0)].append(r)
    agree=disagree=0
    for k,rr in groups.items():
        if len(rr)!=2 or any(x["reward"]!=0 for x in rr): continue
        a=[final(resp(x)) for x in rr]
        if None in a: continue
        if a[0]==a[1]: agree+=1
        else: disagree+=1
    print(f" {n}: groups with two wrong answered samples={agree+disagree}, same final answer={agree} ({agree/(agree+disagree):.1%})")
# excerpts
ex=json.load(open("nvfp4_failures_on_bf16_solved.json"))
random.seed(1)
print("\n=== excerpts: NVFP4 reward-0 samples on prompts BF16 solved ===")
for mode in ["wrong_well_formed","truncated@limit","completed_no_final_answer","truncated@limit+repetitive"]:
    cands=[e for e in ex if e["mode"]==mode]
    print(f"--- mode={mode} n={len(cands)}")
    for e in random.sample(cands,min(2,len(cands))):
        print(f" step {e['step']} bf16_k={e['bk']} status={e['status']} dup12={e['dup12']} cr={e['cr']}")
        print("   PROMPT:", e["prompt"][:200].replace("\n"," "))
        print("   NVFP4 tail:", repr(e["tail"][-280:]))
        print("   BF16 correct tail:", repr(e["bf16_ok_tail"][-120:]))
