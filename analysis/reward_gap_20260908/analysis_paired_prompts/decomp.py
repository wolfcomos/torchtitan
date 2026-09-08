import json, collections, re
import numpy as np
exec(open("format.py").read().split("print(f\"{'block':>9}")[0])
LATEX_ENV=re.compile(r"\\boxed\s*\{|answer:\s*\$|answer:\s*\\\(|answer:\s*\\\[", re.I)
print("Unbiased population decomposition on known-truth prompts (truth = value of a reward-1 sample from either run; same prompt set for both runs).")
print("right-rate = mean[k/8 + (8-k)/8 * 1(min-sample completed & value==truth)];  P(reward|right) = reward/right-rate")
print(f"{'block':>9} {'nprompts':>8} | {'BF16 reward':>11} {'right':>6} {'P(rew|right)':>12} {'right&unrew':>11} {'trunc':>6} | {'NVFP4 reward':>12} {'right':>6} {'P(rew|right)':>12} {'right&unrew':>11} {'trunc':>6} | paired diff right (boot95CI)")
rng=np.random.default_rng(0)
for lo,hi in BLOCKS+[(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi and r["truth"]]
    res={}
    for name,key,kk in (("BF16","B","bk"),("NVFP4","N","nk")):
        rew=[];right=[];ru=[];tr=[]
        for r in rs:
            k=r[kk]; m=r[key]["min"]; t=resp(m)
            mr = (k<8) and m["status"]=="completed" and norm(ans_val(t))==r["truth"]
            mt = (k<8) and m["status"]=="truncated_length"
            rew.append(k/8); right.append(k/8+(8-k)/8*mr); ru.append((8-k)/8*mr); tr.append((8-k)/8*mt)
        res[name]=dict(rew=np.array(rew),right=np.array(right),ru=np.array(ru),tr=np.array(tr))
    d=res["NVFP4"]["right"]-res["BF16"]["right"]
    bs=rng.choice(d,(20000,len(d))).mean(1)
    b=res["BF16"]; n=res["NVFP4"]
    print(f"{lo:>4}-{hi:<4} {len(rs):>8} | {b['rew'].mean():>11.3f} {b['right'].mean():>6.3f} {b['rew'].mean()/b['right'].mean():>12.3f} {b['ru'].mean():>11.3f} {b['tr'].mean():>6.3f} | {n['rew'].mean():>12.3f} {n['right'].mean():>6.3f} {n['rew'].mean()/n['right'].mean():>12.3f} {n['ru'].mean():>11.3f} {n['tr'].mean():>6.3f} | {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
# and on ALL prompts (unknown truth counts as not-right: lower bound on right-rate, exact for reward)
print("\nAll 2400 prompts (unknown-truth prompts contribute right=reward=0 -> right-rate is a lower bound):")
for lo,hi in BLOCKS+[(1,300)]:
    rs=[r for r in rows if lo<=r["step"]<=hi]
    out=[]
    for name,key,kk in (("BF16","B","bk"),("NVFP4","N","nk")):
        rew=np.mean([r[kk]/8 for r in rs])
        right=np.mean([r[kk]/8+(8-r[kk])/8*((r[kk]<8) and r[key]["min"]["status"]=="completed" and r["truth"] is not None and norm(ans_val(resp(r[key]["min"])))==r["truth"]) for r in rs])
        tr=np.mean([(8-r[kk])/8*(r[key]["min"]["status"]=="truncated_length") for r in rs])
        out.append((rew,right,tr))
    print(f" {lo:>3}-{hi:<3} BF16 reward={out[0][0]:.3f} right>={out[0][1]:.3f} trunc~{out[0][2]:.3f} | NVFP4 reward={out[1][0]:.3f} right>={out[1][1]:.3f} trunc~{out[1][2]:.3f}")
