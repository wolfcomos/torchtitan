"""Second pass: where did the two runs' updates land? Overlap of changed-element sets, sign agreement,
magnitude profile of changed vs unchanged elements, and 'update survives bf16 RTNE' bookkeeping."""
import json, sys, time, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from safetensors import safe_open
OUT = "/out/analysis_checkpoint-drift-experiment"
HF = "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/miles_smoke/models/Qwen3-30B-A3B"
RUNS = {"bf16": "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/arm_bf16_4x4_long/parked/step-300",
        "nvfp4": "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/arm_nvfp4_deq_4x4_long/parked/step-300"}
DEV = "cuda"
IDX = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
H = {}
def hf_get(n):
    f = IDX[n]; H.setdefault(f, safe_open(f"{HF}/{f}", framework="pt", device="cpu")); return H[f].get_tensor(n)
def hf_tensor(k):
    if k == "tok_embeddings.weight": return hf_get("model.embed_tokens.weight")
    if k == "lm_head.weight": return hf_get("lm_head.weight")
    if k == "norm.weight": return hf_get("model.norm.weight")
    L = int(k.split(".")[1])
    if k.endswith("w1_EFD"): return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight") for e in range(128)])
    if k.endswith("w2_EDF"): return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight") for e in range(128)])
    if k.endswith("w3_EFD"): return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight") for e in range(128)])
    if k.endswith("attention.wo.weight"): return hf_get(f"model.layers.{L}.self_attn.o_proj.weight")
    if k.endswith("router.gate.weight"): return hf_get(f"model.layers.{L}.mlp.gate.weight")
    if k.endswith("qkv_linear.wq.weight"): return hf_get(f"model.layers.{L}.self_attn.q_proj.weight")
    if k == "tok_embeddings.weight": return hf_get("model.embed_tokens.weight")
    if k == "lm_head.weight": return hf_get("lm_head.weight")
    if k.endswith("attention.q_norm.weight"): return hf_get(f"model.layers.{L}.self_attn.q_norm.weight")
    if k == "norm.weight": return hf_get("model.norm.weight")
    if k.endswith("attention_norm.weight"): return hf_get(f"model.layers.{L}.input_layernorm.weight")
    if k.endswith("ffn_norm.weight"): return hf_get(f"model.layers.{L}.post_attention_layernorm.weight")
    raise KeyError(k)
META = {r: FileSystemReader(d).read_metadata().state_dict_metadata for r, d in RUNS.items()}
def load(run, k):
    m = META[run][k]; sd = {k: torch.empty(tuple(m.size), dtype=m.properties.dtype)}
    dcp.load(sd, checkpoint_id=RUNS[run]); return sd[k]
def qq(x, ps):
    x = x.flatten().float()
    if x.numel() > 16_000_000:
        idx = torch.randint(0, x.numel(), (16_000_000,), generator=torch.Generator(device=x.device).manual_seed(0), device=x.device); x = x[idx]
    return torch.quantile(x, torch.tensor(ps, device=x.device)).tolist()
KEYS = ["lm_head.weight", "tok_embeddings.weight", "layers.0.attention_norm.weight", "layers.39.ffn_norm.weight", "layers.0.attention.q_norm.weight", "norm.weight"]
res = json.load(open(f"{OUT}/overlap_results.json"))
for k in KEYS:
    t0 = time.time()
    w0 = hf_tensor(k).to(DEV); a = load("bf16", k).to(DEV); b = load("nvfp4", k).to(DEV)
    assert w0.shape == a.shape == b.shape, (k, w0.shape, a.shape, b.shape)
    ca = a.view(torch.int16) != w0.view(torch.int16); cb = b.view(torch.int16) != w0.view(torch.int16)
    both = ca & cb
    da = a.float() - w0.float(); db = b.float() - w0.float()
    w0f = w0.float()
    r = {"shape": list(w0.shape), "frac_changed_bf16run": ca.float().mean().item(), "frac_changed_nvfp4run": cb.float().mean().item(),
         "frac_changed_both": both.float().mean().item(),
         "jaccard_changed_sets": (both.float().sum() / (ca | cb).float().sum()).item(),
         "sign_agreement_on_both": ((da.sign() == db.sign())[both]).float().mean().item() if both.any() else None,
         "identical_value_on_both": ((a.view(torch.int16) == b.view(torch.int16))[both]).float().mean().item() if both.any() else None,
         "w0_abs_p50_all": qq(w0f.abs(), [0.5])[0],
         "w0_abs_p50_p90_p99_changed_bf16run": qq(w0f.abs()[ca], [0.5, 0.9, 0.99]) if ca.any() else None,
         "w0_abs_p50_p90_p99_changed_nvfp4run": qq(w0f.abs()[cb], [0.5, 0.9, 0.99]) if cb.any() else None,
         "w0_abs_max_changed_bf16run": w0f.abs()[ca].max().item() if ca.any() else None,
         "w0_abs_max_changed_nvfp4run": w0f.abs()[cb].max().item() if cb.any() else None,
         "frac_w0_abs_below_1e-3": (w0f.abs() < 1e-3).float().mean().item(),
         "frac_changed_among_w0_abs_below_1e-3_bf16run": (ca & (w0f.abs() < 1e-3)).float().sum().item() / max((w0f.abs() < 1e-3).float().sum().item(), 1),
         "frac_changed_among_w0_abs_above_1e-2_bf16run": (ca & (w0f.abs() > 1e-2)).float().sum().item() / max((w0f.abs() > 1e-2).float().sum().item(), 1),
         "rel_fro_drift_bf16run": (da.norm() / w0f.norm()).item(), "rel_fro_drift_nvfp4run": (db.norm() / w0f.norm()).item(),
         "rel_fro_bf16run_vs_nvfp4run": ((a.float() - b.float()).norm() / w0f.norm()).item(),
         "cos_drift_bf16run_vs_nvfp4run": torch.nn.functional.cosine_similarity(da.flatten(), db.flatten(), dim=0).item(),
         "abs_drift_max_bf16run": da.abs().max().item(), "abs_drift_max_nvfp4run": db.abs().max().item(),
         }
    res[k] = r
    print(k, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in r.items() if kk not in ("shape",)}, f"{time.time()-t0:.1f}s", flush=True)
    del w0, a, b, ca, cb, both, da, db, w0f; torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/overlap_results.json", "w"), indent=1)
print("ALL DONE")
