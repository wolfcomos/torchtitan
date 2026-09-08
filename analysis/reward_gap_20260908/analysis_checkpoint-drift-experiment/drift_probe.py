"""Checkpoint drift probe: HF base (w0) vs step-300 DCP weights of the BF16 and NVFP4 runs.

Read-only on all inputs. Writes results JSON + per-tensor stats to /out/analysis_checkpoint-drift-experiment/.
"""
import os, sys, json, time, traceback
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from safetensors import safe_open

OUT = "/out/analysis_checkpoint-drift-experiment"
HF = "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/miles_smoke/models/Qwen3-30B-A3B"
RUNS = {
    "bf16": "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/arm_bf16_4x4_long/parked/step-300",
    "nvfp4": "/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation/outputs/arm_nvfp4_deq_4x4_long/parked/step-300",
}
SMOKE = os.environ.get("SMOKE", "0") == "1"
LAYERS = [0, 20, 39, 44] if not SMOKE else [20]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, "torch", torch.__version__, flush=True)

# ---------------- quantizer import (pure torch path; no CuTe DSL) ----------------
sys.path.insert(0, "/home/scratch.hanlinb_ent_1/repos/ao-nvfp4")
QUANT_SRC = None
try:
    import torchao.prototype.moe_training.nvfp4_training.four_over_six as fos
    import torchao.prototype.moe_training.nvfp4_training.four_over_six_grouped as fosg
    fos._cutedsl_quantize_eligible = lambda x: False  # force the pure-PyTorch reference body
    QUANT_SRC = "torchao ao-nvfp4 four_over_six.py::four_over_six_quantize (pure torch, cutedsl fast path disabled) via four_over_six_grouped._quantize_expert_weights"
    print("quantizer import OK", flush=True)
except Exception as e:
    print("torchao import FAILED:", repr(e), flush=True)
    traceback.print_exc()
    fos = fosg = None

# arm settings (ablation_arms.py:126-135): weight_block="1x16", err_mode="mse", e4m3_scale_bound=256, row_scaled_activation=True
WEIGHT_BLOCK, ERR_MODE, BOUND = "1x16", "mse", 256

def quantize_expert_stack(w_efd_bf16):
    """(E,N,K) bf16 -> codes (E,N,K//2) uint8, scales (E,N,K//16) e4m3, amax (E,) f32, deq (E,N,K) bf16.
    Mirrors four_over_six_grouped.py:466-474 (weight_amax = weight.abs().amax(dim=(1,2)); _quantize_expert_weights)."""
    w = w_efd_bf16.to(DEV).to(torch.bfloat16).contiguous()
    amax = w.abs().amax(dim=(1, 2)).to(torch.float32)
    codes, scales = fosg._quantize_expert_weights(w, amax, WEIGHT_BLOCK, ERR_MODE, BOUND)
    deq = fosg._dequantize_expert_weights(codes, scales, amax, BOUND)
    return codes, scales, amax, deq

# ---------------- HF loading ----------------
HF_INDEX = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
_hf_handles = {}
def hf_get(name):
    f = HF_INDEX[name]
    if f not in _hf_handles:
        _hf_handles[f] = safe_open(f"{HF}/{f}", framework="pt", device="cpu")
    return _hf_handles[f].get_tensor(name)

def hf_tensor(titan_key):
    """HF -> titan name mapping, from titan-nvfp4/torchtitan/models/qwen3/state_dict_adapter.py:33-50."""
    parts = titan_key.split(".")
    L = int(parts[1])
    if titan_key.endswith("inner_experts.w1_EFD"):
        return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight") for e in range(128)])
    if titan_key.endswith("inner_experts.w3_EFD"):
        return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight") for e in range(128)])
    if titan_key.endswith("inner_experts.w2_EDF"):
        return torch.stack([hf_get(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight") for e in range(128)])
    if titan_key.endswith("attention.wo.weight"):
        return hf_get(f"model.layers.{L}.self_attn.o_proj.weight")
    if titan_key.endswith("attention.qkv_linear.wq.weight"):
        return hf_get(f"model.layers.{L}.self_attn.q_proj.weight")
    if titan_key.endswith("moe.router.gate.weight"):
        return hf_get(f"model.layers.{L}.mlp.gate.weight")
    raise KeyError(titan_key)

# ---------------- DCP loading ----------------
_meta = {}
def dcp_load(run, keys):
    d = RUNS[run]
    if run not in _meta:
        _meta[run] = FileSystemReader(d).read_metadata().state_dict_metadata
    md = _meta[run]
    sd = {}
    for k in keys:
        m = md[k]
        sd[k] = torch.empty(tuple(m.size), dtype=m.properties.dtype)
    t0 = time.time()
    dcp.load(sd, checkpoint_id=d)
    print(f"  dcp.load {run} {len(keys)} keys in {time.time()-t0:.1f}s; dtypes={[str(v.dtype) for v in sd.values()]}", flush=True)
    return sd

# ---------------- stats ----------------
def q(x, ps):
    # quantiles on large tensors: subsample deterministically if > 16M
    x = x.flatten()
    if x.numel() > 16_000_000:
        g = torch.Generator(device=x.device).manual_seed(0)
        idx = torch.randint(0, x.numel(), (16_000_000,), generator=g, device=x.device)
        x = x[idx]
    return torch.quantile(x.float(), torch.tensor(ps, device=x.device)).tolist()

def drift_stats(w0, w1, tag):
    """w0, w1 bf16 tensors of same shape (cpu). Returns dict."""
    a = w0.to(DEV).float(); b = w1.to(DEV).float()
    d = b - a
    ad = d.abs()
    rel = ad / (a.abs() + 1e-8)
    changed = (w0.to(DEV).view(torch.int16) != w1.to(DEV).view(torch.int16))
    n = a.numel()
    out = {
        "numel": n,
        "w0_abs_mean": a.abs().mean().item(), "w0_abs_max": a.abs().max().item(), "w0_rms": a.pow(2).mean().sqrt().item(),
        "frac_bf16_changed": changed.float().mean().item(),
        "abs_drift_mean": ad.mean().item(), "abs_drift_rms": d.pow(2).mean().sqrt().item(), "abs_drift_max": ad.max().item(),
        "abs_drift_p50_p90_p99": q(ad, [0.5, 0.9, 0.99]),
        "rel_drift_p50_p90_p99_max": q(rel, [0.5, 0.9, 0.99]) + [rel.max().item()],
        "rel_drift_p50_p90_p99_changed_only": q(rel[changed], [0.5, 0.9, 0.99]) if changed.any() else None,
        "rel_fro_norm_drift": (d.norm() / a.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        # multiples of bf16 ulp: |d| / ulp(w0)
    }
    ulp = torch.where(a == 0, torch.full_like(a, 2.0**-133), (a.abs().log2().floor().exp2()) * 2.0**-7)
    ulps = ad / ulp
    out["drift_in_ulps_p50_p90_p99_max_changed_only"] = (q(ulps[changed], [0.5, 0.9, 0.99]) + [ulps[changed].max().item()]) if changed.any() else None
    out["frac_changed_exactly_1ulp"] = ((ulps >= 0.99) & (ulps <= 1.01) & changed).float().sum().item() / max(changed.sum().item(), 1)
    if a.dim() == 3:  # per-expert
        pe = changed.float().mean(dim=(1, 2))
        out["per_expert_frac_changed_min_med_max"] = [pe.min().item(), pe.median().item(), pe.max().item()]
        out["n_experts_untouched"] = int((pe == 0).sum().item())
        out["per_expert_frac_changed"] = [round(v, 5) for v in pe.tolist()]
        out["per_expert_rel_fro_drift"] = [round(v, 6) for v in (d.flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)).tolist()]
    del a, b, d, ad, rel, changed, ulp, ulps
    torch.cuda.empty_cache()
    return out

def quant_compare(w0, w1, tag):
    """NVFP4 4over6 quantization of w0 vs w1 (both (E,N,K) bf16): code/scale/amax changes + quantization error."""
    c0, s0, a0, d0 = quantize_expert_stack(w0)
    c1, s1, a1, d1 = quantize_expert_stack(w1)
    # unpack nibbles
    lo0, hi0 = c0 & 0xF, c0 >> 4; lo1, hi1 = c1 & 0xF, c1 >> 4
    code_changed = torch.cat([(lo0 != lo1).flatten(), (hi0 != hi1).flatten()])
    scale_changed = (s0.view(torch.uint8) != s1.view(torch.uint8))
    w0f = w0.to(DEV).float(); w1f = w1.to(DEV).float()
    e0 = (d0.float() - w0f); e1 = (d1.float() - w1f)
    deq_changed = (d0.view(torch.int16) != d1.view(torch.int16))
    dd = (d1.float() - d0.float())
    out = {
        "frac_fp4_codes_changed": code_changed.float().mean().item(),
        "frac_block_scales_changed": scale_changed.float().mean().item(),
        "frac_dequant_values_changed": deq_changed.float().mean().item(),
        "n_experts_amax_changed": int((a0 != a1).sum().item()),
        "expert_amax_rel_change_max": ((a1 - a0).abs() / a0).max().item(),
        "quant_err_rel_rms_w0": (e0.norm() / w0f.norm()).item(),
        "quant_err_rel_rms_w300": (e1.norm() / w1f.norm()).item(),
        "quant_err_abs_rms_w0": e0.pow(2).mean().sqrt().item(),
        "quant_err_abs_rms_w300": e1.pow(2).mean().sqrt().item(),
        "dequant_drift_rel_fro": (dd.norm() / d0.float().norm()).item(),
        "dequant_drift_abs_rms": dd.pow(2).mean().sqrt().item(),
        "raw_drift_abs_rms": (w1f - w0f).pow(2).mean().sqrt().item(),
        # how much of the raw drift survives quantization: corr(dd, w1-w0)
        "corr_dequant_drift_vs_raw_drift": torch.nn.functional.cosine_similarity(dd.flatten(), (w1f - w0f).flatten(), dim=0).item(),
        "per_expert_frac_codes_changed": [round(v, 5) for v in torch.cat([(lo0 != lo1), (hi0 != hi1)], dim=-1).float().mean(dim=(1, 2)).tolist()],
        "per_expert_frac_scales_changed": [round(v, 5) for v in scale_changed.float().mean(dim=(1, 2)).tolist()],
        "per_expert_amax_changed": [bool(v) for v in (a0 != a1).tolist()],
    }
    del c0, s0, d0, c1, s1, d1, lo0, hi0, lo1, hi1, w0f, w1f, e0, e1, dd
    torch.cuda.empty_cache()
    return out

# ---------------- main ----------------
results = {"quantizer": QUANT_SRC, "arm_quant_settings": {"weight_block": WEIGHT_BLOCK, "err_mode": ERR_MODE, "e4m3_scale_bound": BOUND},
           "layers": LAYERS, "nvfp4_quantized_layers": "0-39 (nvfp4_bf16_first_last_fqns(48,0,8)); 40-47 bf16 in both arms", "tensors": {}}
keys = []
for L in LAYERS:
    keys += [f"layers.{L}.moe.routed_experts.inner_experts.w1_EFD", f"layers.{L}.moe.routed_experts.inner_experts.w2_EDF", f"layers.{L}.moe.routed_experts.inner_experts.w3_EFD"]
    keys += [f"layers.{L}.attention.wo.weight", f"layers.{L}.moe.router.gate.weight"]
if SMOKE:
    keys = [f"layers.20.moe.router.gate.weight", f"layers.20.attention.wo.weight", f"layers.20.moe.routed_experts.inner_experts.w2_EDF"]

# self-test of quantizer: quantize a tensor and check dequant error is sane
if fos is not None:
    t = torch.randn(2, 256, 512, dtype=torch.bfloat16, device=DEV)
    c, s, a, d = quantize_expert_stack(t)
    print("quantizer self-test rel err", ((d.float() - t.float()).norm() / t.float().norm()).item(), flush=True)

for k in keys:
    t0 = time.time()
    print("=== ", k, flush=True)
    w0 = hf_tensor(k)
    entry = {"shape": list(w0.shape), "hf_dtype": str(w0.dtype), "runs": {}}
    w300 = {}
    for run in RUNS:
        sd = dcp_load(run, [k])
        w300[run] = sd[k]
        entry["ckpt_dtype_" + run] = str(sd[k].dtype)
        entry["runs"][run] = drift_stats(w0, sd[k], f"{k}:{run}")
        print(f"  [{run}] frac_changed={entry['runs'][run]['frac_bf16_changed']:.5f} rel_fro={entry['runs'][run]['rel_fro_norm_drift']:.3e} cos={entry['runs'][run]['cosine']:.6f} absmax={entry['runs'][run]['abs_drift_max']:.3e}", flush=True)
    # cross-run divergence
    entry["bf16_vs_nvfp4_step300"] = drift_stats(w300["bf16"], w300["nvfp4"], f"{k}:cross")
    print(f"  [bf16 vs nvfp4 @300] frac_diff={entry['bf16_vs_nvfp4_step300']['frac_bf16_changed']:.5f} rel_fro={entry['bf16_vs_nvfp4_step300']['rel_fro_norm_drift']:.3e}", flush=True)
    if "inner_experts" in k and fos is not None:
        entry["nvfp4_quant"] = {}
        for run in RUNS:
            try:
                entry["nvfp4_quant"][run] = quant_compare(w0, w300[run], f"{k}:{run}")
                qq = entry["nvfp4_quant"][run]
                print(f"  [quant {run}] codes_changed={qq['frac_fp4_codes_changed']:.5f} scales_changed={qq['frac_block_scales_changed']:.5f} deq_changed={qq['frac_dequant_values_changed']:.5f} amax_changed_experts={qq['n_experts_amax_changed']} qerr_rel_rms w0/w300={qq['quant_err_rel_rms_w0']:.4f}/{qq['quant_err_rel_rms_w300']:.4f} corr_deqdrift_rawdrift={qq['corr_dequant_drift_vs_raw_drift']:.4f}", flush=True)
            except Exception as e:
                entry["nvfp4_quant"][run] = {"error": repr(e)}
                traceback.print_exc()
    results["tensors"][k] = entry
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    del w0, w300
    with open(f"{OUT}/drift_results{'_smoke' if SMOKE else ''}.json", "w") as f:
        json.dump(results, f, indent=1)

# optimizer moments for one expert tensor + one attention tensor (dtype + magnitude)
try:
    for run in RUNS:
        ks = ["optimizer.state.layers.20.moe.routed_experts.inner_experts.w2_EDF.exp_avg",
              "optimizer.state.layers.20.moe.routed_experts.inner_experts.w2_EDF.exp_avg_sq",
              "optimizer.state.layers.20.moe.routed_experts.inner_experts.w2_EDF.step"]
        sd = dcp_load(run, ks)
        m, v, st = sd[ks[0]].to(DEV).float(), sd[ks[1]].to(DEV).float(), sd[ks[2]]
        upd = 1e-6 * m / (v.sqrt() + 1e-8)  # AdamW nominal update (ignoring bias correction)
        wk = "layers.20.moe.routed_experts.inner_experts.w2_EDF"
        w = dcp_load(run, [wk])[wk].to(DEV).float()
        ulp = torch.where(w == 0, torch.full_like(w, 2.0**-133), (w.abs().log2().floor().exp2()) * 2.0**-7)
        frac_upd_survives = (upd.abs() >= ulp / 2).float().mean().item()  # RTNE: update must exceed half an ulp to move a bf16 weight
        frac_upd_survives_10 = (upd.abs() >= ulp / 20).float().mean().item()
        results.setdefault("optimizer_probe", {})[run] = {
            "dtypes": [str(sd[k].dtype) for k in ks], "step": st.item(),
            "exp_avg_abs_mean": m.abs().mean().item(), "exp_avg_sq_mean": v.mean().item(),
            "frac_exp_avg_sq_zero": (v == 0).float().mean().item(), "frac_exp_avg_zero": (m == 0).float().mean().item(),
            "nominal_update_abs_p50_p90_p99": q(upd.abs(), [0.5, 0.9, 0.99]),
            "nominal_update_abs_max": upd.abs().max().item(),
            "frac_elements_nominal_update_ge_half_ulp_of_w300": frac_upd_survives,
            "frac_elements_nominal_update_ge_ulp_over_20": frac_upd_survives_10,
            "bf16_ulp_of_w300_p50": q(ulp, [0.5])[0],
            "w300_abs_p50": q(w.abs(), [0.5])[0],
            "per_expert_frac_exp_avg_sq_nonzero": [round(x, 4) for x in (v != 0).float().mean(dim=(1, 2)).tolist()],
        }
        print(run, {kk: vv for kk, vv in results["optimizer_probe"][run].items() if kk != "per_expert_frac_exp_avg_sq_nonzero"}, flush=True)
except Exception as e:
    results["optimizer_probe_error"] = repr(e); traceback.print_exc()
with open(f"{OUT}/drift_results{'_smoke' if SMOKE else ''}.json", "w") as f:
    json.dump(results, f, indent=1)
print("ALL DONE", flush=True)
