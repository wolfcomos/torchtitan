"""Extract per-step scalars from the six ablation arms' tfevents and render
the results artifacts: per-metric CSV, summary CSV, and line-chart PNGs.

Run inside the vLLM container with tensorboard+matplotlib on PYTHONPATH:
  python3 make_results.py <outputs_dir> <results_dir>
"""

import csv
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTPUTS = sys.argv[1]
RESULTS = sys.argv[2]
os.makedirs(RESULTS, exist_ok=True)

# Fixed arm order and validated categorical palette (slots 1-6, light mode).
ARMS = [
    ("arm_bf16", "BF16", "#2a78d6"),
    ("arm_mxfp8_e2e", "MXFP8 e2e", "#eb6834"),
    ("arm_mxfp8_hp", "MXFP8 + hp bwd", "#1baf7a"),
    ("arm_mxfp8_deq", "MXFP8 + deq bwd", "#eda100"),
    ("arm_nvfp4_hp", "NVFP4-4over6 + hp bwd", "#e87ba4"),
    ("arm_nvfp4_deq", "NVFP4-4over6 + deq bwd", "#008300"),
]

PANELS = [
    ("loss/mean", "GRPO loss (mean)"),
    ("trainer/grad_norm/mean", "Gradient norm"),
    ("rollout_reward/_mean", "Rollout reward (mean of 64 samples)"),
    ("trainer/entropy/mean", "Policy entropy"),
    ("bit_wise/logprob_diff/mean", "Rollout-vs-trainer logprob diff (mean)"),
    ("loss/ratio_clipped_frac", "PPO ratio clipped fraction"),
]

data = {}  # arm -> tag -> [(step, value)]
for arm, _, _ in ARMS:
    ev = glob.glob(os.path.join(OUTPUTS, arm, "events.out.tfevents.*"))
    if not ev:
        print(f"WARNING: no events for {arm}, skipping")
        continue
    acc = EventAccumulator(ev[0])
    acc.Reload()
    tags = acc.Tags()["scalars"]
    data[arm] = {t: [(s.step, s.value) for s in acc.Scalars(t)] for t in tags}

# ---- CSV: long-form per-step metrics (the chart's table view) ----
with open(os.path.join(RESULTS, "training_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "metric", "step", "value"])
    for arm, label, _ in ARMS:
        for tag, _ in PANELS:
            for step, value in data.get(arm, {}).get(tag, []):
                w.writerow([arm, tag, step, value])

# ---- CSV: per-arm summary aggregates ----
def series(arm, tag):
    return [v for _, v in data.get(arm, {}).get(tag, [])]

def val_reward(arm, step):
    for s, v in data.get(arm, {}).get("validation_reward/_mean", []):
        if s == step:
            return v
    return None

with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "arm",
            "steps",
            "loss_mean",
            "grad_norm_mean",
            "reward_mean",
            "reward_last",
            "logprob_diff_mean",
            "clipped_frac_mean",
            "val_reward_step0",
            "val_reward_step10",
        ]
    )
    for arm, label, _ in ARMS:
        loss = series(arm, "loss/mean")
        if not loss:
            continue
        gn = series(arm, "trainer/grad_norm/mean")
        rw = series(arm, "rollout_reward/_mean")
        lp = series(arm, "bit_wise/logprob_diff/mean")
        cf = series(arm, "loss/ratio_clipped_frac")
        w.writerow(
            [
                arm,
                len(loss),
                sum(loss) / len(loss),
                sum(gn) / len(gn),
                sum(rw) / len(rw),
                rw[-1],
                sum(lp) / len(lp),
                sum(cf) / len(cf),
                val_reward(arm, 0),
                val_reward(arm, 10),
            ]
        )

# ---- Charts: 2x3 small multiples, one shared legend ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "grid.color": "#e8e8e2",
        "grid.linewidth": 0.6,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (tag, title) in zip(axes.flat, PANELS):
    for arm, label, color in ARMS:
        pts = data.get(arm, {}).get(tag, [])
        if not pts:
            continue
        steps, values = zip(*pts)
        ax.plot(
            steps,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4.5,
            label=label,
        )
    ax.set_title(title, color="#333333")
    ax.set_xlabel("training step")
    ax.set_xticks(range(1, 11))
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle(
    "Qwen3-30B-A3B GRPO recipe ablation - 10 steps, dapo-math-17k, seed 42, "
    "2 trainer + 2 generator GB200",
    color="#333333",
)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(os.path.join(RESULTS, "training_curves.png"), dpi=160, bbox_inches="tight")
print("wrote", RESULTS)
