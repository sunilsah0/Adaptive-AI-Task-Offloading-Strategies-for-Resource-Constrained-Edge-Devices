"""
plot_results.py  -  figures for the new benchmark.py output.
Put it in scripts\\ (or next to benchmark.py) and run:  python scripts\\plot_results.py
Reads results/summary.csv and results/sweep.csv.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "results"
LABELS = {
    "edge_only": "Edge-only (0.5B)",
    "cloud_only": "Cloud-only (27B)",
    "static_threshold": "Static threshold (prior-art style)",
    "dynamic": "Dynamic (proposed)",
}
COLORS = {"edge_only": "#4C72B0", "cloud_only": "#55A868",
          "static_threshold": "#C44E52", "dynamic": "#8172B2"}
ORDER = list(LABELS)

summary = pd.read_csv(os.path.join(RES, "summary.csv"))
scenarios = list(dict.fromkeys(summary["scenario"]))

# ---- Figure 1: three metrics per network scenario (error bars = 95% CI where available)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
panels = [("SLO_%", "SLO_ci95", "(a) Deadline compliance (%)"),
          ("accuracy", "acc_ci95", "(b) Accuracy (F1 / exact number)"),
          ("on_time_accuracy", None, "(c) On-time accuracy (correct AND within deadline)")]
width = 0.2
x = np.arange(len(scenarios))
for ax, (col, ci, title) in zip(axes, panels):
    for i, pol in enumerate(ORDER):
        d = summary[summary.policy == pol].set_index("scenario").reindex(scenarios)
        err = d[ci] if ci else None
        ax.bar(x + (i - 1.5) * width, d[col], width, yerr=err, capsize=2,
               label=LABELS[pol], color=COLORS[pol], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
axes[0].set_ylim(0, 110)
axes[1].set_ylim(0, 1)
axes[2].set_ylim(0, 1)
axes[0].legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(RES, "fig_scenarios.png"), dpi=300)
print("saved", os.path.join(RES, "fig_scenarios.png"))

# ---- Figure 2: sweep over RTT (where does channel-awareness start to matter?)
sweep_path = os.path.join(RES, "sweep.csv")
if os.path.exists(sweep_path):
    sw = pd.read_csv(sweep_path)
    scales = sorted(sw["deadline_scale"].unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(5 * len(scales), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scales):
        for pol in ORDER:
            d = sw[(sw.policy == pol) & (sw.deadline_scale == sc)].sort_values("rtt")
            ax.plot(d["rtt"], d["on_time_accuracy"], marker="o", label=LABELS[pol], color=COLORS[pol])
        ax.set_xscale("log")
        ax.set_title(f"Deadlines x{sc}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Emulated RTT (s, log scale)")
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("On-time accuracy")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_sweep.png"), dpi=300)
    print("saved", os.path.join(RES, "fig_sweep.png"))