#!/usr/bin/env python3
"""
FIGURE 3 (NORMALIZED): Smoothed Regression Lines + 95% Bootstrapped CI
Acoustic Aggregated — CAPITALIZED FEATURE GROUPS — MIN–MAX NORMALIZED Y
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------- Paths --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
# Raw data is expected in the root or figs directory if running locally
CSV = os.path.join(ROOT, "ALL_PROBING_WITH_PHONEMES.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "fig 3.pdf")

# -------- Font setup --------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.linewidth": 1.0,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "text.antialiased": True
})

# -------- Model depth map --------
true_layer_counts = {
    'whisper-tiny.en': 4, 'whisper-tiny': 4,
    'whisper-base.en': 6, 'whisper-base': 6,
    'whisper-small.en': 12, 'whisper-small': 12,
    'whisper-medium.en': 24, 'whisper-medium': 24,
    'whisper-large': 32, 'whisper-large-v2': 32,
    'whisper-large-v3': 32,
    'whisper-large-v3-turbo': 32,
    'canary-1b': 32, 'canary-1b-flash': 32,
    'canary-qwen2.5b': 40,
    'parakeet-tdt-0.6b-v2': 30,
}

def infer_arch(model):
    m = model.lower()
    if "canary" in m or "parakeet" in m:
        return "Conformer"
    return "Transformer"

def get_depth(df, model):
    if model in true_layer_counts:
        return true_layer_counts[model]
    return int(df[df["Model"] == model]["Layer"].max()) + 1

def bootstrap_lowess(x, y, grid, n_boot=200, frac=0.25):
    if len(x) < 5:
        return np.nan * grid, np.nan * grid, np.nan * grid
    preds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        xs, ys = x[idx], y[idx]
        sm = lowess(ys, xs, frac=frac, return_sorted=True)
        if len(sm) < 2:
            continue
        interp = np.interp(grid, sm[:, 0], sm[:, 1])
        preds.append(interp)
    preds = np.array(preds)
    if preds.shape[0] == 0:
        return np.nan * grid, np.nan * grid, np.nan * grid
    return preds.mean(axis=0), np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)

def plot_fig3():
    if not os.path.exists(CSV):
        print(f"Error: Raw data file not found at {CSV}")
        return

    df = pd.read_csv(CSV)
    df["Architecture"] = df["Model"].apply(infer_arch)
    df["Total_Layers"] = df["Model"].apply(lambda m: get_depth(df, m))
    df["Layer_Normalized"] = df["Layer"] / (df["Total_Layers"] - 1)

    acoustic = [
        "f0_min","f0_mean","f0_max","f0_median",
        "F1_min","F1_mean","F1_max","F1_median",
        "F2_min","F2_mean","F2_max","F2_median",
        "F3_min","F3_mean","F3_max","F3_median",
        "intensity_min","intensity_mean","intensity_max","intensity_median",
        "duration"
    ]

    df["Feature_Group"] = "Other"
    df.loc[df.Feature.isin(acoustic), "Feature_Group"] = "Acoustic"
    df.loc[df.Feature == "gender", "Feature_Group"] = "Gender"
    df.loc[df.Feature == "l1_background", "Feature_Group"] = "Accent"
    df.loc[df.Feature == "phoneme", "Feature_Group"] = "Phoneme"

    keep = ["Acoustic", "Gender", "Accent", "Phoneme"]
    df = df[df.Feature_Group.isin(keep)]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    palette = {
        "Acoustic": "black",
        "Gender": "#1f77b4",
        "Accent": "#ff7f0e",
        "Phoneme": "#2ca02c"
    }

    grid = np.linspace(0, 1, 150)

    for i, arch in enumerate(["Transformer", "Conformer"]):
        ax = axes[i]
        sub = df[df.Architecture == arch]

        for feat in keep:
            d = sub[sub.Feature_Group == feat]
            x = d["Layer_Normalized"].values
            y = d["Test_Score"].values

            if len(y) > 0:
                y_min, y_max = y.min(), y.max()
                y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
            else:
                y_norm = y

            mean, lo, hi = bootstrap_lowess(x, y_norm, grid)
            ax.plot(grid, mean, color=palette[feat], lw=3)
            ax.fill_between(grid, lo, hi, color=palette[feat], alpha=0.2)

        ax.set_xlabel("Normalized Layer Depth", fontsize=18, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Normalized Probing Score", fontsize=18, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.text(0.5, 1.03, arch, ha="center", va="bottom",
                fontsize=20, fontweight='bold', transform=ax.transAxes)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.legend(
        [plt.Line2D([], [], color=palette[k], lw=4) for k in keep],
        keep,
        loc="upper center",
        ncol=4,
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.show()
    print(f"Saved Figure 3 → {OUT_PATH}")

if __name__ == "__main__":
    plot_fig3()
