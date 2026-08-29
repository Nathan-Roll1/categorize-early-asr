#!/usr/bin/env python3
"""
Figure 2: Mean Peak Layer Position by Architecture
Uses local analysis_results.json data instead of Google Drive CSVs.
"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------- Paths --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(ROOT, "analysis_results.json")
OUT_PDF = os.path.join(SCRIPT_DIR, "fig 2.pdf")

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


def plot_avg_peak_position():
    # Load data
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    peak_stats = data["peak_stats"]

    # Build dataframe for plotting
    records = []
    for feature, arch_data in peak_stats.items():
        for arch, stats in arch_data.items():
            records.append({
                "Feature": feature.capitalize(),
                "Architecture": arch,
                "Mean_Peak": stats["mean"],
                "SE": stats["std"] / np.sqrt(stats["n"])
            })

    bar_df = pd.DataFrame(records)

    # Create SE map for error bars
    se_map = {(r["Architecture"], r["Feature"]): r["SE"] for _, r in bar_df.iterrows()}

    # Determine ordering using Transformer only
    transformer_order = (
        bar_df[bar_df["Architecture"] == "Transformer"]
        .sort_values("Mean_Peak")["Feature"]
        .tolist()
    )

    print("Transformer-derived order:", transformer_order)

    # Force architecture order
    arch_order = ["Transformer", "Conformer"]
    bar_df["Architecture"] = pd.Categorical(
        bar_df["Architecture"], categories=arch_order, ordered=True)

    # === Plot ===
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=bar_df,
        x="Architecture",
        y="Mean_Peak",
        hue="Feature",
        hue_order=transformer_order,
        palette="Set2",
        errorbar=None,
        edgecolor="black",
        linewidth=1.5
    )

    # === Add error bars ===
    # Sort bar_df to match seaborn's plotting order
    bar_df["Feature"] = pd.Categorical(bar_df["Feature"], categories=transformer_order, ordered=True)
    bar_df_sorted = bar_df.sort_values(["Feature", "Architecture"])

    for bar, (_, row) in zip(ax.patches, bar_df_sorted.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        se = se_map[(row["Architecture"], row["Feature"])]

        ax.errorbar(
            x, y,
            yerr=se,
            fmt="none",
            ecolor="black",
            elinewidth=2.0,
            capsize=6,
            capthick=2.0
        )

    max_y = bar_df["Mean_Peak"].max()
    plt.ylim(0, max_y + 0.15)

    plt.ylabel("Mean Peak Layer (% of model depth)", fontsize=20, fontweight='bold')
    plt.xlabel("Architecture", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, axis='y', alpha=0.3)

    # Move legend to top
    plt.legend(
        title="",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(transformer_order),
        fontsize=16,
        frameon=False
    )

    ax.set_title("")

    # Remove top and right spines
    sns.despine()

    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.show()

    print(f"Saved PDF → {OUT_PDF}")


if __name__ == "__main__":
    plot_avg_peak_position()
