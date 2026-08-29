#!/usr/bin/env python3
"""
Generate manuscript figures from Google Drive synced data.

This script generates all three figures used in the ASR paper:
- Fig 1: t-SNE visualization comparing Transformer vs Conformer
- Fig 2: Bar plot of average peak position by hierarchy
- Fig 3: LOWESS regression with bootstrap CI

Usage:
    python generate_manuscript_figures.py
    python generate_manuscript_figures.py --root "/path/to/Google Drive/My Drive/t-SNE & Probing"
    python generate_manuscript_figures.py --output ./figs --fig 1 2 3
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess

# Optional imports with graceful fallback
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Fig 1 will use cached labels.")

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Warning: 'rapidfuzz' library not installed. Using simple text matching for Fig 1.")


# =============================================================================
# Configuration
# =============================================================================

def get_default_root():
    """Auto-detect Google Drive path based on OS."""
    home = Path.home()

    # Common Google Drive Desktop paths
    candidates = [
        home / "Google Drive" / "My Drive" / "t-SNE & Probing",
        home / "GoogleDrive" / "My Drive" / "t-SNE & Probing",
        Path("/Volumes/GoogleDrive/My Drive/t-SNE & Probing"),
    ]
    cloud_storage = home / "Library" / "CloudStorage"
    if cloud_storage.exists():
        candidates.extend(
            drive / "My Drive" / "t-SNE & Probing"
            for drive in cloud_storage.glob("GoogleDrive-*")
        )

    for path in candidates:
        if path.exists():
            return path

    # Fallback to first candidate
    return candidates[0]


def setup_fonts(root: Path):
    """Load Helvetica font or fallback to system sans-serif."""
    font_path = root / "fonts" / "HelveticaNeueMedium.otf"

    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_props = fm.FontProperties(fname=str(font_path))
        font_name = font_props.get_name()
        print(f"✅ Loaded font: {font_name}")
    else:
        font_name = "sans-serif"
        print(f"⚠️ Helvetica font not found at {font_path}, using system sans-serif")

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["patch.antialiased"] = True
    plt.rcParams["text.antialiased"] = True

    return font_name


# =============================================================================
# Figure 1: t-SNE Visualization
# =============================================================================

def load_tsne_and_labels(root: Path, model_tag: str, dataset: str, feature: str, dataset_meta=None):
    """Load t-SNE coordinates and labels for a model.

    Args:
        root: Root path to Google Drive data
        model_tag: Model name (e.g. 'whisper-large-v3-turbo')
        dataset: Dataset name (e.g. 'l2-arctic-dataset-250')
        feature: Feature name (e.g. 'l1_background')
        dataset_meta: Optional HuggingFace dataset with labels

    Returns:
        tsne_layers: List of t-SNE coordinates per layer
        labels: Dictionary with 'l1_background', 'speaker', and optionally 'text' labels
    """
    tsne_path = root / "t-SNE" / model_tag / dataset / f"{model_tag}_{dataset}_{feature}_tsne.pkl"

    print(f"📂 Loading t-SNE: {tsne_path}")
    with open(tsne_path, "rb") as f:
        tsne_layers = pickle.load(f)

    # Try to load labels from Layer Representations (Colab path)
    rep_path = root / "Layer Representations" / model_tag / f"{dataset}.pkl"
    if rep_path.exists():
        with open(rep_path, "rb") as f:
            labels = pickle.load(f)["labels"]
    elif dataset_meta is not None:
        # Fallback: extract labels from HuggingFace dataset
        # Find first non-None layer to get sample count
        n_samples = None
        for layer_data in tsne_layers:
            if layer_data is not None:
                n_samples = layer_data.shape[0]
                break
        if n_samples is None:
            raise ValueError(f"All t-SNE layers are None for {model_tag}")

        labels = {
            "l1_background": dataset_meta["l1_background"][:n_samples],
            "speaker": dataset_meta["speaker"][:n_samples],
            "text": dataset_meta["text"][:n_samples],
        }
        print(f"📋 Extracted labels from HuggingFace dataset ({n_samples} samples)")
    else:
        raise FileNotFoundError(
            f"Could not find labels. Either Layer Representations folder or HuggingFace dataset required.\n"
            f"Tried: {rep_path}"
        )

    return tsne_layers, labels


def make_accent_colors(labels):
    """Create colors based on L1 background with speaker variation."""
    accent_enc = LabelEncoder().fit(labels["l1_background"])
    accents = accent_enc.transform(labels["l1_background"])
    speaker_enc = LabelEncoder().fit(labels["speaker"])
    speaker_norm = speaker_enc.transform(labels["speaker"]) / (len(speaker_enc.classes_) - 1)

    accent_colors = np.array([
        (0.5, 0.5, 0.5),       # gray — Arabic
        (0.121, 0.466, 0.705), # blue — Hindi
        (0.172, 0.627, 0.172), # green — Korean
        (0.839, 0.152, 0.156), # red — Mandarin
        (0.894, 0.466, 0.761), # magenta — Spanish
        (1.000, 0.498, 0.054)  # orange — Vietnamese
    ])[:len(accent_enc.classes_)]

    colors = []
    for acc, spk_norm in zip(accents, speaker_norm):
        base = np.array(accent_colors[acc])
        factor = 0.6 + 0.7 * spk_norm
        if acc == 1: factor *= 1.15
        elif acc == 3: factor *= 1.15
        elif acc == 4 and spk_norm > 0.8: factor *= 1.15
        colors.append(np.clip(base * factor, 0, 1))

    return np.array(colors), accent_enc


def build_global_utterance_map(labels_list, similarity_threshold=85):
    """Combine texts from all models and assign consistent cluster IDs."""
    all_texts = []
    for labels in labels_list:
        all_texts.extend(labels.get("text", []))
    all_texts = np.array(all_texts)

    unique_texts, cluster_map = [], {}
    for txt in all_texts:
        matched = False
        for cid, ut in enumerate(unique_texts):
            if HAS_RAPIDFUZZ:
                similarity = fuzz.ratio(txt, ut)
            else:
                # Simple fallback: exact match or substring
                similarity = 100 if txt == ut else (80 if txt in ut or ut in txt else 0)

            if similarity > similarity_threshold:
                cluster_map[txt] = cid
                matched = True
                break
        if not matched:
            cluster_map[txt] = len(unique_texts)
            unique_texts.append(txt)

    print(f"✅ Global utterance clusters: {len(unique_texts)} unique groups")
    return cluster_map, len(unique_texts)


def assign_utterance_colors(labels, cluster_map, n_clusters):
    """Assign colors to utterances based on cluster ID."""
    cluster_ids = np.array([cluster_map.get(txt, 0) for txt in labels["text"]])

    # Complementary hue bands (avoid red/orange/green/blue/magenta)
    hue_bands = [(0.45, 0.52), (0.68, 0.78), (0.92, 1.00)]
    hues = []
    for i in range(n_clusters):
        band = hue_bands[i % len(hue_bands)]
        h = band[0] + (band[1] - band[0]) * ((i // len(hue_bands)) / max(1, n_clusters / len(hue_bands)))
        hues.append(h)

    rgb_colors = [plt.cm.hsv(h)[:3] for h in hues]
    colors = np.array([rgb_colors[cid % len(rgb_colors)] for cid in cluster_ids])
    return colors, rgb_colors


def generate_fig1_tsne(root: Path, output_dir: Path):
    """Generate Figure 1: t-SNE visualization comparing Transformer vs Conformer."""
    print("\n" + "="*60)
    print("Generating Figure 1: t-SNE Visualization")
    print("="*60)

    # Configuration
    DATASET = "l2-arctic-dataset-250"
    MODELS = ["whisper-large-v3-turbo", "canary-1b"]
    FEATURE = "l1_background"

    SELECTED_LAYERS_WHISPER = [1, 12, 24, 32]
    SELECTED_LAYERS_CANARY = [1, 9, 18, 24]
    DEPTHS = [0, 37.5, 75, 100]

    # L1 background mapping
    L1_MAP = {
        0: "Arabic",
        1: "Hindi",
        2: "Korean",
        3: "Mandarin",
        4: "Spanish",
        5: "Vietnamese"
    }

    # Try to get mapping from dataset if available
    dataset_meta = None
    if HAS_DATASETS:
        try:
            dataset_meta = load_dataset("PranavBhalerao/l2-arctic-dataset-250", split="train")
            L1_MAP = dict(enumerate(dataset_meta.features["l1_background"].names))
            print("✅ Loaded dataset metadata from HuggingFace")
        except Exception as e:
            print(f"⚠️ Could not load dataset from HuggingFace: {e}")
            dataset_meta = None

    # Load model data (with dataset_meta for label fallback)
    whisper_tsne, whisper_labels = load_tsne_and_labels(root, "whisper-large-v3-turbo", DATASET, FEATURE, dataset_meta)
    canary_tsne, canary_labels = load_tsne_and_labels(root, "canary-1b", DATASET, FEATURE, dataset_meta)

    # Create colors
    whisper_colors, whisper_accent_enc = make_accent_colors(whisper_labels)
    canary_colors, canary_accent_enc = make_accent_colors(canary_labels)

    # Build consistent utterance colors
    if "text" in whisper_labels and "text" in canary_labels:
        cluster_map, n_clusters = build_global_utterance_map([whisper_labels, canary_labels])
        whisper_utt_colors, utt_rgb_colors = assign_utterance_colors(whisper_labels, cluster_map, n_clusters)
        canary_utt_colors, _ = assign_utterance_colors(canary_labels, cluster_map, n_clusters)
    else:
        # Fallback: use accent colors for utterance panel too
        whisper_utt_colors = whisper_colors
        canary_utt_colors = canary_colors
        utt_rgb_colors = [(0.5, 0.5, 0.5)] * 10

    # Create figure
    ncols = len(DEPTHS) + 1
    fig, axs = plt.subplots(2, ncols, figsize=(ncols*3.3, 6), dpi=300,
                            facecolor="white", gridspec_kw=dict(wspace=0.15, hspace=0.05))

    for row, (model_tag, tsne_layers, colors, utt_colors, selected_layers) in enumerate(
        zip(MODELS, [whisper_tsne, canary_tsne],
            [whisper_colors, canary_colors],
            [whisper_utt_colors, canary_utt_colors],
            [SELECTED_LAYERS_WHISPER, SELECTED_LAYERS_CANARY])
    ):
        for col, (layer_idx, depth) in enumerate(zip(selected_layers, DEPTHS)):
            coords = tsne_layers[layer_idx]
            ax = axs[row, col]
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, edgecolors="black",
                       linewidths=0.25, alpha=0.55, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
            for spine in ax.spines.values(): spine.set_visible(False)
            if row == 0: ax.set_title(f"Depth: {depth}%", fontsize=12, pad=18, weight="medium")

        coords_utt = tsne_layers[selected_layers[-1]]
        ax_utt = axs[row, -1]
        ax_utt.scatter(coords_utt[:, 0], coords_utt[:, 1], c=utt_colors, s=10,
                       edgecolors="black", linewidths=0.25, alpha=0.45, rasterized=True)
        ax_utt.set_xticks([]); ax_utt.set_yticks([]); ax_utt.set_facecolor("white")
        for spine in ax_utt.spines.values(): spine.set_visible(False)
        if row == 0: ax_utt.set_title("Depth: 100% (Utterance)", fontsize=12, pad=18, weight="medium")

    # Centered row labels
    fig.canvas.draw()
    bbox_whisper = axs[0, 0].get_position(); bbox_canary = axs[1, 0].get_position()
    y_whisper = (bbox_whisper.y0 + bbox_whisper.y1) / 2
    y_canary = (bbox_canary.y0 + bbox_canary.y1) / 2
    x_label = bbox_whisper.x0 - 0.055
    fig.text(x_label, y_whisper, "Whisper-large-v3-turbo", ha="center", va="center", fontsize=12, weight="medium")
    fig.text(x_label, y_canary, "Canary-1B", ha="center", va="center", fontsize=12, weight="medium")

    # Double legend
    shared_enc = whisper_accent_enc
    base_colors = np.array([
        (0.5, 0.5, 0.5),
        (0.121, 0.466, 0.705),
        (0.172, 0.627, 0.172),
        (0.839, 0.152, 0.156),
        (0.894, 0.466, 0.761),
        (1.000, 0.498, 0.054)
    ])[:len(shared_enc.classes_)]

    handles_l1 = [plt.Line2D([], [], marker='o', color='w',
                   markerfacecolor=tuple(base_colors[i]),
                   markeredgecolor='black', markersize=6,
                   label=L1_MAP.get(i, f"ID {i}"))
        for i in range(len(shared_enc.classes_))]

    handles_utt = [plt.Line2D([], [], marker='o', color='w',
                   markerfacecolor=tuple(utt_rgb_colors[i % len(utt_rgb_colors)]),
                   markeredgecolor='black', markersize=6,
                   label=f"Utterance {i+1}")
        for i in range(10)]

    fig.legend(handles=handles_l1, loc='lower left', bbox_to_anchor=(0.15, 0.02),
               ncol=len(handles_l1)//2, frameon=False, fontsize=9,
               title="L1 Background", title_fontsize=10)
    fig.legend(handles=handles_utt, loc='lower right', bbox_to_anchor=(0.85, 0.02),
               ncol=5, frameon=False, fontsize=9,
               title="Utterance Number", title_fontsize=10)

    plt.subplots_adjust(bottom=0.18, left=0.15, right=0.98, top=0.92)

    # Save
    output_path = output_dir / "fig 1.pdf"
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"✅ Saved Figure 1 to: {output_path}")


# =============================================================================
# Figure 2: Average Peak Position Bar Plot
# =============================================================================

def generate_fig2_peak_position(root: Path, output_dir: Path):
    """Generate Figure 2: Average peak position by hierarchy and architecture."""
    print("\n" + "="*60)
    print("Generating Figure 2: Peak Position Bar Plot")
    print("="*60)

    # Load data
    csv_path = root / "ALLDATA_aggregate_metrics (with phonemes).csv"
    if not csv_path.exists():
        # Try alternate location
        csv_path = root / "aggregate metrics" / "ALLDATA_aggregate_metrics (with phonemes).csv"

    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded data: {df.shape}")

    # Column groups
    acoustic_cols = [c for c in df.columns
                     if c.startswith(("f0_", "f1_", "f2_", "f3_",
                                      "f3_minus_f2_", "intensity_"))
                     and c.endswith("_peak_layer_percent")]

    temporal_cols = [c for c in df.columns
                     if c.startswith("duration_") and c.endswith("_peak_layer_percent")]

    gender_cols = [c for c in df.columns
                   if c.startswith("gender_") and c.endswith("_peak_layer_percent")]

    accent_cols = [c for c in df.columns
                   if c.startswith("l1_background_") and c.endswith("_peak_layer_percent")]

    phoneme_cols = [c for c in df.columns
                    if c.startswith("phoneme_") and c.endswith("_peak_layer_percent")]

    acoustic_combined = acoustic_cols + temporal_cols

    # Compute per-model means
    summary = []
    for _, row in df.iterrows():
        summary.append({
            "Model": row["Model"],
            "Architecture": row["Architecture"],
            "Acoustic": row[acoustic_combined].mean(),
            "Gender": row[gender_cols].mean(),
            "Accent": row[accent_cols].mean(),
            "Phoneme": row[phoneme_cols].mean(),
        })

    summary_df = pd.DataFrame(summary)

    # Melt for bar plotting
    melted = summary_df.melt(
        id_vars=["Model", "Architecture"],
        value_vars=["Acoustic", "Gender", "Accent", "Phoneme"],
        var_name="Hierarchy", value_name="Mean_Peak"
    )

    # Compute bar heights (means) and SEs
    bar_df = (
        melted.groupby(["Architecture", "Hierarchy"])["Mean_Peak"]
        .mean()
        .reset_index()
    )

    se_df = (
        melted.groupby(["Architecture", "Hierarchy"])["Mean_Peak"]
        .sem()
        .reset_index()
    )

    se_map = {(r["Architecture"], r["Hierarchy"]): r["Mean_Peak"]
              for _, r in se_df.iterrows()}

    # Determine ordering using TRANSFORMER only
    transformer_order = (
        bar_df[bar_df["Architecture"] == "Transformer"]
        .sort_values("Mean_Peak")["Hierarchy"]
        .tolist()
    )

    print(f"Transformer-derived order: {transformer_order}")

    # Force architecture order
    arch_order = ["Transformer", "Conformer"]
    bar_df["Architecture"] = pd.Categorical(
        bar_df["Architecture"], categories=arch_order, ordered=True)

    # Plot
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        data=bar_df,
        x="Architecture",
        y="Mean_Peak",
        hue="Hierarchy",
        hue_order=transformer_order,
        palette="Set2",
        errorbar=None
    )

    # Add whiskers
    for bar, (_, row) in zip(ax.patches, bar_df.iterrows()):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height()
        se = se_map.get((row["Architecture"], row["Hierarchy"]), 0)

        ax.errorbar(
            x, y,
            yerr=se,
            fmt="none",
            ecolor="black",
            elinewidth=1.4,
            capsize=6,
            capthick=1.4
        )

    max_y = bar_df["Mean_Peak"].max()
    plt.ylim(0, max_y + 0.1)

    plt.ylabel("Mean Peak Layer (% of model depth)")
    plt.xlabel("Architecture")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Hierarchy")
    ax.set_title("")

    # Save
    output_path = output_dir / "fig 2.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Figure 2 to: {output_path}")


# =============================================================================
# Figure 3: LOWESS Regression with Bootstrap CI
# =============================================================================

def bootstrap_lowess(x, y, grid, n_boot=200, frac=0.25):
    """Compute LOWESS with bootstrapped confidence intervals."""
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


def generate_fig3_regression(root: Path, output_dir: Path):
    """Generate Figure 3: Smoothed regression lines with bootstrap CI."""
    print("\n" + "="*60)
    print("Generating Figure 3: LOWESS Regression (this may take a minute...)")
    print("="*60)

    # Load data
    csv_path = root / "ALL_PROBING_WITH_PHONEMES.csv"
    if not csv_path.exists():
        csv_path = root / "probing_ALL_TestScores.csv"

    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded data: {df.shape}")

    # Add architecture column
    def infer_arch(model):
        m = model.lower()
        if "canary" in m or "parakeet" in m:
            return "Conformer"
        return "Transformer"

    df["Architecture"] = df["Model"].apply(infer_arch)

    # Model depth map
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

    def get_depth(model):
        if model in true_layer_counts:
            return true_layer_counts[model]
        return int(df[df["Model"] == model]["Layer"].max()) + 1

    df["Total_Layers"] = df["Model"].apply(get_depth)
    df["Layer_Normalized"] = df["Layer"] / (df["Total_Layers"] - 1)

    # Define acoustic features
    acoustic = [
        "f0_min","f0_mean","f0_max","f0_median",
        "F1_min","F1_mean","F1_max","F1_median",
        "F2_min","F2_mean","F2_max","F2_median",
        "F3_min","F3_mean","F3_max","F3_median",
        "intensity_min","intensity_mean","intensity_max","intensity_median",
        "duration"
    ]

    # Feature grouping
    df["Feature_Group"] = "Other"
    df.loc[df.Feature.isin(acoustic), "Feature_Group"] = "Acoustic"
    df.loc[df.Feature == "gender", "Feature_Group"] = "Gender"
    df.loc[df.Feature == "l1_background", "Feature_Group"] = "Accent"
    df.loc[df.Feature == "phoneme", "Feature_Group"] = "Phoneme"

    keep = ["Acoustic", "Gender", "Accent", "Phoneme"]
    df = df[df.Feature_Group.isin(keep)]

    # Plot
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
        print(f"  Processing {arch}...")

        for feat in keep:
            d = sub[sub.Feature_Group == feat]
            x = d["Layer_Normalized"].values
            y = d["Test_Score"].values

            # MIN–MAX NORMALIZATION
            if len(y) > 0:
                y_min, y_max = y.min(), y.max()
                y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
            else:
                y_norm = y

            # Bootstrap LOWESS
            mean, lo, hi = bootstrap_lowess(x, y_norm, grid)

            # Plot
            ax.plot(grid, mean, color=palette[feat], lw=2)
            ax.fill_between(grid, lo, hi, color=palette[feat], alpha=0.2)

        # Axis labels
        ax.set_xlabel("Normalized Layer Depth")
        if i == 0:
            ax.set_ylabel("Normalized Probing Score")

        # Subplot titles
        ax.text(0.5, 1.03, arch, ha="center", va="bottom",
                fontsize=12, transform=ax.transAxes)

        # Aesthetics
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Legend
    fig.legend(
        [plt.Line2D([], [], color=palette[k], lw=2) for k in keep],
        keep,
        loc="upper center",
        ncol=4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save
    output_path = output_dir / "fig 3.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Figure 3 to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures from Google Drive synced data."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root path to Google Drive data folder (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./figs",
        help="Output directory for figures (default: ./figs)"
    )
    parser.add_argument(
        "--fig",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which figures to generate (default: 1 2 3)"
    )

    args = parser.parse_args()

    # Determine root path
    if args.root:
        root = Path(args.root)
    else:
        root = get_default_root()

    print(f"📁 Data root: {root}")

    if not root.exists():
        print(f"❌ Error: Root path does not exist: {root}")
        print("Please specify the correct path with --root or ensure Google Drive Desktop is synced.")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Setup fonts
    setup_fonts(root)

    # Generate requested figures
    if 1 in args.fig:
        try:
            generate_fig1_tsne(root, output_dir)
        except Exception as e:
            print(f"❌ Error generating Figure 1: {e}")
            import traceback
            traceback.print_exc()

    if 2 in args.fig:
        try:
            generate_fig2_peak_position(root, output_dir)
        except Exception as e:
            print(f"❌ Error generating Figure 2: {e}")
            import traceback
            traceback.print_exc()

    if 3 in args.fig:
        try:
            generate_fig3_regression(root, output_dir)
        except Exception as e:
            print(f"❌ Error generating Figure 3: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
