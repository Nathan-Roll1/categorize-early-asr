"""
Figure 1: t-SNE visualization of Transformer vs Conformer representations
Modified: 4 panels per model, final layer colored by utterance with a curated
utterance-color swatch legend on the right.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import colorsys

# Try to import optional dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed.")

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Warning: 'rapidfuzz' not installed. Using exact text matching.")

# === Configuration ===
DATASET = "l2-arctic-dataset-250"
MODELS = ["whisper-large-v3-turbo", "canary-1b"]
FEATURE = "l1_background"

SELECTED_LAYERS_WHISPER = [1, 12, 24, 32]
SELECTED_LAYERS_CANARY = [1, 9, 18, 24]
DEPTHS = [0, 37.5, 75, 100]

# Fallback L1 mapping
L1_MAP = {
    0: "Arabic",
    1: "Hindi",
    2: "Korean",
    3: "Mandarin",
    4: "Spanish",
    5: "Vietnamese"
}

# === Path detection ===
def get_root_path():
    """Auto-detect project root."""
    possible_paths = [
        Path.home() / "Google Drive" / "My Drive" / "t-SNE & Probing",
        Path("/content/drive/MyDrive/t-SNE & Probing"),  # Colab
    ]
    for p in possible_paths:
        if p.exists():
            return p

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.exists():
        for gd in cloud_storage.glob("GoogleDrive-*"):
            candidate = gd / "My Drive" / "t-SNE & Probing"
            if candidate.exists():
                return candidate

    raise FileNotFoundError("Could not find project root path")

# === Font setup ===
def setup_fonts(root):
    """Load custom font or fallback to system font."""
    font_path = root / "fonts" / "HelveticaNeueMedium.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        font_name = prop.get_name()
    else:
        font_name = "Helvetica Neue"

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.sans-serif"] = [font_name, "Arial", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    return font_name

# === Data loading ===
def load_tsne_and_labels(root, model_tag, dataset_meta=None):
    """Load t-SNE coordinates and labels for a model."""
    tsne_path = root / "t-SNE" / model_tag / DATASET / f"{model_tag}_{DATASET}_{FEATURE}_tsne.pkl"

    # Prefer model-specific representation filename, but allow legacy fallback
    rep_candidates = [
        root / "Layer Representations" / model_tag / f"{DATASET}__{model_tag}.pkl",
        root / "Layer Representations" / model_tag / f"{DATASET}.pkl",
    ]

    rep_path = next((p for p in rep_candidates if p.exists()), None)

    print(f"📂 Loading t-SNE: {tsne_path}")
    with open(tsne_path, "rb") as f:
        tsne_layers = pickle.load(f)

    if rep_path is not None:
        print(f"📂 Loading labels: {rep_path}")
        with open(rep_path, "rb") as f:
            labels = pickle.load(f)["labels"]
    elif dataset_meta is not None:
        first_valid = next((l for l in tsne_layers if l is not None), None)
        n_samples = first_valid.shape[0] if first_valid is not None else len(dataset_meta)
        labels = {
            "l1_background": dataset_meta["l1_background"][:n_samples],
            "speaker": dataset_meta["speaker"][:n_samples],
            "text": dataset_meta["text"][:n_samples],
        }
    else:
        raise FileNotFoundError(f"Could not find labels for {model_tag}")

    return tsne_layers, labels

# === Color generation ===
def make_l1_colors(labels):
    """Generate colors based on L1 background with speaker variation."""
    accent_enc = LabelEncoder().fit(labels["l1_background"])
    accents = accent_enc.transform(labels["l1_background"])
    speaker_enc = LabelEncoder().fit(labels["speaker"])
    denom = max(1, len(speaker_enc.classes_) - 1)
    speaker_norm = speaker_enc.transform(labels["speaker"]) / denom

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
        # preserve your tuned boosts
        if acc == 1:
            factor *= 1.15
        elif acc == 3:
            factor *= 1.15
        elif acc == 4 and spk_norm > 0.8:
            factor *= 1.15
        colors.append(np.clip(base * factor, 0, 1))

    return np.array(colors), accent_enc

def build_global_utterance_map(labels_list, similarity_threshold=85):
    """Combine texts from all models and assign consistent cluster IDs."""
    all_texts = []
    for labels in labels_list:
        all_texts.extend(labels.get("text", []))
    all_texts = list(all_texts)

    unique_texts, cluster_map = [], {}
    for txt in all_texts:
        matched = False
        if HAS_RAPIDFUZZ:
            for cid, ut in enumerate(unique_texts):
                if fuzz.ratio(str(txt), str(ut)) > similarity_threshold:
                    cluster_map[txt] = cid
                    matched = True
                    break
        else:
            if txt in cluster_map:
                matched = True

        if not matched:
            cluster_map[txt] = len(unique_texts)
            unique_texts.append(txt)

    print(f"✅ Global utterance clusters: {len(unique_texts)} unique groups")
    return cluster_map, len(unique_texts)

def assign_utterance_colors(labels, cluster_map, n_clusters):
    """
    Generate discrete colors for each utterance cluster.
    """
    cluster_ids = np.array([cluster_map.get(txt, 0) for txt in labels["text"]])

    if n_clusters <= 20:
        cmap = plt.cm.get_cmap("tab20", n_clusters)
    elif n_clusters <= 40:
        cmap = plt.cm.get_cmap("tab20", 20)
        cmap2 = plt.cm.get_cmap("tab20b", 20)
    else:
        cmap = plt.cm.get_cmap("gist_ncar", n_clusters)

    rgb_colors = []
    for i in range(n_clusters):
        if n_clusters <= 20:
            rgb_colors.append(cmap(i)[:3])
        elif n_clusters <= 40:
            if i < 20:
                rgb_colors.append(cmap(i)[:3])
            else:
                rgb_colors.append(cmap2(i - 20)[:3])
        else:
            rgb_colors.append(cmap(i / n_clusters)[:3])

    colors = np.array([rgb_colors[cid % len(rgb_colors)] for cid in cluster_ids])
    return colors, rgb_colors

# === Main plotting function ===
def generate_figure(root, output_path):
    """Generate the Figure 1 t-SNE panel figure."""

    setup_fonts(root)

    dataset_meta = None
    l1_map = L1_MAP.copy()

    if HAS_DATASETS:
        try:
            dataset_meta = load_dataset("PranavBhalerao/l2-arctic-dataset-250", split="train")
            if hasattr(dataset_meta, "features") and "l1_background" in dataset_meta.features:
                names = dataset_meta.features["l1_background"].names
                if names:
                    l1_map = dict(enumerate(names))
            print("✅ Loaded dataset metadata from HuggingFace")
        except Exception as e:
            print(f"⚠️ Could not load dataset metadata: {e}")

    # Load model data
    whisper_tsne, whisper_labels = load_tsne_and_labels(root, "whisper-large-v3-turbo", dataset_meta)
    canary_tsne, canary_labels = load_tsne_and_labels(root, "canary-1b", dataset_meta)

    # Ensure text labels exist
    if dataset_meta is not None:
        if "text" not in whisper_labels or len(whisper_labels.get("text", [])) == 0:
            first_valid = next((l for l in whisper_tsne if l is not None), None)
            n = first_valid.shape[0] if first_valid is not None else len(dataset_meta)
            whisper_labels["text"] = list(dataset_meta["text"][:n])

        if "text" not in canary_labels or len(canary_labels.get("text", [])) == 0:
            first_valid = next((l for l in canary_tsne if l is not None), None)
            n = first_valid.shape[0] if first_valid is not None else len(dataset_meta)
            canary_labels["text"] = list(dataset_meta["text"][:n])

    # L1 colors
    whisper_l1_colors, whisper_enc = make_l1_colors(whisper_labels)
    canary_l1_colors, _ = make_l1_colors(canary_labels)

    # Utterance colors
    cluster_map, n_clusters = build_global_utterance_map([whisper_labels, canary_labels])
    whisper_utt_colors, utt_rgb_colors = assign_utterance_colors(whisper_labels, cluster_map, n_clusters)
    canary_utt_colors, _ = assign_utterance_colors(canary_labels, cluster_map, n_clusters)

    # === Create figure: 2 rows x 4 columns ===
    ncols = 4
    fig, axs = plt.subplots(
        2, ncols,
        figsize=(ncols * 3.3, 6),
        dpi=300,
        facecolor="white",
        gridspec_kw=dict(wspace=0.15, hspace=0.05)
    )

    for row, (tsne_layers, l1_colors, utt_colors, selected_layers) in enumerate([
        (whisper_tsne, whisper_l1_colors, whisper_utt_colors, SELECTED_LAYERS_WHISPER),
        (canary_tsne, canary_l1_colors, canary_utt_colors, SELECTED_LAYERS_CANARY),
    ]):
        for col, (layer_idx, depth) in enumerate(zip(selected_layers, DEPTHS)):
            if tsne_layers[layer_idx] is None:
                for offset in range(1, len(tsne_layers)):
                    if layer_idx + offset < len(tsne_layers) and tsne_layers[layer_idx + offset] is not None:
                        layer_idx = layer_idx + offset
                        break
                    if layer_idx - offset >= 0 and tsne_layers[layer_idx - offset] is not None:
                        layer_idx = layer_idx - offset
                        break

            coords = tsne_layers[layer_idx]
            ax = axs[row, col]

            if col < 3:
                colors_to_use = l1_colors
                alpha = 0.55
                title = f"Depth: {depth}%"
            else:
                colors_to_use = utt_colors
                alpha = 0.45
                title = "Depth: 100% (Utterance)"

            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=colors_to_use,
                s=10,
                edgecolors="black",
                linewidths=0.25,
                alpha=alpha,
                rasterized=True
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(title, fontsize=12, pad=18, weight="medium")

    # === Row labels ===
    fig.canvas.draw()
    bbox_whisper = axs[0, 0].get_position()
    bbox_canary = axs[1, 0].get_position()
    y_whisper = (bbox_whisper.y0 + bbox_whisper.y1) / 2
    y_canary = (bbox_canary.y0 + bbox_canary.y1) / 2
    x_label = bbox_whisper.x0 - 0.055

    fig.text(
        x_label, y_whisper, "Whisper-large-v3-turbo",
        ha="center", va="center",
        fontsize=12, weight="medium", rotation=90
    )
    fig.text(
        x_label, y_canary, "Canary-1B",
        ha="center", va="center",
        fontsize=12, weight="medium", rotation=90
    )

    # === Legends ===
    base_colors = np.array([
        (0.5, 0.5, 0.5),
        (0.121, 0.466, 0.705),
        (0.172, 0.627, 0.172),
        (0.839, 0.152, 0.156),
        (0.894, 0.466, 0.761),
        (1.000, 0.498, 0.054)
    ])[:len(whisper_enc.classes_)]

    handles_l1 = [
        plt.Line2D(
            [], [], marker="o", color="w",
            markerfacecolor=tuple(base_colors[i]),
            markeredgecolor="black",
            markersize=6,
            label=l1_map.get(i, f"ID {i}")
        )
        for i in range(len(whisper_enc.classes_))
    ]

    fig.legend(
        handles=handles_l1,
        loc="lower left",
        bbox_to_anchor=(0.15, 0.02),
        ncol=max(1, len(handles_l1) // 2),
        frameon=False,
        fontsize=9,
        title="L1 Background",
        title_fontsize=10
    )

    # === Utterance ID swatch legend on the right ===
    present_clusters = sorted(
        set(cluster_map.get(txt, 0) for txt in whisper_labels["text"]) |
        set(cluster_map.get(txt, 0) for txt in canary_labels["text"])
    )

    present_color_pairs = []
    for cid in present_clusters:
        rgb = np.array(utt_rgb_colors[cid])
        h, s, v = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        present_color_pairs.append((cid, rgb, h, s, v))

    present_color_pairs = sorted(present_color_pairs, key=lambda x: x[2])

    n_swatch = min(24, len(present_color_pairs))
    if len(present_color_pairs) <= n_swatch:
        selected_pairs = present_color_pairs
    else:
        idxs = np.linspace(0, len(present_color_pairs) - 1, n_swatch, dtype=int)
        selected_pairs = [present_color_pairs[i] for i in idxs]

    # Optional cleanup: remove two visually distracting outlier colors in legend only
    target_dark_green = np.array([0.00, 0.45, 0.20])
    target_pink = np.array([0.95, 0.45, 0.75])

    selected_rgbs = np.array([rgb for _, rgb, _, _, _ in selected_pairs])
    if len(selected_rgbs) >= 2:
        dist_green = np.linalg.norm(selected_rgbs - target_dark_green, axis=1)
        dist_pink = np.linalg.norm(selected_rgbs - target_pink, axis=1)
        remove_idx = {int(np.argmin(dist_green)), int(np.argmin(dist_pink))}

        selected_pairs = [pair for i, pair in enumerate(selected_pairs) if i not in remove_idx]

        used_cids = {cid for cid, _, _, _, _ in selected_pairs}
        remaining_pairs = [pair for pair in present_color_pairs if pair[0] not in used_cids]

        for pair in remaining_pairs:
            if len(selected_pairs) >= n_swatch:
                break
            selected_pairs.append(pair)

        selected_pairs = sorted(selected_pairs, key=lambda x: x[2])

    swatch_colors = np.array([rgb for _, rgb, _, _, _ in selected_pairs])[np.newaxis, :, :]

    bar_left = 0.79
    bar_bottom = 0.035
    bar_width = 0.16
    bar_height = 0.035

    swatch_ax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
    swatch_ax.imshow(swatch_colors, aspect="auto", interpolation="nearest")
    swatch_ax.set_xticks([])
    swatch_ax.set_yticks([])
    for spine in swatch_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    fig.text(
        bar_left + bar_width / 2,
        bar_bottom + bar_height + 0.008,
        "Utterance ID",
        ha="center",
        va="bottom",
        fontsize=10
    )

    # Layout
    plt.subplots_adjust(bottom=0.16, left=0.15, right=0.98, top=0.92)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"✅ Saved figure to: {output_path}")

# === Main ===
if __name__ == "__main__":
    import sys

    log_path = Path(__file__).parent / "fig1_log.txt"

    try:
        root = get_root_path()
        with open(log_path, "w") as f:
            f.write(f"Root path: {root}\n")
        print(f"📂 Using root: {root}")

        output_path = Path(__file__).parent / "fig 1.pdf"
        generate_figure(root, output_path)

        with open(log_path, "a") as f:
            f.write(f"Success! Output: {output_path}\n")

    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"Error: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"Error: {e}")
        sys.exit(1)
