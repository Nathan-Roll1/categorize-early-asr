# ===============================================================
# 🧠 STREAMING-SAFE PHONEME PROBING (row-based adaptive version)
#  - Now supports independent run lists for models & datasets
# ===============================================================

import os, pickle, gc, numpy as np, pandas as pd, warnings
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ----- CONFIG -----
ROOT_DIR   = "/content/drive/MyDrive/t-SNE & Probing"
PKL_ROOT   = "/content/drive/MyDrive/Layer Representations"
OUT_FOLDER = os.path.join(ROOT_DIR, "phoneme probing csv")
os.makedirs(OUT_FOLDER, exist_ok=True)

PCA_TARGET   = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ----------------------------------------------------------------
# 🧩 Master dictionaries (don't touch unless updating metadata)
# ----------------------------------------------------------------
MODEL_META = {
    "parakeet-tdt-0.6b-v2": {"layers": 24},
    "w2v2-conformer": {"layers": 24},
    "whisper-large-v3-turbo": {"layers": 33},
    "granite-speech-3.3-2b": {"layers": 16},
    "Phi-4-multimodal-instruct": {"layers": 33},
}

DATASET_ROWS = {
    "cam_assess_phonemes": 9_771,
    "l2-arctic-dataset-250_phonemes": 117_613,
    "CommonVoice_accent_stratified_phonemes": 276_270,
    "cmu-arctic-train_phonemes": 495_238,
    "ALLSSTAR_2_phonemes": 518_445,
    "SAA_phonemes": 677_965,
}

# ----------------------------------------------------------------
# ✅ Select which models & datasets to actually run this time
# ----------------------------------------------------------------
RUN_MODELS  = ["granite-speech-3.3-2b"]          # e.g., ["parakeet-tdt-0.6b-v2"]
RUN_DATASETS = ["CommonVoice_accent_stratified_phonemes", "ALLSSTAR_2_phonemes", "SAA_phonemes",]                   # e.g., ["cmu-arctic-train_phonemes"]

# ----------------------------------------------------------------
# 🧩 Helper functions
# ----------------------------------------------------------------
def _mask_valid(y):
    y = np.asarray(y, dtype=object)
    return np.array([v is not None for v in y]) & ~pd.isna(y)

def stratified_split_robust(y, test_size=0.2, random_state=42, max_tries=20):
    y = np.asarray(y)
    idx_all = np.arange(len(y))
    _, counts = np.unique(y, return_counts=True)
    if np.min(counts) < 2:
        return train_test_split(idx_all, test_size=test_size, random_state=random_state)
    for i in range(max_tries):
        tr, te = train_test_split(idx_all, test_size=test_size,
                                  random_state=random_state+i, stratify=y)
        if len(np.unique(y[tr])) >= 2 and len(np.unique(y[te])) >= 2:
            return np.array(tr), np.array(te)
    return train_test_split(idx_all, test_size=test_size, random_state=random_state)

def save_results_csv(results, out_path):
    pd.DataFrame(results).to_csv(out_path, index=False)


# ----------------------------------------------------------------
# 🧠 Core probing function
# ----------------------------------------------------------------
def probe_streamed_layer(X_iter, phonemes, results_container,
                         model_tag, dataset_key, layer_idx):
    mask = _mask_valid(phonemes)
    y_all = np.array(phonemes, dtype=str)[mask]
    le = LabelEncoder().fit(y_all)

    tr_mask, te_mask = stratified_split_robust(y_all, TEST_SIZE, RANDOM_STATE)
    idx_valid = np.nonzero(mask)[0]
    tr_idx_global, te_idx_global = idx_valid[tr_mask], idx_valid[te_mask]

    scaler = StandardScaler()
    ipca   = IncrementalPCA(n_components=PCA_TARGET)

    # ---- PASS 1 ----
    row_cursor = 0
    print(f"🌀 Layer {layer_idx}: Pass 1 (fit PCA/scaler)")
    for arr in X_iter(reset=False):
        n = arr.shape[0]; end = row_cursor + n
        sel = [i - row_cursor for i in tr_idx_global if row_cursor <= i < end]
        if sel:
            Xb = arr[sel]
            scaler.partial_fit(Xb)
            ipca.partial_fit(scaler.transform(Xb))
        row_cursor = end
        del arr; gc.collect()

    # ---- PASS 2 ----
    row_cursor = 0
    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []
    print(f"🌀 Layer {layer_idx}: Pass 2 (transform & classify)")
    for arr in X_iter(reset=True):
        n = arr.shape[0]; end = row_cursor + n
        idxs_tr = [i - row_cursor for i in tr_idx_global if row_cursor <= i < end]
        idxs_te = [i - row_cursor for i in te_idx_global if row_cursor <= i < end]
        if idxs_tr:
            X_train_parts.append(ipca.transform(scaler.transform(arr[idxs_tr])))
            y_train_parts.append(le.transform(np.array(phonemes)[tr_idx_global[
                np.isin(tr_idx_global, range(row_cursor, end))]]))
        if idxs_te:
            X_test_parts.append(ipca.transform(scaler.transform(arr[idxs_te])))
            y_test_parts.append(le.transform(np.array(phonemes)[te_idx_global[
                np.isin(te_idx_global, range(row_cursor, end))]]))
        row_cursor = end
        del arr; gc.collect()

    if not X_train_parts or not X_test_parts:
        print(f"⚠️ Skipping layer {layer_idx} (empty splits)")
        return

    X_train = np.vstack(X_train_parts); y_train = np.concatenate(y_train_parts)
    X_test  = np.vstack(X_test_parts);  y_test  = np.concatenate(y_test_parts)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    res = {
        "Layer": layer_idx,
        "Train_Score": accuracy_score(y_train, clf.predict(X_train)),
        "Test_Score":  accuracy_score(y_test,  clf.predict(X_test)),
        "Model": model_tag,
        "Dataset": dataset_key,
        "Feature": "phoneme",
        "PCA_Dim": PCA_TARGET
    }
    results_container.append(res)
    print(f"✅ Layer {layer_idx}: Test = {res['Test_Score']:.4f}")


# ----------------------------------------------------------------
# 🚀 Main loop (uses RUN_MODELS and RUN_DATASETS)
# ----------------------------------------------------------------
for model_tag in RUN_MODELS:
    if model_tag not in MODEL_META:
        print(f"⚠️ Unknown model: {model_tag}")
        continue
    num_layers = MODEL_META[model_tag]["layers"]

    for dataset_key in RUN_DATASETS:
        if dataset_key not in DATASET_ROWS:
            print(f"⚠️ Unknown dataset: {dataset_key}")
            continue
        n_rows = DATASET_ROWS[dataset_key]

        out_name = f"{dataset_key}__{model_tag}__phoneme_pca{PCA_TARGET}.csv"
        out_path = os.path.join(OUT_FOLDER, out_name)
        if os.path.exists(out_path):
            print("⏩ SKIP exists:", out_name)
            continue

        pkl_path = os.path.join(PKL_ROOT, model_tag, "phonemes", f"{dataset_key}.pkl")
        if not os.path.exists(pkl_path):
            print("❌ Missing PKL:", pkl_path)
            continue

        print(f"\n📂 {model_tag} × {dataset_key} ({num_layers} layers, {n_rows:,} phonemes)")
        results = []

        with open(pkl_path, "rb") as f:
            header = pickle.load(f)
            phonemes = header["labels"]["phoneme"]

            # --- Read and group chunks adaptively per layer ---
            for li in range(num_layers):
                layer_chunks, total_rows = [], 0
                while total_rows < n_rows * 0.98:
                    try:
                        chunk = pickle.load(f)
                    except EOFError:
                        break
                    if not isinstance(chunk, np.ndarray):
                        continue
                    layer_chunks.append(chunk)
                    total_rows += chunk.shape[0]
                    if total_rows >= n_rows * 1.05:
                        break  # small overshoot tolerance

                if not layer_chunks:
                    print(f"⚠️ No chunks found for layer {li}, stopping.")
                    break

                def X_iter(reset=False, chunks=layer_chunks):
                    for c in chunks:
                        yield c.astype(np.float32)

                probe_streamed_layer(X_iter, phonemes, results,
                                     model_tag, dataset_key, li)

        if results:
            save_results_csv(results, out_path)
            print("✅ Saved:", out_path)
        else:
            print(f"⚠️ No results for {model_tag} {dataset_key}")
        gc.collect()
