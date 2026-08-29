# =====================================================
# 🧠 HDF5 Streaming PCA Prober (no full RAM load)
# =====================================================

import os, gc, h5py, pickle, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

# -------------- CONFIG --------------
ROOT_DIR   = "/content/drive/MyDrive/t-SNE & Probing"
REP_ROOT   = "/content/drive/MyDrive/Layer Representations"
OUT_FOLDER = os.path.join(ROOT_DIR, "phoneme probing csv")
os.makedirs(OUT_FOLDER, exist_ok=True)

model_tags = ["whisper-large-v3-turbo"]
dataset_keys = ["ALLSSTAR_2_phonemes", "SAA_phonemes"]

PCA_TARGET = 5
PCA_BATCH  = 4096
TEST_SIZE  = 0.2
RANDOM_STATE = 42


# -------------- HELPERS --------------
def stratified_split(y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(y))
    tr, te = train_test_split(idx_all, test_size=test_size,
                              random_state=random_state, stratify=y)
    return tr, te

def save_results(results, path):
    pd.DataFrame(results).to_csv(path, index=False)


# -------------- STREAMING CORE --------------
def probe_layer_stream(h5_dset, phonemes, layer_idx, model_tag, dataset_key):
    """Process one HDF5 dataset in small chunks without loading whole array."""
    n_rows, n_dim = h5_dset.shape
    print(f"🔹 Streaming layer {layer_idx} ({n_rows:,} × {n_dim})")

    # Labels & split
    mask = np.array([p is not None for p in phonemes]) & ~pd.isna(phonemes)
    y_all = np.array(phonemes, dtype=str)[mask]
    le = LabelEncoder().fit(y_all)
    tr_idx, te_idx = stratified_split(y_all, TEST_SIZE, RANDOM_STATE)
    idx_valid = np.nonzero(mask)[0]
    tr_idx_global = idx_valid[tr_idx]
    te_idx_global = idx_valid[te_idx]

    # --- Fit scaler + PCA on training data ---
    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=PCA_TARGET)
    for start in tqdm(range(0, len(tr_idx_global), PCA_BATCH), desc=f"PCA-fit L{layer_idx}"):
        idxs = tr_idx_global[start:start+PCA_BATCH]
        sel = np.sort(idxs)                           # ✅ sorted slice
        Xb = h5_dset[sel, :]
        scaler.partial_fit(Xb)
        Xs = scaler.transform(Xb)
        ipca.partial_fit(Xs)
        del Xb, Xs; gc.collect()

    # --- Transform train/test batches ---
    def transform_stream(idxs):
        chunks = []
        for start in range(0, len(idxs), PCA_BATCH):
            sel = np.sort(idxs[start:start+PCA_BATCH])   # ✅ sorted slice
            Xb = h5_dset[sel, :]
            Xs = scaler.transform(Xb)
            Xp = ipca.transform(Xs)
            chunks.append(Xp)
            del Xb, Xs, Xp; gc.collect()
        return np.vstack(chunks)

    X_train = transform_stream(tr_idx_global)
    X_test  = transform_stream(te_idx_global)
    y_train = le.transform(y_all[tr_idx])
    y_test  = le.transform(y_all[te_idx])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    return dict(
        Layer=layer_idx,
        Train_Score=accuracy_score(y_train, clf.predict(X_train)),
        Test_Score=accuracy_score(y_test, clf.predict(X_test)),
        Model=model_tag,
        Dataset=dataset_key,
        Feature="phoneme",
        PCA_Dim=PCA_TARGET,
        Status="OK"
    )


# -------------- MAIN LOOP --------------
for dataset_key in dataset_keys:
    for model_tag in model_tags:
        base = os.path.join(REP_ROOT, model_tag, "phonemes", dataset_key)
        h5_path = base + ".h5" if os.path.exists(base + ".h5") else base + ".hdf5"
        if not os.path.exists(h5_path):
            print("❌ Missing:", h5_path)
            continue

        out_csv = os.path.join(OUT_FOLDER, f"{dataset_key}__{model_tag}__streamPCA.csv")
        if os.path.exists(out_csv):
            print("⏩ Skip exists:", out_csv)
            continue

        print(f"\n📂 Opening {h5_path}")
        results = []
        with h5py.File(h5_path, "r") as h5:
            # Phoneme labels
            if "labels/phoneme" in h5:
                phonemes = np.array(h5["labels/phoneme"]).astype(str)
            elif "phoneme" in h5:
                phonemes = np.array(h5["phoneme"]).astype(str)
            else:
                print("⚠️ No phoneme labels found.")
                continue

            # Layers
            layer_keys = sorted([k for k in h5.keys() if k.startswith("layer_")])
            print(f"Found {len(layer_keys)} layers")

            for li, k in enumerate(layer_keys):
                try:
                    dset = h5[k]
                    res = probe_layer_stream(dset, phonemes, li, model_tag, dataset_key)
                    results.append(res)
                except Exception as e:
                    print(f"⚠️ Layer {li} failed: {e}")
                    results.append(dict(Layer=li, Model=model_tag, Dataset=dataset_key,
                                        Status=f"Error: {type(e).__name__}"))
                gc.collect()

        save_results(results, out_csv)
        print(f"✅ Saved results → {out_csv}")
        gc.collect()
