# PKL-based streaming PCA=5 phoneme probing


import os, pickle, gc, numpy as np, pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ----- CONFIG -----
ROOT_DIR   = "/content/drive/MyDrive/t-SNE & Probing"
PKL_ROOT   = "/content/drive/MyDrive/Layer Representations"
OUT_FOLDER = os.path.join(ROOT_DIR, "phoneme probing csv")
os.makedirs(OUT_FOLDER, exist_ok=True)


model_tags = [
   'whisper-tiny.en','whisper-tiny','whisper-base.en','whisper-base',
   'whisper-small.en','whisper-small','whisper-medium.en','whisper-medium',
   'whisper-large','whisper-large-v2','whisper-large-v3','whisper-large-v3-turbo',
   'parakeet-tdt-0.6b-v2','canary-1b','canary-1b-flash','canary-qwen-2.5b',
   'granite-speech-3.3-2b','Phi-4-multimodal-instruct',
   "hubert-large-ls960-ft","hubert-xlarge-ls960-ft",
   "wav2vec2-large-960h-lv60","wavlm-large", "w2v2-conformer", "speechbrain-loq"
]
dataset_keys = ["ALLSSTAR_2_phonemes", "SAA_phonemes"]


PCA_TARGET = 5
PCA_BATCH = 4096   # chunk size to run partial fits (tune)
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ----- HELPERS -----
def _mask_valid(y):
    y = np.asarray(y, dtype=object)
    mask = np.array([v is not None for v in y]) & ~pd.isna(y)
    return mask


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
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)


# ----- CORE: process one layer array (numpy 2D) -----
def probe_layer_array(layer_arr, phonemes, results_container, model_tag, dataset_key, layer_idx):
    """
    Validate and probe one layer's hidden representation matrix.
    Adds protection against NaN / inf / overflowed FP16 data.
    """


    # ---- 🩺 SANITY CHECK ----
    if layer_arr is None or not isinstance(layer_arr, np.ndarray):
        print(f"⚠️ Skipping layer {layer_idx}: not a valid numpy array")
        return


    nonfinite = np.sum(~np.isfinite(layer_arr))
    if nonfinite > 0:
        print(f"⚠️ Skipping layer {layer_idx}: {nonfinite:,} non-finite values (NaN/Inf)")
        results_container.append({
            "Layer": layer_idx,
            "Train_Score": np.nan,
            "Test_Score": np.nan,
            "Model": model_tag,
            "Dataset": dataset_key,
            "Feature": "phoneme",
            "PCA_Dim": PCA_TARGET,
            "Status": "CORRUPT"
        })
        return


    layer_std = float(np.std(layer_arr))
    layer_mean = float(np.mean(layer_arr))
    if layer_std > 10 or abs(layer_mean) > 5:
        print(f"⚠️ Layer {layer_idx}: abnormal stats (mean={layer_mean:.2f}, std={layer_std:.2f}) — likely overflow, skipping")
        results_container.append({
            "Layer": layer_idx,
            "Train_Score": np.nan,
            "Test_Score": np.nan,
            "Model": model_tag,
            "Dataset": dataset_key,
            "Feature": "phoneme",
            "PCA_Dim": PCA_TARGET,
            "Status": "CORRUPT"
        })
        return


    # ---- 🧠 Continue only if data looks valid ----
    mask = np.array([v is not None for v in phonemes]) & ~pd.isna(phonemes)
    if mask.sum() == 0:
        print(f"⚠️ Skipping layer {layer_idx}: no valid phoneme labels")
        return


    y_all = np.array(phonemes, dtype=str)[mask]
    le = LabelEncoder().fit(y_all)
    y_enc = le.transform(y_all)


    tr_idx_mask, te_idx_mask = stratified_split_robust(y_all, TEST_SIZE, RANDOM_STATE)
    idx_valid = np.nonzero(mask)[0]
    tr_idx_global = idx_valid[tr_idx_mask]
    te_idx_global = idx_valid[te_idx_mask]


    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=PCA_TARGET)


    # --- train PCA on training data ---
    for start in range(0, len(tr_idx_global), PCA_BATCH):
        batch_idx = tr_idx_global[start:start+PCA_BATCH]
        Xb = layer_arr[batch_idx]
        scaler.partial_fit(Xb)
        Xs = scaler.transform(Xb)
        ipca.partial_fit(Xs)


    # --- build train and test sets ---
    def transform_batches(idxs):
        chunks = []
        for start in range(0, len(idxs), PCA_BATCH):
            batch_idx = idxs[start:start+PCA_BATCH]
            Xb = layer_arr[batch_idx]
            Xs = scaler.transform(Xb)
            Xp = ipca.transform(Xs)
            chunks.append(Xp)
        return np.vstack(chunks)


    X_train = transform_batches(tr_idx_global)
    X_test  = transform_batches(te_idx_global)
    y_train = le.transform(y_all[tr_idx_mask])
    y_test  = le.transform(y_all[te_idx_mask])


    # --- classifier ---
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    yhat_train = clf.predict(X_train)
    yhat_test  = clf.predict(X_test)
    train_acc  = accuracy_score(y_train, yhat_train)
    test_acc   = accuracy_score(y_test, yhat_test)


    results_container.append({
        "Layer": layer_idx,
        "Train_Score": train_acc,
        "Test_Score": test_acc,
        "Model": model_tag,
        "Dataset": dataset_key,
        "Feature": "phoneme",
        "PCA_Dim": PCA_TARGET,
        "Status": "OK"
    })




# ----- MAIN loop over datasets -> models, per-pkl processing one layer at a time -----
for dataset_key in dataset_keys:
    print("DATASET:", dataset_key)
    for model_tag in model_tags:
        out_name = f"{dataset_key}__{model_tag}__phoneme_pca{PCA_TARGET}.csv"
        out_path = os.path.join(OUT_FOLDER, out_name)
        if os.path.exists(out_path):
            print(" SKIP exists:", out_name); continue


        pkl_path = os.path.join(PKL_ROOT, model_tag, "phonemes", f"{dataset_key}.pkl")
        if not os.path.exists(pkl_path):
            print(" MISSING PKL:", pkl_path); continue


        print(" Processing:", model_tag, "|", dataset_key)
        results = []
        # open and handle both common pkl layouts:
        with open(pkl_path, "rb") as f:
            # Load header (dict) first if present
            header = pickle.load(f)
            if isinstance(header, dict) and ("labels" in header or "reps_by_layer" in header):
                labels = header.get("labels", {})
                phonemes = labels.get("phoneme", None)
                reps_by_layer = header.get("reps_by_layer", None)
                # If reps_by_layer is a list already materialized, we can iterate it layer-by-layer
                if isinstance(reps_by_layer, (list, tuple)) and len(reps_by_layer) > 0:
                    for li, arr in enumerate(reps_by_layer):
                        print(f"  layer {li} - array shape {getattr(arr,'shape',None)}")
                        arr = np.asarray(arr, dtype=np.float32)
                        probe_layer_array(arr, phonemes, results, model_tag, dataset_key, li)
                        del arr; gc.collect()
                else:
                    # reps_by_layer not materialized; we expect subsequent pickles to be layer arrays appended sequentially
                    li = 0
                    while True:
                        try:
                            arr = pickle.load(f)
                        except EOFError:
                            break
                        arr = np.asarray(arr, dtype=np.float32)
                        print(f"  loaded layer {li}, shape {arr.shape}")
                        probe_layer_array(arr, phonemes, results, model_tag, dataset_key, li)
                        del arr; li += 1; gc.collect()
            else:
                # Unusual PKL: try to interpret header as first layer array and treat labels missing
                print(" Unrecognized pkl header format; expecting dict with 'labels'. Skipping.")
                continue


        if results:
            save_results_csv(results, out_path)
            print(" ✅ Saved:", out_path)
        else:
            print(" ⚠️ No results for", model_tag, dataset_key)
        gc.collect()
