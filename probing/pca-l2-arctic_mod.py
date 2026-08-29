# === Probing with PCA bottleneck (5D/10D), with special speaker split for L2-Arctic (gender + l1_background) ===


import os, pickle, numpy as np, pandas as pd, gc
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


# =====================
# CONFIG
# =====================
ROOT_DIR   = "/content/drive/MyDrive/t-SNE & Probing"
PKL_ROOT   = "/content/drive/MyDrive/Layer Representations"
CSV_FOLDER = os.path.join(ROOT_DIR, "probing csv")
os.makedirs(CSV_FOLDER, exist_ok=True)


categorical_features = ["gender", "l1_background"]


model_tags = [
   'speechbrain-loq'
]


dataset_keys = [
    "cam_assess",
    "SAA",
    "l2-arctic-dataset-250",
    "sandi",
    "CommonVoice_accent_stratified",
    "cmu-arctic-train",
    "ALLSSTAR_2"
]


PCA_DIMS     = [5, 10]
PCA_CACHE_K  = 10
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_STRATIFY_TRIES = 20


SHIFT_MODELS = {
    'parakeet-tdt-0.6b-v2','canary-1b','canary-1b-flash',
    'canary-qwen-2.5b','granite-speech-3.3-2b','Phi-4-multimodal-instruct'
}


# =====================
# HELPERS
# =====================
def _mask_valid(y):
    y = np.asarray(y, dtype=object)
    mask = np.array([v is not None for v in y])
    mask &= ~pd.isna(y)
    return mask


def stratified_split_robust(y, test_size=0.2, random_state=42, max_tries=20):
    """Standard stratified split with retries to ensure both sets have all classes."""
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


def speaker_split_balanced(labels, mask=None, train_n=10, test_n=5, random_state=42, feat=None, max_tries=100):
    """
    Speaker-level split for l2-arctic: no speaker overlap AND both train/test
    contain >1 class for the given feature (feat). Retries up to max_tries.
    """
    rng = np.random.RandomState(random_state)
    speakers = np.array(labels["speaker"])
    if mask is not None:
        speakers = speakers[mask]
        idx_all = np.where(mask)[0]
    else:
        idx_all = np.arange(len(speakers))


    unique_speakers = np.unique(speakers)
    y = (np.array(labels[feat])[mask].astype(str) if (feat is not None and mask is not None)
         else (np.array(labels[feat]).astype(str) if feat is not None else None))


    last_train_idx, last_test_idx = None, None
    for _ in range(max_tries):
        rng.shuffle(unique_speakers)
        train_speakers = set(unique_speakers[:train_n])
        test_speakers  = set(unique_speakers[train_n:train_n+test_n])


        train_idx = idx_all[np.isin(speakers, list(train_speakers))]
        test_idx  = idx_all[np.isin(speakers, list(test_speakers))]


        last_train_idx, last_test_idx = train_idx, test_idx
        if feat is None:
            return train_idx, test_idx


        y_train = y[np.isin(speakers, list(train_speakers))]
        y_test  = y[np.isin(speakers, list(test_speakers))]


        # Need at least 2 classes in both sets (classification viability)
        if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            return train_idx, test_idx


    print(f"⚠️ Could not guarantee full class coverage for {feat} after {max_tries} tries; using last attempt.")
    return last_train_idx, last_test_idx


def _fit_probe(X_train, y_train, X_test, is_categorical):
    if is_categorical:
        if len(np.unique(y_train)) < 2:
            return None, None
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train, y_train)
        return clf.predict(X_train), clf.predict(X_test)
    else:
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg.predict(X_train), reg.predict(X_test)


def _score(y_train, yhat_train, y_test, yhat_test, is_categorical):
    if yhat_train is None:
        return np.nan, np.nan
    if is_categorical:
        return accuracy_score(y_train, yhat_train), accuracy_score(y_test, yhat_test)
    else:
        return r2_score(y_train, yhat_train), r2_score(y_test, yhat_test)


# =====================
# PROBING LOOP
# =====================
def run_probing(labels, model_tag, dataset_key, reps_by_layer, results):
    for feat in reversed(list(labels.keys())):
        if feat == "speaker":   # 🚨 Skip probing speaker itself
            continue


        print(f"running: {feat}")
        y_raw = np.array(labels[feat], dtype=object)
        if y_raw is None or len(y_raw) == 0:
            continue
        mask = _mask_valid(y_raw)
        if mask.sum() == 0:
            continue


        y = y_raw[mask]
        is_categorical = (feat in categorical_features)
        if is_categorical:
            y = y.astype(str)


        # === Special handling for l2-arctic (gender + l1_background) with balanced speaker split ===
        if dataset_key == "l2-arctic-dataset-250" and feat in {"gender", "l1_background"}:
            train_idx, test_idx = speaker_split_balanced(
                labels, mask=mask, train_n=10, test_n=5, random_state=RANDOM_STATE, feat=feat
            )
        else:
            if is_categorical:
                train_idx, test_idx = stratified_split_robust(y, TEST_SIZE, RANDOM_STATE, MAX_STRATIFY_TRIES)
            else:
                idx_all = np.arange(len(y))
                train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)


        n_layers = len(reps_by_layer)
        for layer_idx in tqdm(range(n_layers), desc=f"{dataset_key} - {feat}", unit="layer"):
            X_full = np.asarray(reps_by_layer[layer_idx])
            X = X_full[mask]
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]


            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_test_scaled  = scaler.transform(X_test_raw)


            pca = PCA(n_components=PCA_CACHE_K, svd_solver="randomized", random_state=RANDOM_STATE)
            X_train_k = pca.fit_transform(X_train_scaled)
            X_test_k  = pca.transform(X_test_scaled)


            for dim in PCA_DIMS:
                train_score, te_score = np.nan, np.nan
                if dim <= X_train_k.shape[1]:
                    X_train = X_train_k[:, :dim]
                    X_test  = X_test_k[:, :dim]
                    if is_categorical:
                        # LabelEncoder trained on TRAIN only
                        le = LabelEncoder().fit(y[train_idx])
                        y_tr = le.transform(y[train_idx])


                        # Filter TEST to classes seen in TRAIN (avoids ValueError & empty results)
                        test_in_train = np.isin(y[test_idx], le.classes_)
                        if np.any(test_in_train):
                            X_test_use = X_test[test_in_train]
                            y_te = le.transform(y[test_idx][test_in_train])


                            # Only score if the filtered test still has variation
                            if len(np.unique(y_te)) >= 2 and len(np.unique(y_tr)) >= 2:
                                yhat_tr, yhat_te = _fit_probe(X_train, y_tr, X_test_use, True)
                                train_score, te_score = _score(y_tr, yhat_tr, y_te, yhat_te, True)
                        # else: leave NaNs (no valid test samples after filtering)
                    else:
                        if np.var(y[test_idx].astype(float)) > 0:
                            yhat_tr, yhat_te = _fit_probe(
                                X_train, y[train_idx].astype(float), X_test, False
                            )
                            train_score, te_score = _score(
                                y[train_idx].astype(float), yhat_tr,
                                y[test_idx].astype(float), yhat_te, False
                            )


                results.append(dict(
                    Layer=(layer_idx+1 if model_tag in SHIFT_MODELS else layer_idx),
                    Train_Score=train_score,
                    Test_Score=te_score,
                    Model=model_tag,
                    Dataset=dataset_key,
                    Feature=feat,
                    PCA_Dim=dim
                ))


        del y, y_raw
        gc.collect()


# =====================
# MAIN
# =====================
def main():
    for model_tag in model_tags:
        for dataset_key in dataset_keys:
            print(f"\n🔍 Running probing for model: {model_tag} on dataset: {dataset_key}")
            pkl_path = os.path.join(PKL_ROOT, model_tag, f"{dataset_key}.pkl")
            if not os.path.exists(pkl_path):
                print(f"❌ Missing .pkl: {pkl_path}")
                continue


            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            reps_by_layer, labels = data.get("reps_by_layer"), data.get("labels")
            if reps_by_layer is None or labels is None:
                print("⚠️ reps_by_layer or labels missing in pkl, skipping.")
                continue


            model_results = []
            run_probing(labels, model_tag, dataset_key, reps_by_layer, model_results)


            if model_results:
                df_out = pd.DataFrame(model_results)
                out_path = os.path.join(CSV_FOLDER, f"{dataset_key}_{model_tag}_probing.csv")
                df_out.to_csv(out_path, index=False)
                print(f"✅ Saved: {out_path}")
            else:
                print(f"⚠️ No results for model {model_tag} on dataset {dataset_key}")


if __name__ == "__main__":
    main()
