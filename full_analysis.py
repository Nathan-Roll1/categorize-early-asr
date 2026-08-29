#!/usr/bin/env python3
"""
Full Analysis Script for ASR Paper

Computes all statistical values, tables, and metrics reported in the manuscript:
- Peak positions and standard deviations for each feature by architecture
- Statistical tests (t-tests, bootstrap CIs)
- Architectural fingerprinting (logistic regression, AUC)
- Regression controlling for model size
- Cross-hierarchy correlations
- Entropy calculations
- Strength-width correlations

Outputs results to a structured format for updating the manuscript.
"""

import os
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, linregress
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

def get_root():
    """Auto-detect Google Drive path."""
    home = Path.home()
    candidates = [
        home / "Google Drive" / "My Drive" / "t-SNE & Probing",
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
    return candidates[0]


# Model metadata
MODEL_INFO = {
    # Transformer models
    "whisper-tiny": {"arch": "Transformer", "params": 39e6, "family": "Whisper", "ssl": False},
    "whisper-tiny.en": {"arch": "Transformer", "params": 39e6, "family": "Whisper", "ssl": False},
    "whisper-base": {"arch": "Transformer", "params": 74e6, "family": "Whisper", "ssl": False},
    "whisper-base.en": {"arch": "Transformer", "params": 74e6, "family": "Whisper", "ssl": False},
    "whisper-small": {"arch": "Transformer", "params": 244e6, "family": "Whisper", "ssl": False},
    "whisper-small.en": {"arch": "Transformer", "params": 244e6, "family": "Whisper", "ssl": False},
    "whisper-medium": {"arch": "Transformer", "params": 769e6, "family": "Whisper", "ssl": False},
    "whisper-medium.en": {"arch": "Transformer", "params": 769e6, "family": "Whisper", "ssl": False},
    "whisper-large": {"arch": "Transformer", "params": 1.55e9, "family": "Whisper", "ssl": False},
    "whisper-large-v2": {"arch": "Transformer", "params": 1.55e9, "family": "Whisper", "ssl": False},
    "whisper-large-v3": {"arch": "Transformer", "params": 1.55e9, "family": "Whisper", "ssl": False},
    "whisper-large-v3-turbo": {"arch": "Transformer", "params": 809e6, "family": "Whisper", "ssl": False},
    "wav2vec2-large-960h-lv60": {"arch": "Transformer", "params": 317e6, "family": "Wav2Vec2", "ssl": True},
    "hubert-base-ls960": {"arch": "Transformer", "params": 95e6, "family": "HuBERT", "ssl": True},
    "hubert-large-ls960-ft": {"arch": "Transformer", "params": 317e6, "family": "HuBERT", "ssl": True},
    "hubert-xlarge-ls960-ft": {"arch": "Transformer", "params": 1e9, "family": "HuBERT", "ssl": True},
    "wavlm-base": {"arch": "Transformer", "params": 94e6, "family": "WavLM", "ssl": True},
    "wavlm-large": {"arch": "Transformer", "params": 317e6, "family": "WavLM", "ssl": True},
    "granite-speech-3.3-2b": {"arch": "Transformer", "params": 3.3e9, "family": "Granite", "ssl": False},
    "Phi-4-multimodal-instruct": {"arch": "Transformer", "params": 2.7e9, "family": "Phi", "ssl": False},
    # Conformer models
    "canary-1b": {"arch": "Conformer", "params": 1e9, "family": "Canary", "ssl": False},
    "canary-1b-flash": {"arch": "Conformer", "params": 1e9, "family": "Canary", "ssl": False},
    "canary-qwen2.5b": {"arch": "Conformer", "params": 2.5e9, "family": "Canary", "ssl": False},
    "parakeet-tdt-0.6b-v2": {"arch": "Conformer", "params": 600e6, "family": "Parakeet", "ssl": False},
    "parakeet-tdt-1.1b": {"arch": "Conformer", "params": 1.1e9, "family": "Parakeet", "ssl": False},
    "wav2vec2-conformer-rel-pos-large": {"arch": "Conformer", "params": 430e6, "family": "Wav2Vec2-Conformer", "ssl": True},
    "speechbrain-asr-conformer-loquacious": {"arch": "Conformer", "params": 135e6, "family": "SpeechBrain", "ssl": False},
}

# Feature groupings
ACOUSTIC_PREFIXES = ("f0_", "f1_", "f2_", "f3_", "f3_minus_f2_", "intensity_")
FEATURE_GROUPS = {
    "acoustic": ACOUSTIC_PREFIXES,
    "gender": ("gender_",),
    "accent": ("l1_background_",),
    "phoneme": ("phoneme_",),
    "duration": ("duration_",)
}


# =============================================================================
# Data Loading
# =============================================================================

def load_aggregate_data(root: Path):
    """Load the aggregate metrics CSV."""
    csv_path = root / "ALLDATA_aggregate_metrics (with phonemes).csv"
    if not csv_path.exists():
        csv_path = root / "aggregate metrics" / "ALLDATA_aggregate_metrics (with phonemes).csv"

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded aggregate data: {df.shape}")
    return df


def load_probing_data(root: Path):
    """Load the full probing results CSV."""
    csv_path = root / "ALL_PROBING_WITH_PHONEMES.csv"
    if not csv_path.exists():
        csv_path = root / "probing_ALL_TestScores.csv"

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded probing data: {df.shape}")
    return df


# =============================================================================
# Feature Extraction Helpers
# =============================================================================

def get_feature_cols(df, group, suffix="_peak_layer_percent"):
    """Get column names for a feature group."""
    cols = []
    for prefix in FEATURE_GROUPS[group]:
        cols.extend([c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)])
    return cols


def get_model_info(model_name):
    """Get model info, handling name variations."""
    # Try exact match first
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]

    # Try lowercase
    for key, info in MODEL_INFO.items():
        if key.lower() == model_name.lower():
            return info

    # Try partial match
    for key, info in MODEL_INFO.items():
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            return info

    # Default based on common patterns
    m = model_name.lower()
    if "canary" in m or "parakeet" in m or "conformer" in m:
        return {"arch": "Conformer", "params": None, "family": "Unknown", "ssl": False}
    return {"arch": "Transformer", "params": None, "family": "Unknown", "ssl": False}


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def compute_peak_stats(df):
    """Compute peak position statistics for each feature group by architecture."""
    results = defaultdict(dict)

    for group in FEATURE_GROUPS.keys():
        pos_cols = get_feature_cols(df, group, "_peak_layer_percent")
        score_cols = get_feature_cols(df, group, "_peak_score")

        if not pos_cols:
            print(f"⚠️ No columns found for {group}")
            continue

        # Compute per-model means
        df[f"{group}_mean_pos"] = df[pos_cols].mean(axis=1)
        df[f"{group}_mean_score"] = df[score_cols].mean(axis=1) if score_cols else np.nan

    # Split by architecture
    for arch in ["Transformer", "Conformer"]:
        arch_df = df[df["Architecture"] == arch]
        n = len(arch_df)

        for group in FEATURE_GROUPS.keys():
            col = f"{group}_mean_pos"
            if col not in df.columns:
                continue

            values = arch_df[col].dropna()
            score_col = f"{group}_mean_score"
            scores = arch_df[score_col].dropna() if score_col in df.columns else pd.Series()

            results[group][arch] = {
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "ci_low": np.percentile(values, 2.5) if len(values) > 1 else np.nan,
                "ci_high": np.percentile(values, 97.5) if len(values) > 1 else np.nan,
                "score_mean": scores.mean() if len(scores) > 0 else np.nan,
                "score_std": scores.std() if len(scores) > 0 else np.nan,
            }

    return results, df


def compute_t_tests(df, results):
    """Compute t-tests between architectures for each feature group."""
    t_test_results = {}

    for group in FEATURE_GROUPS.keys():
        col = f"{group}_mean_pos"
        if col not in df.columns:
            continue

        trans = df[df["Architecture"] == "Transformer"][col].dropna()
        conf = df[df["Architecture"] == "Conformer"][col].dropna()

        if len(trans) < 2 or len(conf) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(trans, conf)
        delta = conf.mean() - trans.mean()

        t_test_results[group] = {
            "delta": delta,
            "t_stat": t_stat,
            "p_value": p_val,
            "df": len(trans) + len(conf) - 2
        }

    return t_test_results


def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    """Compute bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    return np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])


def compute_bootstrap_deltas(df, n_boot=10000):
    """Compute bootstrap CIs for architecture differences."""
    bootstrap_results = {}

    for group in FEATURE_GROUPS.keys():
        col = f"{group}_mean_pos"
        if col not in df.columns:
            continue

        trans = df[df["Architecture"] == "Transformer"][col].dropna().values
        conf = df[df["Architecture"] == "Conformer"][col].dropna().values

        if len(trans) < 2 or len(conf) < 2:
            continue

        # Bootstrap the difference
        deltas = []
        for _ in range(n_boot):
            t_sample = np.random.choice(trans, size=len(trans), replace=True)
            c_sample = np.random.choice(conf, size=len(conf), replace=True)
            deltas.append(c_sample.mean() - t_sample.mean())

        bootstrap_results[group] = {
            "delta_mean": np.mean(deltas),
            "ci_low": np.percentile(deltas, 2.5),
            "ci_high": np.percentile(deltas, 97.5)
        }

    return bootstrap_results


def compute_hierarchy_range(df):
    """Compute gender-to-duration range for each architecture."""
    results = {}

    for arch in ["Transformer", "Conformer"]:
        arch_df = df[df["Architecture"] == arch]

        gender_col = "gender_mean_pos"
        duration_col = "duration_mean_pos"

        if gender_col in arch_df.columns and duration_col in arch_df.columns:
            ranges = arch_df[duration_col] - arch_df[gender_col]
            results[arch] = {
                "mean_range": ranges.mean(),
                "std_range": ranges.std(),
                "values": ranges.values
            }

    # T-test for range difference
    if "Transformer" in results and "Conformer" in results:
        t_stat, p_val = stats.ttest_ind(
            results["Transformer"]["values"],
            results["Conformer"]["values"]
        )
        results["t_test"] = {"t_stat": t_stat, "p_value": p_val}

    return results


def compute_architectural_classifier(df):
    """Train logistic regression to classify architecture from peak positions."""
    feature_cols = ["acoustic_mean_pos", "gender_mean_pos", "accent_mean_pos",
                    "phoneme_mean_pos", "duration_mean_pos"]

    # Check which columns exist
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < 3:
        print(f"⚠️ Not enough feature columns for classifier: {available_cols}")
        return None

    # Prepare data
    X = df[available_cols].dropna()
    valid_idx = X.index
    y = (df.loc[valid_idx, "Architecture"] == "Conformer").astype(int)
    X = X.values

    if len(y) < 5:
        print("⚠️ Not enough samples for classifier")
        return None

    # Leave-one-out cross-validation
    aucs = []
    predictions = []
    true_labels = []

    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y.values, i)
        X_test = X[i:i+1]
        y_test = y.values[i]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        prob = clf.predict_proba(X_test_scaled)[0, 1]
        predictions.append(prob)
        true_labels.append(y_test)

    # Compute AUC
    auc = roc_auc_score(true_labels, predictions)

    # Fit final model for coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_scaled, y)

    # Bootstrap AUC CI
    auc_boots = []
    for _ in range(1000):
        idx = np.random.choice(len(predictions), size=len(predictions), replace=True)
        boot_true = [true_labels[i] for i in idx]
        boot_pred = [predictions[i] for i in idx]
        if len(set(boot_true)) > 1:  # Need both classes
            auc_boots.append(roc_auc_score(boot_true, boot_pred))

    return {
        "auc": auc,
        "auc_ci_low": np.percentile(auc_boots, 2.5) if auc_boots else np.nan,
        "auc_ci_high": np.percentile(auc_boots, 97.5) if auc_boots else np.nan,
        "coefficients": dict(zip(available_cols, clf.coef_[0])),
        "feature_names": available_cols
    }


def compute_regression_controlling_size(df):
    """Fit regression controlling for model size."""
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm

    results = {}

    # Create architecture dummy and log params
    df = df.copy()
    df["is_conformer"] = (df["Architecture"] == "Conformer").astype(int)

    # Get param count from model info if not in data, or convert to numeric
    if "Param_Count" in df.columns:
        df["Param_Count"] = pd.to_numeric(df["Param_Count"], errors='coerce')

    if "Param_Count" not in df.columns or df["Param_Count"].isna().all():
        df["Param_Count"] = df["Model"].apply(
            lambda m: get_model_info(m).get("params", np.nan)
        )

    df["log_params"] = np.log(df["Param_Count"].astype(float))
    df = df.dropna(subset=["log_params"])

    for group in FEATURE_GROUPS.keys():
        col = f"{group}_mean_pos"
        if col not in df.columns:
            continue

        subset = df.dropna(subset=[col, "is_conformer", "log_params"])
        if len(subset) < 5:
            continue

        X = subset[["is_conformer", "log_params"]]
        X = sm.add_constant(X)
        y = subset[col]

        try:
            model = sm.OLS(y, X).fit()
            results[group] = {
                "beta_arch": model.params.get("is_conformer", np.nan),
                "p_arch": model.pvalues.get("is_conformer", np.nan),
                "beta_size": model.params.get("log_params", np.nan),
                "p_size": model.pvalues.get("log_params", np.nan),
                "r_squared": model.rsquared
            }
        except Exception as e:
            print(f"⚠️ Regression failed for {group}: {e}")

    return results


def compute_cross_correlations(df):
    """Compute cross-feature correlations within each architecture."""
    results = {}

    feature_cols = ["acoustic_mean_pos", "gender_mean_pos", "accent_mean_pos",
                    "phoneme_mean_pos", "duration_mean_pos"]
    available_cols = [c for c in feature_cols if c in df.columns]

    for arch in ["Transformer", "Conformer"]:
        arch_df = df[df["Architecture"] == arch].copy()

        # Need log_params for partial correlation
        if "log_params" not in arch_df.columns:
            if "Param_Count" in arch_df.columns:
                arch_df["Param_Count"] = pd.to_numeric(arch_df["Param_Count"], errors='coerce')
            else:
                arch_df["Param_Count"] = arch_df["Model"].apply(
                    lambda m: get_model_info(m).get("params", np.nan)
                )
            arch_df["log_params"] = np.log(arch_df["Param_Count"].astype(float))

        arch_results = {}

        # Key pairs of interest
        pairs = [
            ("gender_mean_pos", "accent_mean_pos"),
            ("acoustic_mean_pos", "duration_mean_pos"),
            ("accent_mean_pos", "duration_mean_pos")
        ]

        for x_col, y_col in pairs:
            if x_col not in available_cols or y_col not in available_cols:
                continue

            subset = arch_df.dropna(subset=[x_col, y_col, "log_params"])
            if len(subset) < 5:
                continue

            # Simple Spearman correlation
            rho, p = spearmanr(subset[x_col], subset[y_col])

            # Partial correlation controlling for params
            try:
                # Rank transform for Spearman
                ranked = subset[[x_col, y_col, "log_params"]].rank()
                # Residualize
                bx = linregress(ranked["log_params"], ranked[x_col])
                by = linregress(ranked["log_params"], ranked[y_col])
                rx = ranked[x_col] - (bx.slope * ranked["log_params"] + bx.intercept)
                ry = ranked[y_col] - (by.slope * ranked["log_params"] + by.intercept)
                partial_rho, partial_p = spearmanr(rx, ry)
            except:
                partial_rho, partial_p = np.nan, np.nan

            arch_results[f"{x_col}_vs_{y_col}"] = {
                "rho": rho,
                "p": p,
                "partial_rho": partial_rho,
                "partial_p": partial_p,
                "n": len(subset)
            }

        results[arch] = arch_results

    return results


def compute_strength_width_correlation(df_probing, df_agg):
    """Compute correlation between peak strength and peak width."""
    results = {}

    # For probing data, compute width
    # Width = proportion of layers where score >= 0.7 * peak_score

    # First, let's use the aggregate data which has peak positions
    for arch in ["Transformer", "Conformer"]:
        arch_df = df_agg[df_agg["Architecture"] == arch]

        # Collect all (strength, width) pairs across features and models
        strengths = []
        widths = []

        for group in FEATURE_GROUPS.keys():
            score_cols = get_feature_cols(df_agg, group, "_peak_score")
            # We don't have width directly, so we'll compute a proxy
            # For now, we'll skip this or use entropy as a proxy

        # Use peak score variance as a proxy for width
        # Higher variance = more concentrated = narrower

    # This analysis requires the full layer-wise data
    # Let's compute from probing data
    if df_probing is not None and len(df_probing) > 0:
        for arch in ["Transformer", "Conformer"]:
            arch_probing = df_probing[df_probing["Model"].apply(
                lambda m: get_model_info(m)["arch"] == arch
            )]

            if len(arch_probing) == 0:
                continue

            # Group by model and feature, compute peak and width
            model_feature_stats = []

            for (model, feature), group_df in arch_probing.groupby(["Model", "Feature"]):
                if len(group_df) < 3:
                    continue

                scores = group_df["Test_Score"].values
                peak_score = scores.max()

                # Width = proportion >= 0.7 * peak
                width = (scores >= 0.7 * peak_score).mean()

                model_feature_stats.append({
                    "model": model,
                    "feature": feature,
                    "peak_score": peak_score,
                    "width": width
                })

            if len(model_feature_stats) < 5:
                continue

            stats_df = pd.DataFrame(model_feature_stats)
            r, p = pearsonr(stats_df["peak_score"], stats_df["width"])

            results[arch] = {
                "r": r,
                "p": p,
                "n": len(stats_df)
            }

    return results


def compute_entropy(df_probing):
    """Compute layer-wise entropy for each feature and architecture."""
    results = defaultdict(lambda: defaultdict(list))

    if df_probing is None or len(df_probing) == 0:
        return results

    for (model, feature), group_df in df_probing.groupby(["Model", "Feature"]):
        arch = get_model_info(model)["arch"]

        scores = group_df["Test_Score"].values
        if len(scores) < 3 or scores.sum() <= 0:
            continue

        # Normalize to probability distribution
        scores_pos = np.maximum(scores, 0)  # Ensure non-negative
        if scores_pos.sum() == 0:
            continue
        q = scores_pos / scores_pos.sum()

        # Compute entropy
        q_nonzero = q[q > 0]
        entropy = -np.sum(q_nonzero * np.log(q_nonzero))

        # Map feature to group
        feature_lower = feature.lower()
        for group, prefixes in FEATURE_GROUPS.items():
            for prefix in prefixes:
                if feature_lower.startswith(prefix.rstrip("_")) or feature_lower == prefix.rstrip("_"):
                    results[arch][group].append(entropy)
                    break

    # Compute statistics
    entropy_stats = {}
    for arch in results:
        entropy_stats[arch] = {}
        for group in results[arch]:
            values = results[arch][group]
            if len(values) > 0:
                entropy_stats[arch][group] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values)
                }

    return entropy_stats


def compute_hierarchy_consistency(df):
    """Check how many models follow the canonical hierarchy."""
    results = {"Transformer": {"follows": 0, "total": 0, "exceptions": []},
               "Conformer": {"follows": 0, "total": 0, "exceptions": []}}

    for _, row in df.iterrows():
        arch = row["Architecture"]
        model = row["Model"]

        # Get positions
        acoustic = row.get("acoustic_mean_pos", np.nan)
        gender = row.get("gender_mean_pos", np.nan)
        accent = row.get("accent_mean_pos", np.nan)
        phoneme = row.get("phoneme_mean_pos", np.nan)
        duration = row.get("duration_mean_pos", np.nan)

        if any(pd.isna([acoustic, gender, accent, duration])):
            continue

        results[arch]["total"] += 1

        if arch == "Transformer":
            # Expected: Acoustic < Gender < {Accent ≈ Duration}
            follows = (acoustic < gender) and (gender < accent) and (gender < duration)
        else:  # Conformer
            # Expected: Gender < Acoustic < Accent < Duration
            follows = (gender < acoustic) and (acoustic < accent) and (accent < duration)

        if follows:
            results[arch]["follows"] += 1
        else:
            results[arch]["exceptions"].append(model)

    return results


def compute_whisper_vs_nonwhisper(df):
    """Compare Whisper vs non-Whisper Transformers."""
    trans_df = df[df["Architecture"] == "Transformer"].copy()

    # Identify Whisper models
    trans_df["is_whisper"] = trans_df["Model"].apply(
        lambda m: "whisper" in m.lower()
    )

    results = {}

    for group in ["gender", "accent", "duration"]:
        col = f"{group}_mean_pos"
        if col not in trans_df.columns:
            continue

        whisper = trans_df[trans_df["is_whisper"]][col].dropna()
        non_whisper = trans_df[~trans_df["is_whisper"]][col].dropna()

        if len(whisper) < 2 or len(non_whisper) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(whisper, non_whisper)

        results[group] = {
            "whisper_mean": whisper.mean(),
            "whisper_std": whisper.std(),
            "whisper_n": len(whisper),
            "non_whisper_mean": non_whisper.mean(),
            "non_whisper_std": non_whisper.std(),
            "non_whisper_n": len(non_whisper),
            "t_stat": t_stat,
            "p_value": p_val,
            "df": len(whisper) + len(non_whisper) - 2
        }

    return results


def compute_ssl_comparison(df):
    """Compare SSL models across architectures."""
    results = {}

    # Identify SSL models
    df = df.copy()
    df["is_ssl"] = df["Model"].apply(
        lambda m: get_model_info(m).get("ssl", False)
    )

    ssl_df = df[df["is_ssl"]]

    for group in ["gender", "accent", "duration"]:
        col = f"{group}_mean_pos"
        if col not in ssl_df.columns:
            continue

        trans = ssl_df[ssl_df["Architecture"] == "Transformer"][col].dropna()
        conf = ssl_df[ssl_df["Architecture"] == "Conformer"][col].dropna()

        results[group] = {
            "ssl_transformer_mean": trans.mean() if len(trans) > 0 else np.nan,
            "ssl_transformer_n": len(trans),
            "ssl_conformer_mean": conf.mean() if len(conf) > 0 else np.nan,
            "ssl_conformer_n": len(conf),
            "delta": (conf.mean() - trans.mean()) if len(trans) > 0 and len(conf) > 0 else np.nan
        }

    return results


# =============================================================================
# Main Analysis
# =============================================================================

def run_full_analysis():
    """Run all analyses and return results."""
    root = get_root()
    print(f"📁 Data root: {root}")

    # Load data
    df_agg = load_aggregate_data(root)
    df_probing = load_probing_data(root)

    # Add architecture info if not present
    if "Architecture" not in df_agg.columns:
        df_agg["Architecture"] = df_agg["Model"].apply(
            lambda m: get_model_info(m)["arch"]
        )

    print(f"\n📊 Models: {len(df_agg)} total")
    print(f"   Transformers: {len(df_agg[df_agg['Architecture'] == 'Transformer'])}")
    print(f"   Conformers: {len(df_agg[df_agg['Architecture'] == 'Conformer'])}")

    # Run analyses
    print("\n🔬 Running analyses...")

    print("   1. Computing peak statistics...")
    peak_stats, df_agg = compute_peak_stats(df_agg)

    print("   2. Computing t-tests...")
    t_tests = compute_t_tests(df_agg, peak_stats)

    print("   3. Computing bootstrap confidence intervals...")
    bootstrap = compute_bootstrap_deltas(df_agg)

    print("   4. Computing hierarchy range...")
    hierarchy_range = compute_hierarchy_range(df_agg)

    print("   5. Training architectural classifier...")
    classifier = compute_architectural_classifier(df_agg)

    print("   6. Computing regression controlling for size...")
    regression = compute_regression_controlling_size(df_agg)

    print("   7. Computing cross-feature correlations...")
    correlations = compute_cross_correlations(df_agg)

    print("   8. Computing strength-width correlations...")
    strength_width = compute_strength_width_correlation(df_probing, df_agg)

    print("   9. Computing entropy statistics...")
    entropy = compute_entropy(df_probing)

    print("  10. Computing hierarchy consistency...")
    consistency = compute_hierarchy_consistency(df_agg)

    print("  11. Computing Whisper vs non-Whisper...")
    whisper_comp = compute_whisper_vs_nonwhisper(df_agg)

    print("  12. Computing SSL comparison...")
    ssl_comp = compute_ssl_comparison(df_agg)

    # Compile results
    results = {
        "peak_stats": peak_stats,
        "t_tests": t_tests,
        "bootstrap": bootstrap,
        "hierarchy_range": hierarchy_range,
        "classifier": classifier,
        "regression": regression,
        "correlations": correlations,
        "strength_width": strength_width,
        "entropy": entropy,
        "consistency": consistency,
        "whisper_comparison": whisper_comp,
        "ssl_comparison": ssl_comp
    }

    return results, df_agg


def format_results(results):
    """Format results for display and manuscript."""
    output = []
    output.append("=" * 80)
    output.append("FULL ANALYSIS RESULTS FOR MANUSCRIPT")
    output.append("=" * 80)

    # Section 4.1: Peak positions
    output.append("\n## Section 4.1: Divergent Processing Hierarchies\n")

    ps = results["peak_stats"]
    for arch in ["Transformer", "Conformer"]:
        output.append(f"\n### {arch} Hierarchy:")
        for group in ["acoustic", "gender", "accent", "phoneme", "duration"]:
            if group in ps and arch in ps[group]:
                s = ps[group][arch]
                output.append(f"  {group.capitalize()}: mean p = {s['mean']:.2f}, SD = {s['std']:.2f}, "
                            f"range [{s['min']:.2f}, {s['max']:.2f}]")

    # Hierarchy range
    hr = results["hierarchy_range"]
    output.append(f"\n### Hierarchy Range (Gender→Duration):")
    if "Transformer" in hr:
        output.append(f"  Transformer: Δ = {hr['Transformer']['mean_range']:.2f}")
    if "Conformer" in hr:
        output.append(f"  Conformer: Δ = {hr['Conformer']['mean_range']:.2f}")
    if "t_test" in hr:
        output.append(f"  t-test: t = {hr['t_test']['t_stat']:.2f}, p = {hr['t_test']['p_value']:.4f}")

    # T-tests
    output.append("\n### T-tests (Conformer - Transformer):")
    for group, tt in results["t_tests"].items():
        output.append(f"  {group.capitalize()}: Δp = {tt['delta']:+.2f}, t({tt['df']}) = {tt['t_stat']:.2f}, "
                     f"p = {tt['p_value']:.4f}")

    # Bootstrap CIs
    output.append("\n### Bootstrap 95% CIs for Δp:")
    for group, bs in results["bootstrap"].items():
        output.append(f"  {group.capitalize()}: Δp = {bs['delta_mean']:+.2f}, "
                     f"95% CI [{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]")

    # Hierarchy consistency
    output.append("\n### Hierarchy Consistency:")
    cons = results["consistency"]
    for arch in ["Transformer", "Conformer"]:
        c = cons[arch]
        pct = 100 * c["follows"] / c["total"] if c["total"] > 0 else 0
        output.append(f"  {arch}: {c['follows']}/{c['total']} ({pct:.0f}%)")
        if c["exceptions"]:
            output.append(f"    Exceptions: {', '.join(c['exceptions'][:3])}")

    # Section 4.2: Feature-specific
    output.append("\n\n## Section 4.2: Positional Dynamics\n")

    # Peak scores
    output.append("### Peak Scores by Feature:")
    for group in ["acoustic", "gender", "accent", "phoneme", "duration"]:
        if group in ps:
            for arch in ["Transformer", "Conformer"]:
                if arch in ps[group]:
                    s = ps[group][arch]
                    output.append(f"  {group.capitalize()} ({arch}): s = {s['score_mean']:.2f} ± {s['score_std']:.2f}")

    # Whisper comparison
    output.append("\n### Whisper vs Non-Whisper Transformers:")
    wc = results["whisper_comparison"]
    for group, stats in wc.items():
        output.append(f"  {group.capitalize()}:")
        output.append(f"    Whisper: mean p = {stats['whisper_mean']:.2f} (n={stats['whisper_n']})")
        output.append(f"    Non-Whisper: mean p = {stats['non_whisper_mean']:.2f} (n={stats['non_whisper_n']})")
        output.append(f"    t({stats['df']}) = {stats['t_stat']:.2f}, p = {stats['p_value']:.4f}")

    # Section 4.3: Strength and encoding
    output.append("\n\n## Section 4.3: Representational Strength\n")

    # Strength-width correlation
    output.append("### Strength-Width Correlation:")
    sw = results["strength_width"]
    for arch, stats in sw.items():
        output.append(f"  {arch}: r = {stats['r']:.2f}, p = {stats['p']:.6f}, N = {stats['n']}")

    # Entropy
    output.append("\n### Layer-wise Entropy:")
    ent = results["entropy"]
    for arch in ["Transformer", "Conformer"]:
        if arch in ent:
            output.append(f"  {arch}:")
            for group, stats in ent[arch].items():
                output.append(f"    {group}: H = {stats['median']:.2f}")

    # Section 4.4: Classifier
    output.append("\n\n## Section 4.4: Architectural Fingerprints\n")

    clf = results["classifier"]
    if clf:
        output.append(f"### Logistic Regression Classifier:")
        output.append(f"  AUC = {clf['auc']:.2f}, 95% CI [{clf['auc_ci_low']:.2f}, {clf['auc_ci_high']:.2f}]")
        output.append("  Coefficients:")
        for feat, coef in clf["coefficients"].items():
            output.append(f"    {feat}: β = {coef:.2f}")

    # Regression controlling for size
    output.append("\n### Regression Controlling for Model Size:")
    reg = results["regression"]
    for group, stats in reg.items():
        output.append(f"  {group.capitalize()}:")
        output.append(f"    β_arch = {stats['beta_arch']:.3f} (p = {stats['p_arch']:.4f})")
        output.append(f"    β_size = {stats['beta_size']:.3f} (p = {stats['p_size']:.4f})")

    # Cross-correlations
    output.append("\n### Cross-Feature Correlations (partial, controlling for params):")
    corr = results["correlations"]
    for arch, arch_corrs in corr.items():
        output.append(f"  {arch}:")
        for pair, stats in arch_corrs.items():
            output.append(f"    {pair}: ρ = {stats['partial_rho']:.2f}, p = {stats['partial_p']:.3f}, N = {stats['n']}")

    # SSL comparison
    output.append("\n\n## Section 4.6: SSL-only Comparison\n")
    ssl = results["ssl_comparison"]
    for group, stats in ssl.items():
        output.append(f"  {group.capitalize()}:")
        output.append(f"    SSL-Transformer: mean p = {stats['ssl_transformer_mean']:.2f} (n={stats['ssl_transformer_n']})")
        output.append(f"    SSL-Conformer: mean p = {stats['ssl_conformer_mean']:.2f} (n={stats['ssl_conformer_n']})")
        if not np.isnan(stats['delta']):
            output.append(f"    Δp = {stats['delta']:+.2f}")

    return "\n".join(output)


def save_results(results, output_path):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results, df = run_full_analysis()

    # Format and print
    formatted = format_results(results)
    print(formatted)

    # Save to file
    output_dir = Path(__file__).parent
    save_results(results, output_dir / "analysis_results.json")

    with open(output_dir / "analysis_results.txt", 'w') as f:
        f.write(formatted)
    print(f"✅ Text results saved to: {output_dir / 'analysis_results.txt'}")
