#!/usr/bin/env python3
"""Reproduce the promised camera-ready robustness analyses.

This script consumes the reviewed PCA-5 layer-wise probing table and aggregate
model table. It writes only aggregate, redistributable results; the raw input
may contain measurements derived from the private Cambridge corpus and must
not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEED = 20260829
BOOTSTRAP_SAMPLES = 10_000
CATEGORY_ORDER = ["Acoustic", "Gender", "Accent", "Phoneme", "Duration"]
GOLD_TRANSCRIPT_DATASETS = {"cmu-arctic-train", "l2-arctic-dataset-250"}
CAMBRIDGE_DATASET = "cam_assess"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_for(feature: str) -> str:
    value = feature.lower()
    if value.startswith(("f0", "f1", "f2", "f3", "intensity")):
        return "Acoustic"
    mapping = {
        "gender": "Gender",
        "l1_background": "Accent",
        "phoneme": "Phoneme",
        "duration": "Duration",
    }
    if value not in mapping:
        raise ValueError(f"Unrecognized feature: {feature}")
    return mapping[value]


def no_skill_baseline(feature: str) -> float:
    return {
        "gender": 0.5,
        "l1_background": 1.0 / 6.0,
        "phoneme": 1.0 / 39.0,
    }.get(feature.lower(), 0.0)


def aggregate_peak_from_reviewed_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    prefixes = {
        "Acoustic": ("f0_", "f1_", "f2_", "f3_", "f3_minus_f2_", "intensity_"),
        "Gender": ("gender_",),
        "Accent": ("l1_background_",),
        "Phoneme": ("phoneme_",),
        "Duration": ("duration_",),
    }
    rows: list[dict[str, object]] = []
    for _, model_row in aggregate.iterrows():
        for category, starts in prefixes.items():
            columns = [
                column
                for column in aggregate.columns
                if column.lower().startswith(starts)
                and column.lower().endswith("_peak_layer_percent")
            ]
            if not columns:
                raise ValueError(f"No reviewed aggregate columns for {category}")
            rows.append(
                {
                    "Model": model_row["Model"],
                    "Category": category,
                    "Reviewed_Peak": float(model_row[columns].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_curve_metrics(raw: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    required = {"Model", "Dataset", "Feature", "Layer", "PCA_Dim", "Test_Score"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Layer-wise input is missing columns: {sorted(missing)}")

    raw = raw.loc[raw["PCA_Dim"].eq(5)].copy()
    architectures = aggregate.set_index("Model")["Architecture"].to_dict()
    model_depths = raw.groupby("Model")["Layer"].max().astype(int).to_dict()
    if set(raw["Model"]) != set(architectures):
        raise ValueError("The layer-wise and aggregate model sets do not match")

    rows: list[dict[str, object]] = []
    grouped = raw.groupby(["Model", "Dataset", "Feature"], sort=True)
    for (model, dataset, feature), curve in grouped:
        curve = curve.sort_values("Layer")
        depth = model_depths[model]
        normalized_depth = curve["Layer"].to_numpy(dtype=float) / depth
        scores = curve["Test_Score"].to_numpy(dtype=float)
        weights = np.maximum(scores - no_skill_baseline(feature), 0.0)
        center = float(np.dot(normalized_depth, weights) / weights.sum()) if weights.sum() else np.nan
        peak_index = int(np.nanargmax(scores))
        rows.append(
            {
                "Model": model,
                "Architecture": architectures[model],
                "Dataset": dataset,
                "Feature": feature,
                "Category": category_for(feature),
                "Model_Depth": depth,
                "Peak": float(normalized_depth[peak_index]),
                "Center_of_Mass": center,
            }
        )

    metrics = pd.DataFrame(rows)
    model_category = (
        metrics.groupby(["Model", "Architecture", "Dataset", "Category"], as_index=False)[
            ["Peak", "Center_of_Mass"]
        ]
        .mean()
    )
    return model_category


def validate_reviewed_baseline(model_category: pd.DataFrame, aggregate: pd.DataFrame) -> float:
    reconstructed = (
        model_category.groupby(["Model", "Category"], as_index=False)["Peak"].mean()
    )
    reviewed = aggregate_peak_from_reviewed_table(aggregate)
    compared = reconstructed.merge(reviewed, on=["Model", "Category"], validate="one_to_one")
    max_error = float((compared["Peak"] - compared["Reviewed_Peak"]).abs().max())
    if max_error > 1e-12:
        raise ValueError(f"Reconstructed peak positions differ from reviewed baseline: {max_error}")
    return max_error


def bootstrap_difference(
    conformer: np.ndarray,
    transformer: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        c_sample = rng.choice(conformer, size=len(conformer), replace=True)
        t_sample = rng.choice(transformer, size=len(transformer), replace=True)
        differences[index] = c_sample.mean() - t_sample.mean()
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def summarize(
    model_category: pd.DataFrame,
    analysis: str,
    metrics: tuple[str, ...],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        category_rows = model_category.loc[model_category["Category"].eq(category)]
        if category_rows.empty:
            continue
        for metric in metrics:
            conformer = category_rows.loc[
                category_rows["Architecture"].eq("Conformer"), metric
            ].dropna().to_numpy(dtype=float)
            transformer = category_rows.loc[
                category_rows["Architecture"].eq("Transformer"), metric
            ].dropna().to_numpy(dtype=float)
            if min(len(conformer), len(transformer)) < 2:
                raise ValueError(f"Insufficient data for {analysis}/{category}/{metric}")
            test = stats.ttest_ind(conformer, transformer, equal_var=False)
            ci_low, ci_high = bootstrap_difference(conformer, transformer, rng)
            rows.append(
                {
                    "Analysis": analysis,
                    "Category": category,
                    "Metric": metric,
                    "Conformer_Mean": float(conformer.mean()),
                    "Transformer_Mean": float(transformer.mean()),
                    "Difference_C_minus_T": float(conformer.mean() - transformer.mean()),
                    "CI_2.5": ci_low,
                    "CI_97.5": ci_high,
                    "Welch_t": float(test.statistic),
                    "Welch_p": float(test.pvalue),
                    "N_Conformer": len(conformer),
                    "N_Transformer": len(transformer),
                }
            )
    return pd.DataFrame(rows)


def model_level(model_category: pd.DataFrame) -> pd.DataFrame:
    return (
        model_category.groupby(["Model", "Architecture", "Category"], as_index=False)[
            ["Peak", "Center_of_Mass"]
        ]
        .mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    layer_path = args.data_dir / "ALL_PROBING_WITH_PHONEMES.csv"
    aggregate_path = args.data_dir / "ALLDATA_aggregate_metrics (with phonemes).csv"
    raw = pd.read_csv(layer_path)
    aggregate = pd.read_csv(aggregate_path)
    curve_metrics = build_curve_metrics(raw, aggregate)
    baseline_error = validate_reviewed_baseline(curve_metrics, aggregate)
    rng = np.random.default_rng(SEED)

    all_models = model_level(curve_metrics)
    center_summary = summarize(
        all_models,
        analysis="All corpora",
        metrics=("Peak", "Center_of_Mass"),
        rng=rng,
    )

    gold = curve_metrics.loc[
        curve_metrics["Dataset"].isin(GOLD_TRANSCRIPT_DATASETS)
        & curve_metrics["Category"].eq("Phoneme")
    ]
    gold_summary = summarize(
        model_level(gold),
        analysis="Gold transcripts only",
        metrics=("Peak", "Center_of_Mass"),
        rng=rng,
    )

    no_cambridge = curve_metrics.loc[curve_metrics["Dataset"].ne(CAMBRIDGE_DATASET)]
    no_cambridge_summary = summarize(
        model_level(no_cambridge),
        analysis="Leave Cambridge out",
        metrics=("Peak", "Center_of_Mass"),
        rng=rng,
    )

    non_whisper = all_models.loc[
        all_models["Architecture"].eq("Conformer")
        | ~all_models["Model"].str.startswith("whisper-")
    ]
    non_whisper_summary = summarize(
        non_whisper,
        analysis="Conformer vs non-Whisper Transformer",
        metrics=("Center_of_Mass",),
        rng=rng,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_metrics.to_csv(args.output_dir / "corpus_category_depth_metrics.csv", index=False)
    all_models.to_csv(args.output_dir / "model_category_depth_metrics.csv", index=False)
    center_summary.to_csv(args.output_dir / "center_of_mass_summary.csv", index=False)
    gold_summary.to_csv(args.output_dir / "gold_transcript_only_summary.csv", index=False)
    no_cambridge_summary.to_csv(args.output_dir / "leave_cambridge_out_summary.csv", index=False)
    non_whisper_summary.to_csv(args.output_dir / "center_of_mass_non_whisper_summary.csv", index=False)

    peak_rows = center_summary.loc[center_summary["Metric"].eq("Peak")].set_index("Category")
    center_rows = center_summary.loc[
        center_summary["Metric"].eq("Center_of_Mass")
    ].set_index("Category")
    agreement = {}
    for category in CATEGORY_ORDER:
        peak_difference = float(peak_rows.loc[category, "Difference_C_minus_T"])
        center_difference = float(center_rows.loc[category, "Difference_C_minus_T"])
        agreement[category] = {
            "same_direction": bool(np.sign(peak_difference) == np.sign(center_difference)),
            "peak_difference": peak_difference,
            "center_difference": center_difference,
            "center_ci_excludes_zero": bool(
                center_rows.loc[category, "CI_2.5"] > 0
                or center_rows.loc[category, "CI_97.5"] < 0
            ),
        }

    summary = {
        "seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "pca_dimension": 5,
        "input_rows": int(len(raw)),
        "models": int(raw["Model"].nunique()),
        "datasets": sorted(raw["Dataset"].unique().tolist()),
        "reviewed_baseline_max_abs_error": baseline_error,
        "input_sha256": {
            layer_path.name: sha256(layer_path),
            aggregate_path.name: sha256(aggregate_path),
        },
        "center_of_mass_baselines": {
            "R2 probes": 0.0,
            "Gender": 0.5,
            "Accent": 1.0 / 6.0,
            "Phoneme": 1.0 / 39.0,
        },
        "peak_center_direction_agreement": agreement,
    }
    (args.output_dir / "robustness_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(center_summary.to_string(index=False))
    print("\nGold-transcript-only phoneme analysis")
    print(gold_summary.to_string(index=False))
    print("\nLeave-Cambridge-out analysis")
    print(no_cambridge_summary.to_string(index=False))
    print(f"\nReviewed peak reconstruction max error: {baseline_error:.3g}")


if __name__ == "__main__":
    main()

