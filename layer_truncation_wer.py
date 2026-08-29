#!/usr/bin/env python3
"""Evaluate ASR WER under matched Transformer/Conformer layer truncation.

The two checkpoints are depth-matched Wav2Vec2-family CTC systems. Truncation
keeps the first k encoder blocks and the checkpoint's original final
normalization and CTC head. Models are evaluated sequentially to bound memory.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor


SEED = 20260829
DATASET_ID = "PranavBhalerao/l2-arctic-dataset-250"
DATASET_REVISION = "7e3c8665e26e55080fe9a4eb1601524600395af9"
L1_NAMES = ["Arabic", "Hindi", "Korean", "Mandarin", "Spanish", "Vietnamese"]
MODELS = {
    "Transformer": {
        "id": "facebook/wav2vec2-large-960h-lv60",
        "revision": "8e7d14742e8f98c6bbb24e5231406af321a8f9ce",
        "backbone": "wav2vec2",
        "encoder_layers": 24,
        "hidden_size": 1024,
        "pretraining": "Libri-Light; see pinned model card",
        "fine_tuning": "LibriSpeech-960h",
    },
    "Conformer": {
        "id": "facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
        "revision": "ca7f36f527f234b3cd4f05ecee30361f971e8e33",
        "backbone": "wav2vec2_conformer",
        "encoder_layers": 24,
        "hidden_size": 1024,
        "pretraining": "LibriSpeech-960h; see pinned model card",
        "fine_tuning": "LibriSpeech-960h",
    },
}
DEPTH_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
BOOTSTRAP_SAMPLES = 10_000


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def decode_audio(audio: dict, target_rate: int = 16_000) -> np.ndarray:
    if audio.get("bytes") is not None:
        samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path"):
        samples, sample_rate = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("Audio record contains neither bytes nor path")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        samples = resample_poly(
            samples,
            target_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return samples


def select_balanced_examples(per_l1: int) -> list[dict[str, object]]:
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
    )
    buckets: dict[int, list[dict[str, object]]] = {index: [] for index in range(6)}
    for row in dataset:
        label = int(row["l1_background"])
        if label not in buckets or len(buckets[label]) >= per_l1:
            continue
        speaker = str(row["speaker"])
        file_id = str(row["file_id"])
        buckets[label].append(
            {
                "sample_id": f"{speaker}/{file_id}",
                "speaker": speaker,
                "l1": L1_NAMES[label],
                "reference": normalize_text(str(row["text"])),
                "audio": decode_audio(row["audio"]),
            }
        )
        if all(len(values) == per_l1 for values in buckets.values()):
            break
    counts = {L1_NAMES[index]: len(values) for index, values in buckets.items()}
    if any(count != per_l1 for count in counts.values()):
        raise ValueError(f"Could not construct the requested balanced sample: {counts}")
    return [row for label in range(6) for row in buckets[label]]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    result = jiwer.process_words(reference, hypothesis)
    errors = result.substitutions + result.deletions + result.insertions
    reference_words = result.hits + result.substitutions + result.deletions
    return int(errors), int(reference_words)


def run_model(
    architecture: str,
    examples: list[dict[str, object]],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    specification = MODELS[architecture]
    processor = AutoProcessor.from_pretrained(
        specification["id"], revision=specification["revision"]
    )
    model = AutoModelForCTC.from_pretrained(
        specification["id"], revision=specification["revision"]
    ).to(device)
    model.eval()
    model.config.layerdrop = 0.0
    backbone = getattr(model, specification["backbone"])
    original_layers = backbone.encoder.layers
    total_layers = len(original_layers)
    if total_layers != 24:
        raise ValueError(f"Expected a 24-layer {architecture}, found {total_layers}")
    if int(model.config.hidden_size) != 1024:
        raise ValueError(
            f"Expected hidden size 1024 for {architecture}, found {model.config.hidden_size}"
        )

    predictions_path = output_dir / "layer_truncation_predictions.csv"
    if predictions_path.exists():
        completed = pd.read_csv(predictions_path)
    else:
        completed = pd.DataFrame()

    for fraction in DEPTH_FRACTIONS:
        layers_used = int(round(total_layers * fraction))
        expected_ids = {str(example["sample_id"]) for example in examples}
        if not completed.empty:
            existing = completed.loc[
                completed["Architecture"].eq(architecture)
                & completed["Layers_Used"].eq(layers_used)
            ]
            if set(existing["Sample_ID"].astype(str)) == expected_ids:
                print(f"Skipping completed {architecture} at {layers_used}/{total_layers}")
                continue

        backbone.encoder.layers = torch.nn.ModuleList(list(original_layers)[:layers_used])
        started = time.time()
        rows: list[dict[str, object]] = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = processor(
                [example["audio"] for example in batch],
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode():
                logits = model(**model_inputs).logits
            token_ids = logits.argmax(dim=-1).cpu()
            hypotheses = [normalize_text(text) for text in processor.batch_decode(token_ids)]
            for example, hypothesis in zip(batch, hypotheses, strict=True):
                errors, reference_words = error_counts(str(example["reference"]), hypothesis)
                rows.append(
                    {
                        "Architecture": architecture,
                        "Model_ID": specification["id"],
                        "Revision": specification["revision"],
                        "Layers_Used": layers_used,
                        "Total_Layers": total_layers,
                        "Depth_Fraction": layers_used / total_layers,
                        "Sample_ID": example["sample_id"],
                        "Speaker": example["speaker"],
                        "L1": example["l1"],
                        "Reference": example["reference"],
                        "Hypothesis": hypothesis,
                        "Word_Errors": errors,
                        "Reference_Words": reference_words,
                    }
                )
            print(
                f"{architecture} {layers_used}/{total_layers}: "
                f"{min(start + batch_size, len(examples))}/{len(examples)}",
                flush=True,
            )

        if not completed.empty:
            completed = completed.loc[
                ~(
                    completed["Architecture"].eq(architecture)
                    & completed["Layers_Used"].eq(layers_used)
                )
            ]
        completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
        completed.to_csv(predictions_path, index=False)
        print(
            f"Finished {architecture} {layers_used}/{total_layers} in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )

    backbone.encoder.layers = original_layers
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def corpus_wer(rows: pd.DataFrame) -> float:
    return float(rows["Word_Errors"].sum() / rows["Reference_Words"].sum())


def paired_bootstrap_degradation(
    truncated: pd.DataFrame,
    full: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[float, float]:
    paired = truncated[["Sample_ID", "Word_Errors", "Reference_Words"]].merge(
        full[["Sample_ID", "Word_Errors", "Reference_Words"]],
        on="Sample_ID",
        suffixes=("_truncated", "_full"),
        validate="one_to_one",
    )
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        truncated_wer = sample["Word_Errors_truncated"].sum() / sample["Reference_Words_truncated"].sum()
        full_wer = sample["Word_Errors_full"].sum() / sample["Reference_Words_full"].sum()
        estimates[index] = truncated_wer - full_wer
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def architecture_contrasts(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fraction in sorted(set(predictions["Depth_Fraction"])):
        if math.isclose(float(fraction), 1.0):
            continue
        by_architecture: dict[str, pd.DataFrame] = {}
        point_degradation: dict[str, float] = {}
        for architecture in MODELS:
            arch = predictions.loc[predictions["Architecture"].eq(architecture)]
            truncated = arch.loc[arch["Depth_Fraction"].eq(fraction)]
            full = arch.loc[arch["Depth_Fraction"].eq(1.0)]
            paired = truncated[["Sample_ID", "Word_Errors", "Reference_Words"]].merge(
                full[["Sample_ID", "Word_Errors", "Reference_Words"]],
                on="Sample_ID",
                suffixes=("_truncated", "_full"),
                validate="one_to_one",
            )
            by_architecture[architecture] = paired
            point_degradation[architecture] = float(
                summary.loc[
                    summary["Architecture"].eq(architecture)
                    & summary["Depth_Fraction"].eq(fraction),
                    "Absolute_Degradation",
                ].iloc[0]
            )

        paired = by_architecture["Conformer"].merge(
            by_architecture["Transformer"],
            on="Sample_ID",
            suffixes=("_conformer", "_transformer"),
            validate="one_to_one",
        )
        estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for index in range(BOOTSTRAP_SAMPLES):
            sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
            degradations = {}
            for architecture in ("conformer", "transformer"):
                truncated_wer = (
                    sample[f"Word_Errors_truncated_{architecture}"].sum()
                    / sample[f"Reference_Words_truncated_{architecture}"].sum()
                )
                full_wer = (
                    sample[f"Word_Errors_full_{architecture}"].sum()
                    / sample[f"Reference_Words_full_{architecture}"].sum()
                )
                degradations[architecture] = truncated_wer - full_wer
            estimates[index] = degradations["conformer"] - degradations["transformer"]
        low, high = np.percentile(estimates, [2.5, 97.5])
        difference = point_degradation["Conformer"] - point_degradation["Transformer"]
        rows.append(
            {
                "Depth_Fraction": float(fraction),
                "Conformer_Degradation": point_degradation["Conformer"],
                "Transformer_Degradation": point_degradation["Transformer"],
                "Difference_Conformer_minus_Transformer": difference,
                "Difference_CI_2.5": float(low),
                "Difference_CI_97.5": float(high),
                "Supports_Smaller_Conformer_Degradation": bool(high < 0),
            }
        )
    return pd.DataFrame(rows)


def summarize(output_dir: Path) -> None:
    predictions_path = output_dir / "layer_truncation_predictions.csv"
    predictions = pd.read_csv(predictions_path, dtype={"Sample_ID": str})
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for architecture in MODELS:
        architecture_rows = predictions.loc[predictions["Architecture"].eq(architecture)]
        full = architecture_rows.loc[architecture_rows["Depth_Fraction"].eq(1.0)]
        if full.empty:
            continue
        full_wer = corpus_wer(full)
        for fraction, depth_rows in architecture_rows.groupby("Depth_Fraction"):
            wer = corpus_wer(depth_rows)
            if math.isclose(fraction, 1.0):
                ci_low = ci_high = 0.0
            else:
                ci_low, ci_high = paired_bootstrap_degradation(depth_rows, full, rng)
            rows.append(
                {
                    "Architecture": architecture,
                    "Model_ID": depth_rows["Model_ID"].iloc[0],
                    "Layers_Used": int(depth_rows["Layers_Used"].iloc[0]),
                    "Total_Layers": int(depth_rows["Total_Layers"].iloc[0]),
                    "Depth_Fraction": float(fraction),
                    "N_Utterances": len(depth_rows),
                    "Reference_Words": int(depth_rows["Reference_Words"].sum()),
                    "WER": wer,
                    "Full_Depth_WER": full_wer,
                    "Absolute_Degradation": wer - full_wer,
                    "Degradation_CI_2.5": ci_low,
                    "Degradation_CI_97.5": ci_high,
                }
            )
    summary = pd.DataFrame(rows).sort_values(["Architecture", "Depth_Fraction"])
    summary.to_csv(output_dir / "layer_truncation_wer_summary.csv", index=False)
    if set(summary["Architecture"]) == set(MODELS):
        contrasts = architecture_contrasts(predictions, summary, rng)
        contrasts.to_csv(
            output_dir / "layer_truncation_architecture_contrast.csv", index=False
        )
        print("\nArchitecture contrast in absolute WER degradation")
        print(contrasts.to_string(index=False))
    manifest = {
        "seed": SEED,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "selection": "first N examples per L1 category in dataset order",
        "l1_categories": L1_NAMES,
        "models": MODELS,
        "depth_fractions": DEPTH_FRACTIONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "audio_sampling_rate_hz": 16_000,
        "n_utterances": int(
            predictions.loc[
                predictions["Architecture"].eq(next(iter(MODELS)))
                & predictions["Depth_Fraction"].eq(1.0),
                "Sample_ID",
            ].nunique()
        ),
        "l1_counts": {
            str(label): int(count)
            for label, count in predictions.loc[
                predictions["Architecture"].eq(next(iter(MODELS)))
                & predictions["Depth_Fraction"].eq(1.0)
            ]["L1"].value_counts().sort_index().items()
        },
        "reference_words": int(
            predictions.loc[
                predictions["Architecture"].eq(next(iter(MODELS)))
                & predictions["Depth_Fraction"].eq(1.0),
                "Reference_Words",
            ].sum()
        ),
        "text_normalization": "lowercase; ASCII letters, digits, apostrophes; collapse whitespace",
    }
    (output_dir / "layer_truncation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("camera_ready_results"))
    parser.add_argument("--per-l1", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture",
        choices=("Transformer", "Conformer", "both"),
        default="both",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = select_balanced_examples(args.per_l1)
    device = choose_device(args.device)
    print(f"Using {device} for {len(examples)} utterances")
    architectures = list(MODELS) if args.architecture == "both" else [args.architecture]
    for architecture in architectures:
        run_model(architecture, examples, args.output_dir, args.batch_size, device)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
