# Camera-ready robustness results

These files implement the four analyses promised during review. Raw probing inputs are intentionally excluded because Cambridge Assessment-derived records cannot be redistributed.

## Probing robustness analyses

Run from the repository root:

```bash
python camera_ready_robustness.py \
  --data-dir /path/to/private/probing-inputs \
  --output-dir camera_ready_results
```

The script verifies that reconstructed PCA-5 peak positions match the reviewed aggregate table before producing performance-weighted center-of-mass, gold-transcript-only, leave-Cambridge-out, and non-Whisper summaries. Input hashes, baselines, the fixed seed, and the reconstruction error are recorded in `robustness_manifest.json`.

## Layer-truncation WER

The reported run used Python 3.11.15 on Apple MPS:

```bash
python layer_truncation_wer.py \
  --output-dir camera_ready_results \
  --per-l1 40 \
  --batch-size 8 \
  --architecture both \
  --device mps
```

The evaluation pins the dataset and model revisions, resamples audio to 16 kHz, and uses 10,000 paired utterance bootstrap samples. It retains each checkpoint's original CTC head and final encoder normalization while keeping only the first 6, 12, 18, or 24 of 24 encoder blocks.

The low-latency prediction was not supported: both checkpoints collapsed below full depth, and Conformer degradation was larger at every truncated depth. WER values above 1 are possible because insertions count as errors.

## Main results

- Center of mass agrees with peak direction for Gender, Accent, Phoneme, and Duration; Acoustic is near zero under both metrics and changes sign.
- Gold-transcript-only phoneme analysis retains earlier pooled Conformer depth.
- Leaving Cambridge out retains the pooled direction for all five peak-position comparisons.
- The matched truncation test does not support converting early probe accessibility into an early-exit deployment claim.
