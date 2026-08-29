# Categorize Early, Integrate Late? Representational Profiles Across Sampled ASR Model Families

Supplemental code, analysis artifacts, and figure-generation resources for the forthcoming EMNLP 2026 paper [“Categorize Early, Integrate Late? Representational Profiles Across Sampled ASR Model Families”](https://arxiv.org/abs/2601.06972).

This repository contains supplemental materials only; manuscript source and compiled manuscript files are not included.

## Abstract

Across 24 pretrained encoders, the sampled Conformer group shows earlier pooled phoneme, gender, and accent peaks and a later duration peak than the sampled Transformer group. Performance-weighted center of mass, gold-transcript-only, and leave-Cambridge-out replications retain the principal pooled directions. However, non-Whisper and family-random-intercept controls do not identify a significant architecture effect, so the comparison is an association within this model suite rather than an architecture-only result. A matched Wav2Vec2/Conformer truncation test also contradicts the proposed early-exit prediction: both unmodified systems collapse below full depth, with larger Conformer degradation.

## Main results

- The study compares 17 Transformer and 7 Conformer encoders.
- The pooled Conformer group peaks earlier for gender, accent, and phoneme accessibility and later for duration.
- Center of mass agrees with the pooled peak direction for four categories; Acoustic is null under both metrics and changes sign.
- Gold-transcript-only phoneme results and leave-Cambridge-out peak results retain the relevant pooled directions.
- No non-Whisper peak contrast or family-random-intercept architecture coefficient is significant at 0.05.
- The matched 24-layer truncation experiment does not support the low-latency prediction; both checkpoints fail below full depth, and the Conformer degrades more.

## Repository structure

```text
.
├── Master_Analysis.ipynb        # Primary analysis notebook
├── analysis_toolkit.py          # Shared model, representation, and probing utilities
├── full_analysis.py             # Statistical analysis pipeline
├── analysis_results.json        # Machine-readable reported results
├── analysis_results.txt         # Human-readable reported results
├── appendix_data.json           # Appendix statistics and tables
├── generate_manuscript_figures.py
├── camera_ready_robustness.py  # Center-of-mass and corpus-sensitivity analyses
├── layer_truncation_wer.py     # Matched Transformer/Conformer WER experiment
├── camera_ready_results/       # Aggregate outputs and pinned manifests
├── requirements-camera-ready.txt
├── figs/                        # Figure-generation scripts
├── inference/                   # Model-specific representation extraction
└── probing/                     # Linear-probing scripts
```

## Reproducing the analyses

`Master_Analysis.ipynb` is the main entry point. It coordinates model loading, representation extraction, linear probing, statistical analysis, and figure generation through `analysis_toolkit.py` and the supporting scripts.

The experiments require Python, PyTorch, Hugging Face model dependencies, and access to the relevant speech corpora. Public datasets should be obtained under their original terms. Cambridge Assessment data used in the paper are private and are not redistributed by this repository.

The checked-in `analysis_results.json`, `analysis_results.txt`, and `appendix_data.json` contain the aggregate outputs reported in the paper. Model-level and redistributable outputs are included where their licenses permit redistribution.

The camera-ready robustness analyses are documented in [`camera_ready_results/README.md`](camera_ready_results/README.md). Raw probing tables and utterance-level truncation predictions are excluded: the former contain Cambridge Assessment-derived records, and the latter are unnecessary to reproduce the aggregate tables. Exact model and dataset revisions, seeds, input hashes, and bootstrap settings are pinned in the manifests.

## Citation

```bibtex
@inproceedings{roll2026categorize,
  title     = {Categorize Early, Integrate Late? Representational Profiles Across Sampled ASR Model Families},
  author    = {Roll, Nathan and Bhalerao, Pranav and Bartelds, Martijn and
               Pawar, Arjun and Tatsumi, Yuka and Ògúnrẹ̀mí, Tolúlọpẹ́ and
               Shani, Chen and Graham, Calbert and Sumner, Meghan and Jurafsky, Dan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {Forthcoming}
}
```
