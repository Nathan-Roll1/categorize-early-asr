# Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition

Supplemental code, analysis artifacts, and figure-generation resources for the forthcoming EMNLP 2026 paper [“Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition”](https://arxiv.org/abs/2601.06972).

This repository contains supplemental materials only; manuscript source and compiled manuscript files are not included.

## Abstract

In speech language modeling, two architectures dominate the frontier: the Transformer and the Conformer. However, it remains unknown whether their comparable performance stems from convergent processing strategies or distinct architectural inductive biases. We introduce *Architectural Fingerprinting*, a probing framework that isolates the effect of architecture on representation, and apply it to a controlled suite of 24 pre-trained encoders (39M–3.3B parameters). Our analysis reveals divergent hierarchies: Conformers implement a “Categorize Early” strategy, resolving phoneme categories 29% earlier in depth and speaker gender within the first 16% of network depth (vs. 28% in Transformers). In contrast, Transformers “Integrate Late,” deferring phoneme, accent, and duration encoding to deep layers (49–57%). These fingerprints motivate testable hypotheses: Conformers’ front-loaded categorization may benefit low-latency streaming, while Transformers’ deep integration may favor tasks requiring rich context and cross-utterance normalization.

## Appendix robustness findings

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
  title     = {Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition},
  author    = {Roll, Nathan and Bhalerao, Pranav and Bartelds, Martijn and
               Pawar, Arjun and Tatsumi, Yuka and Ògúnrẹ̀mí, Tolúlọpẹ́ and
               Shani, Chen and Graham, Calbert and Sumner, Meghan and Jurafsky, Dan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {Forthcoming}
}
```
