#!/usr/bin/env python3
"""
Supplementary Analysis for Manuscript Revisions
1. Sensitivity analysis excluding granite-speech outlier
2. Paired comparison of multilingual vs .en Whisper models
"""

import numpy as np
from scipy import stats

# ============================================================================
# Data from Table in Appendix
# ============================================================================

# Conformer models: [Acoustic, Gender, Accent, Phoneme, Duration]
conformers = {
    'canary-1b': [0.153, 0.139, 0.347, 0.215, 0.611],
    'canary-1b-flash': [0.146, 0.089, 0.260, 0.130, 0.849],
    'canary-qwen-2.5b': [0.151, 0.099, 0.255, 0.312, 0.464],
    'granite-speech-3.3-2b': [0.466, 0.323, 0.375, 0.229, 0.812],  # OUTLIER
    'parakeet-tdt-0.6b-v2': [0.194, 0.167, 0.340, 0.340, 0.611],
    'speechbrain-loq': [0.206, 0.083, 0.537, 0.157, 0.759],
    'w2v2-conformer': [0.192, 0.194, 0.319, 0.056, 0.764],  # SSL
}

# Transformer models
transformers = {
    'Phi-4-multimodal': [0.332, 0.281, 0.266, 0.896, 0.109],
    'hubert-large': [0.140, 0.181, 0.278, 0.000, 0.681],
    'hubert-xlarge': [0.091, 0.073, 0.271, 0.156, 0.538],
    'wav2vec2-large': [0.137, 0.090, 0.278, 0.014, 0.424],
    'wavlm-large': [0.186, 0.118, 0.618, 0.500, 0.667],
    'whisper-base': [0.306, 0.333, 0.722, 0.472, 0.472],
    'whisper-base.en': [0.258, 0.278, 0.667, 0.778, 0.472],
    'whisper-large': [0.217, 0.297, 0.698, 0.245, 0.552],
    'whisper-large-v2': [0.202, 0.328, 0.667, 0.797, 0.990],
    'whisper-large-v3': [0.202, 0.406, 0.708, 0.604, 1.000],
    'whisper-large-v3-turbo': [0.208, 0.396, 0.651, 0.495, 0.380],
    'whisper-medium': [0.172, 0.292, 0.618, 0.417, 0.465],
    'whisper-medium.en': [0.238, 0.403, 0.750, 0.611, 0.493],
    'whisper-small': [0.225, 0.306, 0.639, 0.542, 0.403],
    'whisper-small.en': [0.214, 0.306, 0.653, 0.625, 0.486],
    'whisper-tiny': [0.208, 0.375, 0.667, 0.667, 0.458],
    'whisper-tiny.en': [0.262, 0.333, 0.583, 0.542, 0.375],
}

features = ['Acoustic', 'Gender', 'Accent', 'Phoneme', 'Duration']

# ============================================================================
# 1. SENSITIVITY ANALYSIS: Excluding granite-speech
# ============================================================================

print("=" * 80)
print("SENSITIVITY ANALYSIS: Excluding granite-speech-3.3-2b")
print("=" * 80)

# With all Conformers
conf_all = np.array(list(conformers.values()))
trans_all = np.array(list(transformers.values()))

# Without granite-speech
conf_no_granite = np.array([v for k, v in conformers.items() if k != 'granite-speech-3.3-2b'])

print(f"\nConformer sample size: N={len(conf_all)} -> N={len(conf_no_granite)} (without granite)")
print(f"Transformer sample size: N={len(trans_all)}")

print("\n### Comparison: With vs Without granite-speech ###\n")
print(f"{'Feature':<12} | {'With granite':<25} | {'Without granite':<25} | {'Change':<10}")
print("-" * 80)

for i, feat in enumerate(features):
    # With all
    conf_mean_all = np.mean(conf_all[:, i])
    trans_mean = np.mean(trans_all[:, i])
    delta_all = conf_mean_all - trans_mean
    t_all, p_all = stats.ttest_ind(conf_all[:, i], trans_all[:, i])

    # Without granite
    conf_mean_no = np.mean(conf_no_granite[:, i])
    delta_no = conf_mean_no - trans_mean
    t_no, p_no = stats.ttest_ind(conf_no_granite[:, i], trans_all[:, i])

    print(f"{feat:<12} | Δp={delta_all:+.3f}, p={p_all:.4f} | Δp={delta_no:+.3f}, p={p_no:.4f} | {delta_no - delta_all:+.3f}")

print("\n### Conformer means comparison ###\n")
print(f"{'Feature':<12} | {'With granite':<15} | {'Without granite':<15} | {'Δ':<10}")
print("-" * 60)
for i, feat in enumerate(features):
    mean_all = np.mean(conf_all[:, i])
    mean_no = np.mean(conf_no_granite[:, i])
    print(f"{feat:<12} | {mean_all:.3f}            | {mean_no:.3f}            | {mean_no - mean_all:+.3f}")

# ============================================================================
# 2. MULTILINGUAL EFFECT: Paired Whisper comparison
# ============================================================================

print("\n" + "=" * 80)
print("MULTILINGUAL EFFECT: Paired comparison of .en vs multilingual Whisper")
print("=" * 80)

# Paired models (multilingual, .en)
paired_whisper = [
    ('whisper-tiny', 'whisper-tiny.en'),
    ('whisper-base', 'whisper-base.en'),
    ('whisper-small', 'whisper-small.en'),
    ('whisper-medium', 'whisper-medium.en'),
]

print("\n### Paired differences (Multilingual - English-only) ###\n")
print(f"{'Model Pair':<30} | {'Gender':<12} | {'Accent':<12} | {'Phoneme':<12} | {'Duration':<12}")
print("-" * 90)

diffs = {feat: [] for feat in features}

for multi, en in paired_whisper:
    multi_vals = transformers[multi]
    en_vals = transformers[en]

    gender_diff = multi_vals[1] - en_vals[1]
    accent_diff = multi_vals[2] - en_vals[2]
    phoneme_diff = multi_vals[3] - en_vals[3]
    duration_diff = multi_vals[4] - en_vals[4]

    diffs['Gender'].append(gender_diff)
    diffs['Accent'].append(accent_diff)
    diffs['Phoneme'].append(phoneme_diff)
    diffs['Duration'].append(duration_diff)

    print(f"{multi:<30} | {gender_diff:+.3f}       | {accent_diff:+.3f}       | {phoneme_diff:+.3f}       | {duration_diff:+.3f}")

print("-" * 90)

# Summary statistics with paired t-test
print("\n### Summary: Mean difference (Multilingual - English-only) ###\n")
print(f"{'Feature':<12} | {'Mean Δ':<12} | {'95% CI':<20} | {'t-stat':<10} | {'p-value':<10}")
print("-" * 70)

for feat in ['Gender', 'Accent', 'Phoneme', 'Duration']:
    d = np.array(diffs[feat])
    mean_d = np.mean(d)
    sem = stats.sem(d)
    ci = stats.t.interval(0.95, len(d)-1, loc=mean_d, scale=sem)
    t_stat, p_val = stats.ttest_1samp(d, 0)

    print(f"{feat:<12} | {mean_d:+.3f}       | [{ci[0]:+.3f}, {ci[1]:+.3f}]    | {t_stat:+.2f}      | {p_val:.4f}")

print("\n### Interpretation ###")
print("Positive Δ = Multilingual peaks LATER than English-only")
print("Negative Δ = Multilingual peaks EARLIER than English-only")

# ============================================================================
# 3. Additional: SSL-only Conformer analysis
# ============================================================================

print("\n" + "=" * 80)
print("ADDITIONAL: Excluding both granite-speech AND w2v2-conformer (SSL)")
print("=" * 80)

# Core supervised Conformers only
conf_supervised = np.array([v for k, v in conformers.items()
                            if k not in ['granite-speech-3.3-2b', 'w2v2-conformer']])

print(f"\nSupervised Conformers only: N={len(conf_supervised)}")
print("(Excluding granite-speech outlier and w2v2-conformer SSL model)")

print("\n### T-tests: Supervised Conformers vs All Transformers ###\n")
print(f"{'Feature':<12} | {'Δp':<10} | {'t-stat':<10} | {'p-value':<10} | {'Survives Bonf.':<15}")
print("-" * 60)

for i, feat in enumerate(features):
    conf_mean = np.mean(conf_supervised[:, i])
    trans_mean = np.mean(trans_all[:, i])
    delta = conf_mean - trans_mean
    t_stat, p_val = stats.ttest_ind(conf_supervised[:, i], trans_all[:, i])
    survives = "Yes" if p_val < 0.01 else "No"

    print(f"{feat:<12} | {delta:+.3f}     | {t_stat:+.2f}      | {p_val:.4f}     | {survives}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
