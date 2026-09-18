#!/usr/bin/env python3
"""Generate Head-to-Head Comparative Report: Muon vs AdamW Best Downstream Checkpoints."""
import os, os as _os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in [
    _REPO_ROOT,
    _os.path.join(_REPO_ROOT, "benchmarks"),
    _os.path.join(_REPO_ROOT, "benchmarks", "korean_pos_suite", "training"),
    _os.path.join(_REPO_ROOT, "benchmarks", "korean_pos_suite", "evaluations"),
    _os.path.join(_REPO_ROOT, "benchmarks", "korean_pos_suite", "reporting"),
    _os.path.join(_REPO_ROOT, "data", "korean_pos_mc"),
    _os.path.join(_REPO_ROOT, "huggingface_model", "gemma"),
    _os.path.join(_REPO_ROOT, "data", "template", "utils", "korean"),
]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load Muon data
with open(os.path.join(REPO_ROOT, "results_muon_20ep_all.json")) as f:
    muon_all = json.load(f)
with open(os.path.join(REPO_ROOT, "results_charhead_without_lmhead24_muon_all_metrics.json")) as f:
    muon_no_ch = json.load(f)

# From generate_full_reports.py:
# Muon models definition
MUON_MODELS = [
    ("baseline_character", "best_ckpt", "Baseline Character Tokenizer (V=4,924)"),
    ("baseline_char_bpe", "best_ckpt", "Baseline Byte-Fallback BPE (V=4,924)"),
    ("baseline_pure_byte", "20_epochs", "Baseline Pure Byte Tokenizer (V=256)"),
    ("three_hot_ind_char_byte", "20_epochs", "Three-Hot Independent Char/Byte (Song 2018)"),
    ("three_hot_cond_char_byte", "best_ckpt", "Three-Hot Conditional Char/Byte (Cognetta 2023)"),
    ("mc_full_pos_weighted_byte", "5_epochs", "Hangul Factorizer + Char Stream (Full POS, Weighted)"),
    ("mc_full_pos_unweighted_byte", "15_epochs", "Hangul Factorizer + Char Stream (Full POS, Unweighted)"),
    ("mc_coarse_pos_weighted_byte", "5_epochs", "Hangul Factorizer + Char Stream (Coarse POS, Weighted)"),
    ("mc_coarse_pos_unweighted_byte", "5_epochs", "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)"),
    ("nochar_full_pos_new_weighted", "best_ckpt", "No-Char Hangul Factorizer (Full POS, Newly Weighted C06)"),
    ("nochar_full_pos_unweighted", "best_ckpt", "No-Char Hangul Factorizer (Full POS, Unweighted)"),
    ("nochar_coarse_pos_new_weighted", "15_epochs", "No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)"),
    ("nochar_coarse_pos_unweighted", "20_epochs", "No-Char Hangul Factorizer (Coarse POS, Unweighted)"),
    ("nochar_full_pos_charhead_h11", "20_epochs", "Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)"),
    ("nochar_full_pos_charhead_unweighted", "15_epochs", "Output-Only Char Head Factorizer (Full POS, Unweighted)"),
    ("nochar_coarse_pos_charhead_h11", "20_epochs", "Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)"),
    ("nochar_coarse_pos_charhead_unweighted", "15_epochs", "Output-Only Char Head Factorizer (Coarse POS, Unweighted)"),
]

# AdamW models definition & mapping
ADAMW_MODELS = [
    ("baseline_character", "5_epoch", "Baseline Character Tokenizer (V=4,924)"),
    ("baseline_char_bpe", "best_ckpt", "Baseline Byte-Fallback BPE (V=4,924)"),
    ("baseline_pure_byte", "best_ckpt", "Baseline Pure Byte Tokenizer (V=256)"),
    ("three_hot_ind_char_byte", "best_ckpt", "Three-Hot Independent Char/Byte (Song 2018)"),
    ("three_hot_cond_char_byte", "best_ckpt", "Three-Hot Conditional Char/Byte (Cognetta 2023)"),
    ("full_pos_weighted_byte", "best_ckpt", "Hangul Factorizer + Char Stream (Full POS, Weighted)"),
    ("full_pos_unweighted_byte", "best_ckpt", "Hangul Factorizer + Char Stream (Full POS, Unweighted)"),
    ("coarse_pos_weighted_byte", "15_epochs", "Hangul Factorizer + Char Stream (Coarse POS, Weighted)"),
    ("coarse_pos_unweighted_byte", "20_epochs", "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)"),
    ("nochar_full_pos_new_weighted", "15_epochs", "No-Char Hangul Factorizer (Full POS, Newly Weighted C06)"),
    ("nochar_full_pos_unweighted", "15_epochs", "No-Char Hangul Factorizer (Full POS, Unweighted)"),
    ("nochar_coarse_pos_new_weighted", "best_ckpt", "No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)"),
    ("nochar_coarse_pos_unweighted", "15_epochs", "No-Char Hangul Factorizer (Coarse POS, Unweighted)"),
    ("nochar_full_pos_charhead_h11", "20_epochs", "Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)"),
    ("nochar_full_pos_charhead_unweighted", "best_ckpt", "Output-Only Char Head Factorizer (Full POS, Unweighted)"),
    ("nochar_coarse_pos_charhead_h11", "3_epoch", "Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)"),
    ("nochar_coarse_pos_charhead_unweighted", "best_ckpt", "Output-Only Char Head Factorizer (Coarse POS, Unweighted)"),
]

# AdamW master metric dict
adamw_metrics_map = {
    "baseline_character": {"iter": "18,165", "vl": 1.5703, "bpj": 1.8232, "chrf": 6.03, "bleu": 0.56, "ner": 8.01, "dp": 10.83, "hs_raw": 28.0, "hs_len": 21.0, "km": 22.0, "kob": 41.67, "lat": 152.8, "t_loss": 2.4096, "t_ppl": 11.13, "jamo_surg": 37.50, "choseong": 43.75, "syl_cnt": 43.75, "d1": 36.90, "d10": 10.03, "oov": 78.58},
    "baseline_char_bpe": {"iter": "71,000", "vl": 1.4768, "bpj": 1.6665, "chrf": 5.89, "bleu": 0.23, "ner": 8.35, "dp": 11.46, "hs_raw": 28.0, "hs_len": 20.0, "km": 21.0, "kob": 44.44, "lat": 241.1, "t_loss": 2.2316, "t_ppl": 9.31, "jamo_surg": 50.00, "choseong": 37.50, "syl_cnt": 45.83, "d1": 38.50, "d10": 12.10, "oov": 172.10},
    "baseline_pure_byte": {"iter": "106,000", "vl": 1.0072, "bpj": 1.3713, "chrf": 3.34, "bleu": 0.05, "ner": 2.93, "dp": 2.99, "hs_raw": 29.0, "hs_len": 24.0, "km": 18.0, "kob": 38.89, "lat": 196.9, "t_loss": 1.0343, "t_ppl": 2.81, "jamo_surg": 25.00, "choseong": 43.75, "syl_cnt": 47.92, "d1": 8.60, "d10": 5.80, "oov": 8.46},
    "three_hot_ind_char_byte": {"iter": "72,000", "vl": 0.5595, "bpj": 0.8144, "chrf": 2.68, "bleu": 0.12, "ner": 2.05, "dp": 2.45, "hs_raw": 27.0, "hs_len": 25.0, "km": 16.0, "kob": 30.56, "lat": 335.0, "t_loss": 1.0720, "t_ppl": 2.92, "jamo_surg": 25.00, "choseong": 31.25, "syl_cnt": 35.42, "d1": 42.10, "d10": 18.20, "oov": 5.12},
    "three_hot_cond_char_byte": {"iter": "65,000", "vl": 0.4575, "bpj": 0.5941, "chrf": 5.03, "bleu": 0.35, "ner": 1.31, "dp": 1.39, "hs_raw": 25.0, "hs_len": 17.0, "km": 19.0, "kob": 36.11, "lat": 338.0, "t_loss": 0.7890, "t_ppl": 2.20, "jamo_surg": 37.50, "choseong": 37.50, "syl_cnt": 33.33, "d1": 39.80, "d10": 15.60, "oov": 4.35},
    "full_pos_weighted_byte": {"iter": "72,000", "vl": 0.5790, "bpj": 0.4875, "chrf": 3.75, "bleu": 0.03, "ner": 2.59, "dp": 2.66, "hs_raw": 31.0, "hs_len": 27.0, "km": 22.0, "kob": 41.67, "lat": 2135.9, "t_loss": 0.6529, "t_ppl": 1.92, "jamo_surg": 37.50, "choseong": 43.75, "syl_cnt": 43.75, "d1": 45.60, "d10": 16.50, "oov": 7.91},
    "full_pos_unweighted_byte": {"iter": "65,000", "vl": 0.2639, "bpj": 0.4761, "chrf": 3.34, "bleu": 0.04, "ner": 2.59, "dp": 2.66, "hs_raw": 32.0, "hs_len": 24.0, "km": 22.0, "kob": 50.00, "lat": 2124.8, "t_loss": 0.6377, "t_ppl": 1.89, "jamo_surg": 50.00, "choseong": 50.00, "syl_cnt": 50.00, "d1": 45.60, "d10": 16.50, "oov": 8.44},
    "coarse_pos_weighted_byte": {"iter": "54,496", "vl": 0.5828, "bpj": 0.4986, "chrf": 3.15, "bleu": 0.04, "ner": 2.54, "dp": 2.64, "hs_raw": 35.0, "hs_len": 31.0, "km": 22.0, "kob": 41.67, "lat": 381.4, "t_loss": 0.6670, "t_ppl": 1.95, "jamo_surg": 37.50, "choseong": 43.75, "syl_cnt": 43.75, "d1": 47.00, "d10": 16.70, "oov": 8.36},
    "coarse_pos_unweighted_byte": {"iter": "72,661", "vl": 0.2664, "bpj": 0.4790, "chrf": 3.04, "bleu": 0.04, "ner": 2.55, "dp": 2.64, "hs_raw": 31.0, "hs_len": 30.0, "km": 23.0, "kob": 44.44, "lat": 376.0, "t_loss": 0.6410, "t_ppl": 1.90, "jamo_surg": 37.50, "choseong": 50.00, "syl_cnt": 45.83, "d1": 46.50, "d10": 16.65, "oov": 7.93},
    "nochar_full_pos_new_weighted": {"iter": "33,931", "vl": 0.8112, "bpj": 0.5019, "chrf": 2.83, "bleu": 0.00, "ner": 1.69, "dp": 1.46, "hs_raw": 29.0, "hs_len": 22.0, "km": 23.0, "kob": 36.11, "lat": 368.6, "t_loss": 0.6185, "t_ppl": 1.86, "jamo_surg": 37.50, "choseong": 37.50, "syl_cnt": 33.33, "d1": 81.00, "d10": 16.55, "oov": 95.38},
    "nochar_full_pos_unweighted": {"iter": "33,931", "vl": 1.0312, "bpj": 0.5049, "chrf": 3.21, "bleu": 0.02, "ner": 1.76, "dp": 1.47, "hs_raw": 29.0, "hs_len": 26.0, "km": 23.0, "kob": 38.89, "lat": 549.1, "t_loss": 0.6220, "t_ppl": 1.86, "jamo_surg": 37.50, "choseong": 43.75, "syl_cnt": 35.42, "d1": 81.00, "d10": 16.55, "oov": 95.38},
    "nochar_coarse_pos_new_weighted": {"iter": "45,000", "vl": 0.7984, "bpj": 0.5003, "chrf": 3.08, "bleu": 0.00, "ner": 1.71, "dp": 1.46, "hs_raw": 33.0, "hs_len": 25.0, "km": 18.0, "kob": 44.44, "lat": 370.6, "t_loss": 0.6165, "t_ppl": 1.85, "jamo_surg": 50.00, "choseong": 43.75, "syl_cnt": 39.58, "d1": 86.10, "d10": 17.49, "oov": 65.38},
    "nochar_coarse_pos_unweighted": {"iter": "33,931", "vl": 1.0189, "bpj": 0.4995, "chrf": 3.21, "bleu": 0.03, "ner": 1.76, "dp": 1.48, "hs_raw": 26.0, "hs_len": 26.0, "km": 23.0, "kob": 36.11, "lat": 634.6, "t_loss": 0.6180, "t_ppl": 1.86, "jamo_surg": 25.00, "choseong": 43.75, "syl_cnt": 39.58, "d1": 81.60, "d10": 16.31, "oov": 65.38},
    # Charhead WITH
    "nochar_full_pos_charhead_h11_with": {"iter": "45,542", "vl": 0.7757, "bpj": 0.5585, "chrf": 5.33, "bleu": 0.35, "ner": 1.69, "dp": 1.45, "hs_raw": 30.0, "hs_len": 22.0, "km": 23.0, "kob": 44.44, "lat": 368.9, "t_loss": 0.6880, "t_ppl": 1.99, "jamo_surg": 37.50, "choseong": 50.00, "syl_cnt": 45.83, "d1": 81.50, "d10": 16.50, "oov": 2.55},
    "nochar_full_pos_charhead_h11_no": {"iter": "45,542", "vl": 0.7757, "bpj": 0.5033, "chrf": 3.25, "bleu": 0.00, "ner": 1.69, "dp": 1.45, "hs_raw": 30.0, "hs_len": 22.0, "km": 23.0, "kob": 44.44, "lat": 388.9, "t_loss": 0.6200, "t_ppl": 1.86, "jamo_surg": 37.50, "choseong": 50.00, "syl_cnt": 45.83, "d1": 81.50, "d10": 16.50, "oov": 2.55},
    "nochar_full_pos_charhead_unweighted_with": {"iter": "34,082", "vl": 1.1512, "bpj": 0.5696, "chrf": 4.31, "bleu": 0.19, "ner": 1.75, "dp": 1.47, "hs_raw": 27.0, "hs_len": 26.0, "km": 23.0, "kob": 36.11, "lat": 391.0, "t_loss": 0.7020, "t_ppl": 2.02, "jamo_surg": 37.50, "choseong": 37.50, "syl_cnt": 33.33, "d1": 82.10, "d10": 16.60, "oov": 2.62},
    "nochar_full_pos_charhead_unweighted_no": {"iter": "34,082", "vl": 1.1512, "bpj": 0.5039, "chrf": 2.70, "bleu": 0.02, "ner": 1.75, "dp": 1.47, "hs_raw": 27.0, "hs_len": 26.0, "km": 23.0, "kob": 36.11, "lat": 375.4, "t_loss": 0.6210, "t_ppl": 1.86, "jamo_surg": 37.50, "choseong": 37.50, "syl_cnt": 33.33, "d1": 82.10, "d10": 16.60, "oov": 2.62},
    "nochar_coarse_pos_charhead_h11_with": {"iter": "7,088", "vl": 0.7635, "bpj": 0.6053, "chrf": 5.02, "bleu": 0.21, "ner": 1.80, "dp": 1.51, "hs_raw": 31.0, "hs_len": 28.0, "km": 23.0, "kob": 47.22, "lat": 383.1, "t_loss": 0.7450, "t_ppl": 2.11, "jamo_surg": 50.00, "choseong": 50.00, "syl_cnt": 41.67, "d1": 85.20, "d10": 17.20, "oov": 2.58},
    "nochar_coarse_pos_charhead_h11_no": {"iter": "7,088", "vl": 0.7635, "bpj": 0.5438, "chrf": 2.12, "bleu": 0.00, "ner": 1.80, "dp": 1.51, "hs_raw": 31.0, "hs_len": 28.0, "km": 23.0, "kob": 36.11, "lat": 386.4, "t_loss": 0.6700, "t_ppl": 1.95, "jamo_surg": 37.50, "choseong": 43.75, "syl_cnt": 27.08, "d1": 85.20, "d10": 17.20, "oov": 2.58},
    "nochar_coarse_pos_charhead_unweighted_with": {"iter": "34,082", "vl": 1.1398, "bpj": 0.5709, "chrf": 4.03, "bleu": 0.16, "ner": 1.76, "dp": 1.48, "hs_raw": 27.0, "hs_len": 26.0, "km": 23.0, "kob": 44.44, "lat": 385.5, "t_loss": 0.7036, "t_ppl": 2.02, "jamo_surg": 50.00, "choseong": 43.75, "syl_cnt": 39.58, "d1": 83.40, "d10": 16.80, "oov": 2.60},
    "nochar_coarse_pos_charhead_unweighted_no": {"iter": "34,082", "vl": 1.1398, "bpj": 0.5034, "chrf": 2.96, "bleu": 0.08, "ner": 1.76, "dp": 1.48, "hs_raw": 27.0, "hs_len": 26.0, "km": 23.0, "kob": 44.44, "lat": 381.8, "t_loss": 0.6205, "t_ppl": 1.86, "jamo_surg": 50.00, "choseong": 43.75, "syl_cnt": 39.58, "d1": 83.40, "d10": 16.80, "oov": 2.60},
}

# Muon tail and OOV map
muon_vt_map = {
    "baseline_character": [36.70, 10.08, 7.31],
    "baseline_char_bpe": [38.20, 10.03, 7.53],
    "baseline_pure_byte": [8.55, 2.81, 2.76],
    "three_hot_ind_char_byte": [41.90, 2.34, 1.98],
    "three_hot_cond_char_byte": [39.50, 1.36, 1.29],
    "mc_full_pos_weighted_byte": [45.20, 2.65, 2.56],
    "mc_full_pos_unweighted_byte": [45.10, 2.55, 2.46],
    "mc_coarse_pos_weighted_byte": [46.80, 2.69, 2.58],
    "mc_coarse_pos_unweighted_byte": [46.20, 2.71, 2.65],
    "nochar_full_pos_new_weighted": [80.50, 1.44, 1.65],
    "nochar_full_pos_unweighted": [80.80, 1.45, 1.70],
    "nochar_coarse_pos_new_weighted": [85.70, 1.44, 1.67],
    "nochar_coarse_pos_unweighted": [81.20, 1.45, 1.70],
    "nochar_full_pos_charhead_h11_with": [81.10, 1.43, 1.65],
    "nochar_full_pos_charhead_h11_no": [81.10, 1.43, 1.65],
    "nochar_full_pos_charhead_unweighted_with": [81.80, 1.45, 1.71],
    "nochar_full_pos_charhead_unweighted_no": [81.80, 1.45, 1.71],
    "nochar_coarse_pos_charhead_h11_with": [84.90, 1.44, 1.67],
    "nochar_coarse_pos_charhead_h11_no": [84.90, 1.44, 1.67],
    "nochar_coarse_pos_charhead_unweighted_with": [83.00, 1.45, 1.72],
    "nochar_coarse_pos_charhead_unweighted_no": [83.00, 1.45, 1.72],
}

print("Loaded all mappings successfully.")
