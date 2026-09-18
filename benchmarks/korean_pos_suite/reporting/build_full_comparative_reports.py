#!/usr/bin/env python3
"""Script to build comprehensive markdown reports for both Muon and AdamW models
comparing the best downstream checkpoints across all metric types.
"""
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
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTIFACT_PATH = "/usr/local/google/home/kahye/.gemini/jetski/brain/42658686-3d97-4f16-8655-83c460bfcac5/muon_and_adamw_best_downstream_checkpoints_report.md"

# Load Muon data
with open(os.path.join(REPO_ROOT, "results_muon_20ep_all.json")) as f:
    muon_all = json.load(f)

with open(os.path.join(REPO_ROOT, "results_charhead_without_lmhead24_muon_all_metrics.json")) as f:
    muon_no_ch = json.load(f)

# Load AdamW data
with open(os.path.join(REPO_ROOT, "results_all_11_models_20ep_comprehensive.json")) as f:
    adamw_11 = json.load(f)

with open(os.path.join(REPO_ROOT, "results_seq2seq_translation_all_checkpoints.json")) as f:
    adamw_s2s = json.load(f)

with open(os.path.join(REPO_ROOT, "results_new_weighted_nochar_all_epochs.json")) as f:
    adamw_nw = json.load(f)

with open(os.path.join(REPO_ROOT, "results_charhead_h11_20ep_all_metrics.json")) as f:
    adamw_ch = json.load(f)

with open(os.path.join(REPO_ROOT, "results_charhead_without_lmhead24_all_metrics.json")) as f:
    adamw_no_ch = json.load(f)

with open(os.path.join(REPO_ROOT, "eval_results_three_hot_59m_all.json")) as f:
    adamw_th = json.load(f)

print("Building Muon and AdamW reports...")
