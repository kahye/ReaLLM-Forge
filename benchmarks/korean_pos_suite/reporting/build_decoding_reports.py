#!/usr/bin/env python3
"""Build comprehensive reports for Standard Greedy vs Joint Factor-Product Decoding for Muon and AdamW."""
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
ARTIFACT_DIR = "/usr/local/google/home/kahye/.gemini/jetski/brain/42658686-3d97-4f16-8655-83c460bfcac5"

# Load measured joint decoding results
with open(os.path.join(REPO_ROOT, "results_joint_factor_decoding_comparison.json")) as f:
    res1 = json.load(f)

with open(os.path.join(REPO_ROOT, "results_remaining_nochar_decoding.json")) as f:
    res2 = json.load(f)

# Combine results
measured = {**res1, **res2}

print("Loaded measured keys:", list(measured.keys()))
