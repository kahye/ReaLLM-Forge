#!/usr/bin/env python3
"""Unified Master Runner and Downstream Benchmark Suite for 20-Epoch Muon Models.

Trains models to 20 epochs using the modern optimizer and architecture flags:
  --optimizer muon
  --weight_decay 0.0
  --use_qk_norm
  --use_qk_norm_scale
  --use_rotary_embeddings
  --no-use_abs_pos_embeddings

Saves milestone checkpoints at 5, 10, 15, and 20 epochs, plus best_ckpt.pt.
Evaluates all saved checkpoints on:
1. Seq2Seq English-to-Korean translation (BPJ, Target Loss, Target PPL, BLEU-4, chrF)
2. KoBALT & CUTE-Hangul Diagnostic Suite (Choseong Quiz, Particles, Assimilation, Vowel Harmony, Jamo Surgery)
3. OOV Stress Test (Archaic Hangul, Rare Hanja, Ancient Scripts, Emojis, Slang)
4. Vocabulary Tail Perplexity (Deciles 1 to 10)
"""
from __future__ import annotations
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

import argparse
from collections import Counter
from dataclasses import fields
from inspect import signature
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from jamo import h2j, j2hcj

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")
if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from hangul_factorizer import HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
EVAL_INTERVAL = 1000

_CHAR_TO_DECILE_CACHE: Optional[Dict[str, int]] = None

def get_cached_char_to_decile() -> Dict[str, int]:
    global _CHAR_TO_DECILE_CACHE
    if _CHAR_TO_DECILE_CACHE is None:
        corpus_txt = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "input.txt")
        from benchmarks.run_vocab_tail_perplexity import build_syllable_deciles
        c2d, _ = build_syllable_deciles(corpus_txt)
        _CHAR_TO_DECILE_CACHE = c2d
    return _CHAR_TO_DECILE_CACHE


# 23 Factor Lanes for Char+Byte Stream
COMMON_CHAR_LANES = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness",
    "korean_pos_mc/jung_round", "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim",
    "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
]

# 23 Factor Lanes for No-Char BPE Stream
COMMON_NOCHAR_LANES = [
    "korean_pos_mc_nochar/script_bpe", "korean_pos_mc_nochar/choseong", "korean_pos_mc_nochar/jungseong",
    "korean_pos_mc_nochar/jongseong", "korean_pos_mc_nochar/jung_base1", "korean_pos_mc_nochar/jung_base2",
    "korean_pos_mc_nochar/jung_has_w", "korean_pos_mc_nochar/jung_has_y", "korean_pos_mc_nochar/jung_has_i",
    "korean_pos_mc_nochar/jong_base1", "korean_pos_mc_nochar/jong_base2", "korean_pos_mc_nochar/jong_base3",
    "korean_pos_mc_nochar/choseong_tense", "korean_pos_mc_nochar/choseong_aspirated",
    "korean_pos_mc_nochar/choseong_nasal_liquid", "korean_pos_mc_nochar/choseong_place",
    "korean_pos_mc_nochar/jung_height", "korean_pos_mc_nochar/jung_backness", "korean_pos_mc_nochar/jung_round",
    "korean_pos_mc_nochar/jong_complex", "korean_pos_mc_nochar/has_batchim",
    "korean_pos_mc_nochar/syllable_index_mod", "korean_pos_mc_nochar/codepoint_mod",
]

# Winning Loss Weights
C06_WEIGHTS = [1.0, 1.0, 1.0, 1.0] + [0.05] * 19 + [0.5]
H11_WEIGHTS = [1.0, 0.5, 0.5, 0.5] + [0.05] * 19 + [0.5, 0.5]

MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. Baseline Models
    # --------------------------------------------------------------------------
    "baseline_character": {
        "display_name": "Baseline Character Tokenizer (V=4,924)",
        "family": "Baselines",
        "out_dir": "./out_baseline_korean_pos_muon",
        "dataset": "korean_pos_mc/char",
        "training_mode": "single",
        "type": "baseline_char",
        "multicontext": False,
        "loss_args": [],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "baseline_char_bpe": {
        "display_name": "Baseline Byte-Fallback BPE (V=4,924)",
        "family": "Baselines",
        "out_dir": "./out_baseline_char_bpe_muon",
        "dataset": "korean_pos_mc/char_bpe",
        "training_mode": "single",
        "type": "baseline_bpe",
        "multicontext": False,
        "loss_args": [],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "baseline_pure_byte": {
        "display_name": "Baseline Pure Byte Tokenizer (V=256)",
        "family": "Baselines",
        "out_dir": "./out_baseline_pure_byte_muon",
        "dataset": "korean_pos_mc/byte",
        "training_mode": "single",
        "type": "baseline_byte",
        "multicontext": False,
        "loss_args": [],
        "milestone_iters": {"5_epochs": 28108, "10_epochs": 56217, "15_epochs": 84325, "20_epochs": 112434},
        "lr_decay_iters": 112434,
    },

    # --------------------------------------------------------------------------
    # 2. Three-Hot Tokenizers
    # --------------------------------------------------------------------------
    "three_hot_ind_char_byte": {
        "display_name": "Three-Hot Independent Char/Byte (Song 2018)",
        "family": "Three-Hot",
        "out_dir": "./out_three_hot_ind_char_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_three_hot",
        "multicontext": True,
        "lanes": [
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ],
        "extra_train_args": [],
        "loss_args": [],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "three_hot_cond_char_byte": {
        "display_name": "Three-Hot Conditional Char/Byte (Cognetta 2023)",
        "family": "Three-Hot",
        "out_dir": "./out_three_hot_cond_char_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_three_hot",
        "multicontext": True,
        "lanes": [
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ],
        "extra_train_args": ["--three_hot_conditional"],
        "loss_args": [],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },

    # --------------------------------------------------------------------------
    # 3. Hangul Factorizer with Character Stream & Byte Fallback (25 streams)
    # --------------------------------------------------------------------------
    "mc_full_pos_weighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "family": "25-Lane Char Stream",
        "out_dir": "./out_mc_full_pos_weighted_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_full",
        "multicontext": True,
        "lanes": [*COMMON_CHAR_LANES, "korean_pos_mc/pos_full", "korean_pos_mc/char_byte"],
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "mc_full_pos_unweighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "family": "25-Lane Char Stream",
        "out_dir": "./out_mc_full_pos_unweighted_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_full",
        "multicontext": True,
        "lanes": [*COMMON_CHAR_LANES, "korean_pos_mc/pos_full", "korean_pos_mc/char_byte"],
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "mc_coarse_pos_weighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "family": "25-Lane Char Stream",
        "out_dir": "./out_mc_coarse_pos_weighted_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_coarse",
        "multicontext": True,
        "lanes": [*COMMON_CHAR_LANES, "korean_pos_mc/pos", "korean_pos_mc/char_byte"],
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },
    "mc_coarse_pos_unweighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "family": "25-Lane Char Stream",
        "out_dir": "./out_mc_coarse_pos_unweighted_byte_muon",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "type": "mc_25_coarse",
        "multicontext": True,
        "lanes": [*COMMON_CHAR_LANES, "korean_pos_mc/pos", "korean_pos_mc/char_byte"],
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "milestone_iters": {"5_epochs": 18165, "10_epochs": 36330, "15_epochs": 54495, "20_epochs": 72660},
        "lr_decay_iters": 72660,
    },

    # --------------------------------------------------------------------------
    # 4. No-Char Hangul Factorizer + Bytefallback BPE (24 streams)
    # --------------------------------------------------------------------------
    "nochar_full_pos_new_weighted": {
        "display_name": "No-Char Hangul Factorizer (Full POS, Newly Weighted C06)",
        "family": "24-Lane No-Char",
        "out_dir": "./out_mc_nochar_full_pos_new_weighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_full",
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos_full"],
        "loss_args": ["--multicontext_loss_weights", *[str(w) for w in C06_WEIGHTS]],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_full_pos_unweighted": {
        "display_name": "No-Char Hangul Factorizer (Full POS, Unweighted)",
        "family": "24-Lane No-Char",
        "out_dir": "./out_mc_nochar_full_pos_unweighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_full",
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos_full"],
        "loss_args": ["--multicontext_loss_weights", *["1.0"] * 24],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_coarse_pos_new_weighted": {
        "display_name": "No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)",
        "family": "24-Lane No-Char",
        "out_dir": "./out_mc_nochar_coarse_pos_new_weighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_coarse",
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos"],
        "loss_args": ["--multicontext_loss_weights", *[str(w) for w in C06_WEIGHTS]],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_coarse_pos_unweighted": {
        "display_name": "No-Char Hangul Factorizer (Coarse POS, Unweighted)",
        "family": "24-Lane No-Char",
        "out_dir": "./out_mc_nochar_coarse_pos_unweighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_coarse",
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos"],
        "loss_args": ["--multicontext_loss_weights", *["1.0"] * 24],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },

    # --------------------------------------------------------------------------
    # 5. Output-Only Target Lane Hangul Factorizer (25 streams, output_only_lanes=[24])
    # --------------------------------------------------------------------------
    "nochar_full_pos_charhead_h11": {
        "display_name": "Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)",
        "family": "Output-Only Target Lane",
        "out_dir": "./out_mc_nochar_full_pos_charhead_h11_muon",
        "src_dir": "./out_mc_nochar_full_pos_new_weighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_full",
        "is_charhead": True,
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos_full", "korean_pos_mc_nochar/char"],
        "loss_args": ["--multicontext_loss_weights", *[str(w) for w in H11_WEIGHTS]],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_full_pos_charhead_unweighted": {
        "display_name": "Output-Only Char Head Factorizer (Full POS, Unweighted)",
        "family": "Output-Only Target Lane",
        "out_dir": "./out_mc_nochar_full_pos_charhead_unweighted_muon",
        "src_dir": "./out_mc_nochar_full_pos_unweighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_full",
        "is_charhead": True,
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos_full", "korean_pos_mc_nochar/char"],
        "loss_args": ["--multicontext_loss_weights", *["1.0"] * 25],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_coarse_pos_charhead_h11": {
        "display_name": "Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)",
        "family": "Output-Only Target Lane",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_h11_muon",
        "src_dir": "./out_mc_nochar_coarse_pos_new_weighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_coarse",
        "is_charhead": True,
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos", "korean_pos_mc_nochar/char"],
        "loss_args": ["--multicontext_loss_weights", *[str(w) for w in H11_WEIGHTS]],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
    "nochar_coarse_pos_charhead_unweighted": {
        "display_name": "Output-Only Char Head Factorizer (Coarse POS, Unweighted)",
        "family": "Output-Only Target Lane",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_unweighted_muon",
        "src_dir": "./out_mc_nochar_coarse_pos_unweighted_muon",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "type": "mc_24_coarse",
        "is_charhead": True,
        "multicontext": True,
        "lanes": [*COMMON_NOCHAR_LANES, "korean_pos_mc_nochar/pos", "korean_pos_mc_nochar/char"],
        "loss_args": ["--multicontext_loss_weights", *["1.0"] * 25],
        "milestone_iters": {"5_epochs": 11310, "10_epochs": 22620, "15_epochs": 33930, "20_epochs": 45240},
        "lr_decay_iters": 45240,
    },
}

MILESTONE_LIST = ["5_epochs", "10_epochs", "15_epochs", "20_epochs"]


# ==============================================================================
# Helper: Warmstart 25-lane Checkpoint with Char LM Head
# ==============================================================================
def prepare_warmstart_25lane_checkpoint(src_24lane_ckpt: str, dst_25lane_ckpt: str) -> int:
    """Create a 25-lane checkpoint with output_only_lanes=[24] from the 24-lane checkpoint."""
    ckpt = torch.load(src_24lane_ckpt, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    vocab_sizes = list(model_args["vocab_sizes"])
    if len(vocab_sizes) == 24:
        vocab_sizes.append(11173)
    model_args["vocab_sizes"] = vocab_sizes
    model_args["output_only_lanes"] = [24]
    ckpt["model_args"] = model_args

    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    wte_1 = state_dict["transformer.wte_1.weight"]
    wte_2 = state_dict["transformer.wte_2.weight"]
    wte_3 = state_dict["transformer.wte_3.weight"]
    n_embd = wte_1.shape[1]

    lm_head_24_w = torch.zeros((11173, n_embd), dtype=wte_1.dtype)
    lm_head_24_w[0] = (wte_1[0] + wte_2[0] + wte_3[0]) / 3.0

    idx = 1
    for c in range(1, 20):
        for v in range(1, 22):
            for f in range(0, 28):
                lm_head_24_w[idx] = (wte_1[c] + wte_2[v] + wte_3[f]) / 3.0
                idx += 1

    state_dict["transformer.lm_head_24.weight"] = lm_head_24_w
    ckpt["best_val_loss"] = torch.tensor(float("inf"))
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        mc_datasets = ckpt["config"].get("multicontext_datasets", [])
        if len(mc_datasets) == 24:
            ckpt["config"]["multicontext_datasets"] = list(mc_datasets) + ["korean_pos_mc_nochar/char"]

    torch.save(ckpt, dst_25lane_ckpt)
    return ckpt.get("iter_num", 11310)


# ==============================================================================
# Model Training Logic
# ==============================================================================
def train_single_model(model_key: str, cfg: Dict[str, Any], max_stage: Optional[str] = None):
    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    prev_ckpt = os.path.join(out_dir, "ckpt.pt")

    print("\n" + "=" * 80)
    print(f" TRAINING: {model_key.upper()} ({cfg['display_name']})")
    print(f" Output Dir: {out_dir}")
    print("=" * 80)

    # If this is a charhead model, train by warmstarting from source 24-lane milestones
    if cfg.get("is_charhead", False):
        src_dir = os.path.join(REPO_ROOT, cfg["src_dir"])
        best_overall_val_loss = float("inf")
        best_overall_ckpt_path = None

        for stage_name in MILESTONE_LIST:
            if max_stage and MILESTONE_LIST.index(stage_name) > MILESTONE_LIST.index(max_stage):
                break
            stage_ckpt = os.path.join(out_dir, f"{stage_name}.pt")
            if os.path.exists(stage_ckpt):
                try:
                    d = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                    v_loss = float(d.get("best_val_loss", float("inf")))
                    if v_loss < best_overall_val_loss:
                        best_overall_val_loss = v_loss
                        best_overall_ckpt_path = stage_ckpt
                    print(f"[{model_key}] Milestone {stage_name} already exists (loss={v_loss:.4f}). Skipping.")
                    continue
                except Exception:
                    pass

            src_ckpt = os.path.join(src_dir, f"{stage_name}.pt")
            if not os.path.exists(src_ckpt):
                print(f"[{model_key}] Source checkpoint {src_ckpt} not found! Cannot finetune {stage_name} yet.")
                continue

            start_iter = prepare_warmstart_25lane_checkpoint(src_ckpt, prev_ckpt)
            finetune_iters = 300
            target_iters = start_iter + finetune_iters

            print(f"[{model_key}] Finetuning {stage_name} ({start_iter} -> {target_iters} iters)...")
            cmd = [
                sys.executable, "train.py",
                "--dataset", cfg["dataset"],
                "--training_mode", cfg["training_mode"],
                "--multicontext",
                "--multicontext_datasets", *cfg["lanes"],
                "--output_only_datasets", "korean_pos_mc_nochar/char",
                *cfg["loss_args"],
                "--batch_size", str(BATCH_SIZE),
                "--block_size", str(BLOCK_SIZE),
                "--max_iters", str(target_iters),
                "--no-lr_decay_match_max_iters",
                "--lr_decay_iters", str(cfg["lr_decay_iters"]),
                "--eval_interval", "100",
                "--eval_iters", str(EVAL_ITERS),
                "--always_save_checkpoint",
                "--init_from", "resume",
                "--dropout", "0.1",
                "--device", DEVICE,
                "--optimizer", "muon",
                "--weight_decay", "0.0",
                "--use_qk_norm",
                "--use_qk_norm_scale",
                "--use_rotary_embeddings",
                "--no-use_abs_pos_embeddings",
                "--out_dir", out_dir,
            ]
            t0 = time.time()
            res = subprocess.run(cmd, cwd=REPO_ROOT)
            t1 = time.time()
            if res.returncode != 0:
                raise RuntimeError(f"Finetuning failed for {model_key} at {stage_name} (exit code {res.returncode})")

            shutil.copyfile(prev_ckpt, stage_ckpt)
            d = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
            v_loss = float(d.get("best_val_loss", float("inf")))
            print(f"[{model_key}] Saved {stage_name}.pt (val_loss={v_loss:.4f}) in {t1 - t0:.1f}s")
            if v_loss < best_overall_val_loss:
                best_overall_val_loss = v_loss
                best_overall_ckpt_path = stage_ckpt

        best_dest = os.path.join(out_dir, "best_ckpt.pt")
        if best_overall_ckpt_path and os.path.exists(best_overall_ckpt_path):
            shutil.copyfile(best_overall_ckpt_path, best_dest)
            print(f"[{model_key}] Saved best validation checkpoint: {best_dest} (loss={best_overall_val_loss:.4f})")
        return

    # Standard models (Baselines, Three-Hot, Char+Byte Hangul, No-Char Hangul)
    for stage_name in MILESTONE_LIST:
        if max_stage and MILESTONE_LIST.index(stage_name) > MILESTONE_LIST.index(max_stage):
            break
        target_iters = cfg["milestone_iters"][stage_name]
        stage_ckpt = os.path.join(out_dir, f"{stage_name}.pt")

        if os.path.exists(stage_ckpt):
            try:
                d = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                curr_iter = d.get("iter_num", 0)
                if curr_iter >= target_iters:
                    print(f"[{model_key}] Milestone {stage_name} already completed at iter {curr_iter} >= {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{model_key}] Training stage {stage_name} up to iter {target_iters:,}...")

        cmd = [
            sys.executable, "train.py",
            "--dataset", cfg["dataset"],
            "--training_mode", cfg["training_mode"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", str(cfg["lr_decay_iters"]),
            "--eval_interval", str(EVAL_INTERVAL),
            "--eval_iters", str(EVAL_ITERS),
            "--always_save_checkpoint",
            "--dropout", "0.1",
            "--device", DEVICE,
            "--optimizer", "muon",
            "--weight_decay", "0.0",
            "--use_qk_norm",
            "--use_qk_norm_scale",
            "--use_rotary_embeddings",
            "--no-use_abs_pos_embeddings",
            "--out_dir", out_dir,
        ]

        if cfg["multicontext"]:
            cmd.extend(["--multicontext", "--multicontext_datasets", *cfg["lanes"]])
            cmd.extend(cfg["loss_args"])
            if cfg.get("extra_train_args"):
                cmd.extend(cfg["extra_train_args"])

        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {model_key} at stage {stage_name} (code {res.returncode})")

        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{model_key}] Saved milestone checkpoint: {stage_ckpt} (took {t1 - t0:.1f}s)")

    best_ckpt = os.path.join(out_dir, "best_ckpt.pt")
    if not os.path.exists(best_ckpt) and os.path.exists(prev_ckpt):
        shutil.copyfile(prev_ckpt, best_ckpt)


# ==============================================================================
# Model Evaluation Infrastructure
# ==============================================================================
def load_eval_model(ckpt_path: str, device: str = DEVICE) -> Tuple[GPT, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    filtered_args = {k: v for k, v in model_args.items() if k in allowed}
    filtered_args["dropout"] = 0.0
    conf = GPTConfig(**filtered_args)
    model = GPT(conf)
    state_dict = checkpoint["model"]
    prefix = "_orig_mod."
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            clean_sd[k[len(prefix):]] = v
        else:
            clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    model.eval().to(device)
    return model, checkpoint


def eval_checkpoint_suite(model_key: str, ckpt_path: str, fast_mode: bool = False) -> Dict[str, Any]:
    """Run all 4 downstream evaluation suites on a single checkpoint."""
    cfg = MODELS_CONFIG[model_key]
    m_type = cfg["type"]
    model, ckpt_data = load_eval_model(ckpt_path, DEVICE)
    val_loss = float(ckpt_data.get("best_val_loss", float("inf")))
    step_iter = int(ckpt_data.get("iter_num", 0))

    results: Dict[str, Any] = {
        "val_loss": round(val_loss, 4),
        "step_iter": step_iter,
    }

    # 1. KoBALT & CUTE-Hangul Diagnostic Suite
    from benchmarks.run_kobalt_cute_hangul_eval import (
        TASK_DATASETS,
        score_candidate_single,
        score_candidate_mc24,
        score_candidate_mc25,
    )
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_25_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")

    encode_fn = None
    stois = []
    if m_type.startswith("baseline_"):
        meta_file = "meta.pkl"
        meta_path = os.path.join(REPO_ROOT, cfg["out_dir"], meta_file)
        if not os.path.exists(meta_path):
            meta_path = os.path.join(REPO_ROOT, "data", cfg["dataset"], meta_file)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        encode_fn, _ = get_tokenizer_functions(meta)
    elif m_type.startswith("mc_25") or m_type.startswith("three_hot"):
        lane_names = cfg.get("lanes", [])
        for lp in lane_names:
            p = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    stois.append(pickle.load(f)["stoi"])

    kobalt_scores = {}
    total_corr, total_items = 0, 0
    for task_name, items in TASK_DATASETS.items():
        corr = 0
        for item in items:
            ctx, cands, ans = item["ctx"], item["candidates"], item["answer"]
            c_scores = []
            for cand in cands:
                if m_type.startswith("baseline_"):
                    s = score_candidate_single(model, encode_fn, ctx, cand, DEVICE)
                elif m_type.startswith("mc_24"):
                    tok_24 = tok_24_coarse if "coarse" in model_key else tok_24_full
                    s = score_candidate_mc24(model, tok_24, ctx, cand, DEVICE)
                elif m_type.startswith("mc_25"):
                    s = score_candidate_mc25(model, tok_25_full, stois, ctx, cand, DEVICE)
                elif m_type.startswith("three_hot"):
                    s = score_candidate_mc25(model, tok_25_full, stois, ctx, cand, DEVICE)
                else:
                    s = -math.inf
                c_scores.append(s)
            pred = int(max(range(len(c_scores)), key=lambda i: c_scores[i]))
            if pred == ans:
                corr += 1
        acc = round(corr / max(1, len(items)) * 100.0, 2)
        kobalt_scores[task_name] = acc
        total_corr += corr
        total_items += len(items)
    kobalt_scores["overall_accuracy"] = round(total_corr / max(1, total_items) * 100.0, 2)
    results["kobalt_cute_hangul"] = kobalt_scores

    # 2. Seq2Seq English-to-Korean Translation Benchmark
    from run_seq2seq_translation_benchmark import TEST_JSONL, TranslationEvaluator, canonicalize
    test_samples = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    eval_samples = test_samples[:25] if fast_mode else test_samples
    gen_samples = test_samples[:10] if fast_mode else test_samples[:20]

    eval_info = {
        "id": model_key,
        "name": cfg["display_name"],
        "type": m_type,
        "dir": cfg["out_dir"],
    }
    evaluator = TranslationEvaluator(eval_info, ckpt_data, device=DEVICE)
    corpus_metrics = evaluator.evaluate_teacher_forced(model, eval_samples)
    hypotheses, lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)

    ref_canons = [canonicalize(s["ko"]) for s in gen_samples]
    hyp_canons = [canonicalize(h) for h in hypotheses]
    b_score = sacrebleu.corpus_bleu(hyp_canons, [ref_canons])
    c_score = sacrebleu.corpus_chrf(hyp_canons, [ref_canons], char_order=18)

    results["seq2seq"] = {
        "bpj": round(corpus_metrics.get("bpj", 0.0), 4),
        "target_loss": round(corpus_metrics.get("target_loss", 0.0), 4),
        "target_ppl": round(corpus_metrics.get("target_ppl", 0.0), 2),
        "bleu": round(b_score.score, 2),
        "chrf": round(c_score.score, 2),
        "latency_ms": round(lat_ms, 1),
    }

    # 3. OOV Stress Test
    from benchmarks.run_oov_evaluations import get_prompts
    all_oov = get_prompts()
    oov_prompts = all_oov[:5] if fast_mode else all_oov
    oov_nlls = []
    for p in oov_prompts:
        p_text = p if isinstance(p, str) else p.get("prompt", "")
        if m_type.startswith("baseline_"):
            p_ids = encode_fn(p_text)
            if len(p_ids) > 1:
                v_size = getattr(model.config, "vocab_size", 4924)
                inp = torch.tensor([p_ids[:-1]], device=DEVICE, dtype=torch.long).clamp(0, v_size - 1)
                tgt = torch.tensor([p_ids[1:]], device=DEVICE, dtype=torch.long).clamp(0, v_size - 1)
                with torch.no_grad():
                    _, loss = model(inp, targets=tgt)
                if not torch.isnan(loss):
                    oov_nlls.append(loss.item())
        elif m_type.startswith("mc_24"):
            tok_24 = tok_24_coarse if "coarse" in model_key else tok_24_full
            steps = tok_24.encode_text(p_text)
            if len(steps) > 1:
                n_l = min(24, len(steps[0]))
                t_lists = [[s[i] for s in steps] for i in range(n_l)]
                vocab_sizes = getattr(model.config, "vocab_sizes", None)
                inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long) for l in t_lists]
                tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long) for l in t_lists]
                if vocab_sizes is not None and len(vocab_sizes) >= n_l:
                    inp = [inp[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_l)]
                    tgt = [tgt[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_l)]
                td = {f"lane_{i}": l for i, l in enumerate(inp)}
                tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
                with torch.no_grad():
                    _, losses = model(token_dict=td, target_dict=tg)
                vl = [l for l in losses if not torch.isnan(l)]
                if vl:
                    oov_nlls.append((sum(vl) / len(vl)).item())
        elif m_type.startswith("mc_25") or m_type.startswith("three_hot"):
            f_seq = tok_25_full.encode_text(p_text)
            if len(f_seq) > 1 and stois:
                n_lanes = len(stois)
                token_lists = [[] for _ in range(n_lanes)]
                for item in f_seq:
                    ch = item["char"]
                    idx = item["indices"]
                    if n_lanes == 4:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[0].append(byte_id)
                        for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                            t_char = tok_25_full.token_for(factor_idx, idx[factor_idx])
                            token_lists[lane_i].append(stois[lane_i].get(t_char, 0))
                    else:
                        for i in range(min(24, n_lanes - 1)):
                            t_char = tok_25_full.token_for(i, idx[i])
                            token_lists[i].append(stois[i].get(t_char, 0))
                        if n_lanes > 24:
                            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                            token_lists[24].append(byte_id)
                vocab_sizes = getattr(model.config, "vocab_sizes", None)
                inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long) for l in token_lists]
                tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long) for l in token_lists]
                if vocab_sizes is not None and len(vocab_sizes) >= n_lanes:
                    inp = [inp[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_lanes)]
                    tgt = [tgt[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_lanes)]
                td = {f"lane_{i}": l for i, l in enumerate(inp)}
                tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
                with torch.no_grad():
                    _, losses = model(token_dict=td, target_dict=tg)
                vl = [l for l in losses if not torch.isnan(l)]
                if vl:
                    oov_nlls.append((sum(vl) / len(vl)).item())
    results["oov_stress"] = {
        "avg_loss": round(float(np.mean(oov_nlls)), 4) if oov_nlls else None,
        "perplexity": round(float(np.exp(np.mean(oov_nlls))), 2) if oov_nlls else None,
    }

    # 4. Vocabulary Tail Decile Test
    char_to_decile = get_cached_char_to_decile()
    decile_to_chars: Dict[int, List[str]] = {}
    for c, d in char_to_decile.items():
        decile_to_chars.setdefault(d, []).append(c)

    sample_n = 5 if fast_mode else 15
    decile_prompts: List[Tuple[str, int, str]] = []
    for d in range(1, 11):
        chars_d = decile_to_chars.get(d, [])
        for c in chars_d[:sample_n]:
            decile_prompts.append((c, d, f"단어: {c}"))

    decile_losses: Dict[int, List[float]] = {d: [] for d in range(1, 11)}

    for c, d, prompt_str in decile_prompts:
        if m_type.startswith("baseline_"):
            ids = encode_fn(prompt_str)
            if len(ids) > 1:
                v_size = getattr(model.config, "vocab_size", 4924)
                inp = torch.tensor([ids[:-1]], device=DEVICE, dtype=torch.long).clamp(0, v_size - 1)
                tgt = torch.tensor([ids[1:]], device=DEVICE, dtype=torch.long).clamp(0, v_size - 1)
                with torch.no_grad():
                    logits, _ = model(inp, targets=tgt)
                lp = torch.log_softmax(logits, dim=-1).squeeze(0)
                tgt_token = tgt[0, -1].item()
                decile_losses[d].append(-lp[-1, tgt_token].item())
        elif m_type.startswith("mc_24"):
            tok_24 = tok_24_coarse if "coarse" in model_key else tok_24_full
            steps = tok_24.encode_text(prompt_str)
            if len(steps) > 1:
                n_l = 24
                t_lists = [[s[i] for s in steps] for i in range(n_l)]
                inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long) for l in t_lists]
                tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long) for l in t_lists]
                td = {f"lane_{i}": l for i, l in enumerate(inp)}
                tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
                with torch.no_grad():
                    logits_list, _ = model(token_dict=td, target_dict=tg)
                s_lp = 0.0
                for l_i in (0, 1, 2, 3):
                    lp = torch.log_softmax(logits_list[l_i], dim=-1).squeeze(0)
                    s_lp += lp[-1, tgt[l_i][0, -1].item()].item()
                decile_losses[d].append(-s_lp)
        elif m_type.startswith("mc_25") or m_type.startswith("three_hot"):
            f_seq = tok_25_full.encode_text(prompt_str)
            if len(f_seq) > 1 and stois:
                n_lanes = len(stois)
                token_lists = [[] for _ in range(n_lanes)]
                for item in f_seq:
                    ch = item["char"]
                    idx = item["indices"]
                    if n_lanes == 4:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[0].append(byte_id)
                        for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                            t_char = tok_25_full.token_for(factor_idx, idx[factor_idx])
                            token_lists[lane_i].append(stois[lane_i].get(t_char, 0))
                    else:
                        for i in range(min(24, n_lanes - 1)):
                            t_char = tok_25_full.token_for(i, idx[i])
                            token_lists[i].append(stois[i].get(t_char, 0))
                        if n_lanes > 24:
                            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                            token_lists[24].append(byte_id)
                inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long) for l in token_lists]
                tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long) for l in token_lists]
                td = {f"lane_{i}": l for i, l in enumerate(inp)}
                tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
                with torch.no_grad():
                    logits_list, _ = model(token_dict=td, target_dict=tg)
                s_lp = 0.0
                check_lanes = (1, 2, 3) if n_lanes == 4 else (1, 2, 3, 24)
                for l_i in check_lanes:
                    if l_i < len(logits_list):
                        lp = torch.log_softmax(logits_list[l_i], dim=-1).squeeze(0)
                        s_lp += lp[-1, tgt[l_i][0, -1].item()].item()
                decile_losses[d].append(-s_lp)

    decile_summary = {}
    for d in range(1, 11):
        vals = decile_losses[d]
        decile_summary[f"decile_{d}"] = round(float(np.exp(np.mean(vals))), 2) if vals else None
    results["vocab_tail_deciles"] = decile_summary

    # 5. Four Capabilities: KLUE-NER, KLUE-DP, NSMC & UnSmile 80% Noise, KorMedMCQA
    temp_4cap = f"/tmp/eval_4cap_{model_key}_{os.path.basename(ckpt_path).replace('.pt', '')}.json"
    max_ex_4cap = 25 if fast_mode else 100
    cmd_4cap = [
        sys.executable, "benchmarks/run_four_capability_evals.py",
        "--ckpts", f"{model_key}:{ckpt_path}",
        "--device", DEVICE,
        "--max_examples", str(max_ex_4cap),
        "--output_json", temp_4cap,
    ]
    try:
        subprocess.run(cmd_4cap, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_4cap):
            with open(temp_4cap, "r", encoding="utf-8") as f:
                d4 = json.load(f).get(model_key, {})
            results["four_capabilities"] = d4
            if os.path.exists(temp_4cap):
                os.remove(temp_4cap)
    except Exception as e:
        print(f"[{model_key}] Four capabilities eval error: {e}")

    # 6. Ko-HellaSwag Commonsense Reasoning (All 4 Normalization Modes)
    temp_hs = f"/tmp/eval_hs_{model_key}_{os.path.basename(ckpt_path).replace('.pt', '')}.json"
    max_ex_hs = 25 if fast_mode else 100
    cmd_hs = [
        sys.executable, "benchmarks/run_ko_hellaswag.py",
        "--ckpt_path", ckpt_path,
        "--device", DEVICE,
        "--eval_all_norms",
        "--max_examples", str(max_ex_hs),
        "--output_json", temp_hs,
    ]
    try:
        subprocess.run(cmd_hs, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_hs):
            with open(temp_hs, "r", encoding="utf-8") as f:
                dhs = json.load(f)
            results["ko_hellaswag"] = dhs.get("accuracies", {})
            if os.path.exists(temp_hs):
                os.remove(temp_hs)
    except Exception as e:
        print(f"[{model_key}] Ko-HellaSwag eval error: {e}")

    return results


def evaluate_model_milestones(model_key: str, fast_mode: bool = False, existing_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = MODELS_CONFIG[model_key]
    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
    model_res = dict(existing_results or {})

    milestones_to_check = [*MILESTONE_LIST, "best_ckpt"]
    for m in milestones_to_check:
        if m in model_res:
            print(f"[{model_key}] Milestone {m} already evaluated in results. Skipping.")
            continue
        fname = f"{m}.pt" if not m.endswith(".pt") else m
        ckpt_path = os.path.join(out_dir, fname)
        if os.path.exists(ckpt_path):
            print(f"[{model_key}] Evaluating milestone {m} ({ckpt_path})...")
            try:
                res = eval_checkpoint_suite(model_key, ckpt_path, fast_mode=fast_mode)
                model_res[m] = res
                print(f"[{model_key} @ {m}] Val Loss={res['val_loss']} | BPJ={res['seq2seq']['bpj']} | BLEU={res['seq2seq']['bleu']} | KoBALT={res['kobalt_cute_hangul']['overall_accuracy']}%")
            except Exception as e:
                print(f"[{model_key} @ {m}] Evaluation error: {e}")
        else:
            print(f"[{model_key}] Checkpoint {fname} not found in {out_dir}. Skipping.")

    return model_res


# ==============================================================================
# Master Report Generation Logic
# ==============================================================================
def build_markdown_report(results: Dict[str, Any], output_path: str = "report/comprehensive_muon_20ep_comparison_report.md"):
    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, output_path)), exist_ok=True)

    lines = []
    lines.append("# Comprehensive Downstream Evaluation Report: 20-Epoch Muon Models Across All Milestones & Tasks\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Experimental Protocol & Architecture Flags:**")
    lines.append("> - **Optimizer:** Muon (`--optimizer muon`, `--weight_decay 0.0`)")
    lines.append("> - **Modern Architecture:** QK-Norm (`--use_qk_norm --use_qk_norm_scale`) + Rotary Positional Embeddings (`--use_rotary_embeddings --no-use_abs_pos_embeddings`)")
    lines.append("> - **Softmax:** Preserved standard softmax for active Flash Attention acceleration (no `relu2max`).")
    lines.append("> - **Checkpoint Milestones:** Evaluated at 5, 10, 15, and 20 epochs, plus best validation loss checkpoint (`best_ckpt.pt`).")
    lines.append("> - **Loss Formulations:** Newly weighted C06 for No-Char Factorizers (`wjamo=1.0, waux=0.05, wpos=0.5`), winning H11 for Char-Head Factorizers (`wjamo=0.5, wchar=0.5, wpos=0.5, waux=0.05`).\n")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary & Evaluated Architectures\n")
    lines.append("This report compares the **best validation loss checkpoints (`best_ckpt.pt`)** and **milestones (5, 10, 15, 20 epochs)** across all **17 models (5 families)** trained with Muon, RoPE, and QK-Norm:")
    lines.append("1. **Baseline Character Tokenizer** (`baseline_character`, $V=4{,}924$)")
    lines.append("2. **Baseline Byte-Fallback BPE** (`baseline_char_bpe`, $V=4{,}924$)")
    lines.append("3. **Baseline Pure Byte Tokenizer** (`baseline_pure_byte`, $V=256$)")
    lines.append("4. **Three-Hot Tokenizers**: Independent (`three_hot_ind_char_byte`, Song 2018) and Conditional (`three_hot_cond_char_byte`, Cognetta 2023)")
    lines.append("5. **Hangul Factorizer WITH Character Stream & Byte Fallback (25 Streams)**: Full/Coarse POS, Weighted/Unweighted")
    lines.append("6. **No-Char Hangul Factorizer WITH Byte-Fallback BPE (24 Streams)**: Full/Coarse POS, Newly Weighted C06 / Unweighted")
    lines.append("7. **Output-Only Target Lane Hangul Factorizer WITH Byte-Fallback BPE (25 Streams, `output_only_lanes=[24]`)**: Full/Coarse POS, Newly Weighted H11 / Unweighted\n")

    # 2. Master Overview Table
    lines.append("## 2. Master Overview Table: Best Validation Checkpoints (`best_ckpt.pt`) Across All Downstream Tasks\n")
    lines.append("| Model Family & Variant | Best Iter | Val Loss | Seq2Seq BPJ (↓) | Seq2Seq chrF (↑) | KLUE-NER PPL (↓) | KLUE-DP PPL (↓) | Ko-HellaSwag Raw (↑) | Ko-HellaSwag Len-Norm (↑) | KorMedMCQA Acc (↑) | KoBALT & CUTE Overall Acc (↑) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        s_iter = best_data.get("step_iter", "-")
        v_loss = best_data.get("val_loss", "-")
        s2s = best_data.get("seq2seq", {})
        bpj = s2s.get("bpj", "-")
        chrf = s2s.get("chrf", "-")
        fc = best_data.get("four_capabilities", {})
        ner_ppl = fc.get("klue_ner", {}).get("perplexity", "-")
        if isinstance(ner_ppl, (int, float)):
            ner_ppl = f"{ner_ppl:.2f}"
        dp_ppl = fc.get("klue_dp", {}).get("perplexity", "-")
        if isinstance(dp_ppl, (int, float)):
            dp_ppl = f"{dp_ppl:.2f}"
        hs = best_data.get("ko_hellaswag", {})
        hs_raw = hs.get("none", "-")
        if isinstance(hs_raw, (int, float)):
            hs_raw = f"{hs_raw*100:.1f}%"
        hs_len = hs.get("length", "-")
        if isinstance(hs_len, (int, float)):
            hs_len = f"{hs_len*100:.1f}%"
        mcqa = fc.get("kormedqa", {}).get("accuracy", "-")
        if isinstance(mcqa, (int, float)):
            mcqa = f"{mcqa*100:.1f}%"
        kobalt = best_data.get("kobalt_cute_hangul", {}).get("overall_accuracy", "-")
        if isinstance(kobalt, (int, float)):
            kobalt = f"{kobalt:.2f}%"
        lines.append(f"| **{cfg['display_name']}** | `{s_iter}` | `{v_loss}` | `{bpj}` | `{chrf}` | `{ner_ppl}` | `{dp_ppl}` | **`{hs_raw}`** | **`{hs_len}`** | **`{mcqa}`** | **`{kobalt}`** |")

    # 3. Detailed Seq2Seq Translation Benchmark
    lines.append("\n---\n")
    lines.append("## 3. Detailed English-to-Korean Seq2Seq Translation Benchmark (`best_ckpt.pt`)\n")
    lines.append("All models evaluated on the English-to-Korean Seq2Seq Translation Benchmark (`data/korean_seq2seq_bench/test.jsonl`):\n")
    lines.append("| Model Family & Variant | Best Iter | Teacher-Forced BPJ (↓) | Target Loss (↓) | Target PPL (↓) | BLEU-4 (↑) | chrF (↑) | Latency (`ms / sent`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        s_iter = best_data.get("step_iter", "-")
        s2s = best_data.get("seq2seq", {})
        bpj = s2s.get("bpj", "-")
        t_loss = s2s.get("target_loss", "-")
        t_ppl = s2s.get("target_ppl", "-")
        bleu = s2s.get("bleu", "-")
        chrf = s2s.get("chrf", "-")
        lat = s2s.get("latency_ms", "-")
        lines.append(f"| **{cfg['display_name']}** | `{s_iter}` | **`{bpj}`** | `{t_loss}` | `{t_ppl}` | `{bleu}` | **`{chrf}`** | `{lat} ms` |")

    # 4. Detailed Four Capabilities & Zero-Shot Reasoning
    lines.append("\n---\n")
    lines.append("## 4. Detailed Four Capabilities & Zero-Shot Reasoning Evaluation (`best_ckpt.pt`)\n")
    lines.append("### 4.1. Syntactic & Entity Perplexity (`KLUE-NER` & `KLUE-DP`) and Domain QA (`KorMedMCQA`)\n")
    lines.append("| Model Family & Variant | KLUE-NER Avg LogProb | KLUE-NER PPL (↓) | KLUE-DP Avg LogProb | KLUE-DP PPL (↓) | KorMedMCQA Acc. (↑) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        fc = best_data.get("four_capabilities", {})
        ner_lp = fc.get("klue_ner", {}).get("avg_logprob", "-")
        if isinstance(ner_lp, (int, float)):
            ner_lp = f"{ner_lp:.4f}"
        ner_ppl = fc.get("klue_ner", {}).get("perplexity", "-")
        if isinstance(ner_ppl, (int, float)):
            ner_ppl = f"{ner_ppl:.2f}"
        dp_lp = fc.get("klue_dp", {}).get("avg_logprob", "-")
        if isinstance(dp_lp, (int, float)):
            dp_lp = f"{dp_lp:.4f}"
        dp_ppl = fc.get("klue_dp", {}).get("perplexity", "-")
        if isinstance(dp_ppl, (int, float)):
            dp_ppl = f"{dp_ppl:.2f}"
        mcqa = fc.get("kormedqa", {}).get("accuracy", "-")
        if isinstance(mcqa, (int, float)):
            mcqa = f"{mcqa*100:.1f}%"
        lines.append(f"| **{cfg['display_name']}** | `{ner_lp}` | **`{ner_ppl}`** | `{dp_lp}` | **`{dp_ppl}`** | **`{mcqa}`** |")

    lines.append("\n### 4.2. Zero-Shot Commonsense Reasoning (`Ko-HellaSwag` Across All 4 Normalizations)\n")
    lines.append("| Model Family & Variant | Raw Acc (`none`) | Length-Norm (`length`) | Prior-Norm (`prior`) | Unigram-Norm (`unigram`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        hs = best_data.get("ko_hellaswag", {})
        r_raw = hs.get("none", "-")
        if isinstance(r_raw, (int, float)):
            r_raw = f"{r_raw*100:.1f}%"
        r_len = hs.get("length", "-")
        if isinstance(r_len, (int, float)):
            r_len = f"{r_len*100:.1f}%"
        r_pri = hs.get("prior", "-")
        if isinstance(r_pri, (int, float)):
            r_pri = f"{r_pri*100:.1f}%"
        r_uni = hs.get("unigram", "-")
        if isinstance(r_uni, (int, float)):
            r_uni = f"{r_uni*100:.1f}%"
        lines.append(f"| **{cfg['display_name']}** | **`{r_raw}`** | **`{r_len}`** | `{r_pri}` | `{r_uni}` |")

    # 5. KoBALT & CUTE-Hangul Diagnostic Suite
    lines.append("\n---\n")
    lines.append("## 5. KoBALT & CUTE-Hangul Diagnostic Breakdown at Best Checkpoint\n")
    lines.append("| Model Name | Choseong Quiz | Particles | Assimilation | Vowel Harmony | Jamo Surgery | **Overall** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        kb = best_data.get("kobalt_cute_hangul", {})
        q1 = f"{kb.get('choseong_quiz', '-')}%"
        q2 = f"{kb.get('allomorphic_particles', '-')}%"
        q3 = f"{kb.get('phonological_assimilation', '-')}%"
        q4 = f"{kb.get('vowel_harmony_ideophones', '-')}%"
        q5 = f"{kb.get('cute_jamo_manipulation', '-')}%"
        ov = f"{kb.get('overall_accuracy', '-')}%"
        lines.append(f"| {cfg['display_name']} | {q1} | {q2} | {q3} | {q4} | {q5} | **{ov}** |")

    # 6. Extreme Noise Resilience
    lines.append("\n---\n")
    lines.append("## 6. Extreme Noise Resilience (`NSMC 80%` & `UnSmile 80%`) Evaluation (`best_ckpt.pt`)\n")
    lines.append("| Model Family & Variant | NSMC Clean LogProb | NSMC 80% Deg. (↓) | UnSmile Clean LogProb | UnSmile 80% Deg. (↓) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        fc = best_data.get("four_capabilities", {})
        nt = fc.get("noisy_text", {})
        nsmc = nt.get("nsmc", {})
        unsmile = nt.get("unsmile", {})
        n_cln = nsmc.get("clean_logprob", "-")
        if isinstance(n_cln, (int, float)):
            n_cln = f"{n_cln:.4f}"
        n_deg = nsmc.get("degradation", "-")
        if isinstance(n_deg, (int, float)):
            n_deg = f"{n_deg:.2f}"
        u_cln = unsmile.get("clean_logprob", "-")
        if isinstance(u_cln, (int, float)):
            u_cln = f"{u_cln:.4f}"
        u_deg = unsmile.get("degradation", "-")
        if isinstance(u_deg, (int, float)):
            u_deg = f"{u_deg:.2f}"
        lines.append(f"| **{cfg['display_name']}** | `{n_cln}` | **`{n_deg}`** | `{u_cln}` | **`{u_deg}`** |")

    # 7. Milestone Progression Across Epochs
    lines.append("\n---\n")
    lines.append("## 7. Milestone Progression Across Epochs (5, 10, 15, and 20 Epochs)\n")
    lines.append(r"| Model Name | Milestone | Val Loss | BPJ $\downarrow$ | BLEU-4 $\uparrow$ | chrF $\uparrow$ | KoBALT Overall $\uparrow$ |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        for ms in MILESTONE_LIST:
            if ms in m_data:
                d = m_data[ms]
                v_loss = d.get("val_loss", "-")
                s2s = d.get("seq2seq", {})
                bpj = s2s.get("bpj", "-")
                bleu = s2s.get("bleu", "-")
                chrf = s2s.get("chrf", "-")
                kobalt = d.get("kobalt_cute_hangul", {}).get("overall_accuracy", "-")
                if isinstance(kobalt, (int, float)):
                    kobalt = f"{kobalt:.1f}%"
                lines.append(f"| {cfg['display_name']} | {ms} | `{v_loss}` | `{bpj}` | `{bleu}` | `{chrf}` | `{kobalt}` |")

    # 8. Vocabulary Tail Perplexity & OOV
    lines.append("\n---\n")
    lines.append("## 8. Vocabulary Tail Perplexity (Deciles 1 to 10 Zero-Shot) & OOV Stress Test\n")
    lines.append("| Model Name | Decile 1 (Top 10%) | Decile 3 | Decile 5 | Decile 7 | Decile 10 (Rarest 10%) | OOV PPL (↓) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        if not best_data:
            continue
        vt = best_data.get("vocab_tail_deciles", {})
        d1 = vt.get("decile_1", "-")
        d3 = vt.get("decile_3", "-")
        d5 = vt.get("decile_5", "-")
        d7 = vt.get("decile_7", "-")
        d10 = vt.get("decile_10", "-")
        oov = best_data.get("oov_stress", {}).get("perplexity", "-")
        lines.append(f"| {cfg['display_name']} | `{d1}` | `{d3}` | `{d5}` | `{d7}` | `{d10}` | `{oov}` |")

    # 9. Architectural Comparison (Muon vs. AdamW)
    lines.append("\n---\n")
    lines.append("## 9. Architectural Comparison: 20-Epoch Muon + RoPE + QK-Norm (New) vs. Prior 20-Epoch AdamW Baselines\n")
    lines.append("| Model Name | Prior Val Loss (AdamW) | New Val Loss (Muon) | Prior BPJ | New BPJ | Prior chrF | New chrF | Prior KoBALT | New KoBALT | Prior Ko-HellaSwag Len | New Ko-HellaSwag Len |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    prior_metrics = {
        "baseline_character": {"val_loss": "1.4720", "bpj": "1.6775", "chrf": "5.90", "kobalt": "41.67%", "hs_len": "18.0%"},
        "baseline_char_bpe": {"val_loss": "1.4768", "bpj": "1.6665", "chrf": "5.89", "kobalt": "44.44%", "hs_len": "20.0%"},
        "baseline_pure_byte": {"val_loss": "1.0072", "bpj": "1.3713", "chrf": "3.34", "kobalt": "38.89%", "hs_len": "21.0%"},
        "three_hot_ind_char_byte": {"val_loss": "0.5595", "bpj": "0.8144", "chrf": "2.68", "kobalt": "30.56%", "hs_len": "25.0%"},
        "three_hot_cond_char_byte": {"val_loss": "0.4575", "bpj": "0.5941", "chrf": "5.03", "kobalt": "36.11%", "hs_len": "17.0%"},
        "mc_full_pos_weighted_byte": {"val_loss": "0.5790", "bpj": "0.4875", "chrf": "3.75", "kobalt": "41.67%", "hs_len": "25.0%"},
        "mc_full_pos_unweighted_byte": {"val_loss": "0.2639", "bpj": "0.4761", "chrf": "3.34", "kobalt": "50.00%", "hs_len": "25.0%"},
        "mc_coarse_pos_weighted_byte": {"val_loss": "0.5723", "bpj": "0.4948", "chrf": "2.91", "kobalt": "41.67%", "hs_len": "24.0%"},
        "mc_coarse_pos_unweighted_byte": {"val_loss": "0.2664", "bpj": "0.4839", "chrf": "2.48", "kobalt": "44.44%", "hs_len": "23.0%"},
        "nochar_full_pos_new_weighted": {"val_loss": "0.8112", "bpj": "0.5019", "chrf": "2.83", "kobalt": "36.11%", "hs_len": "22.0%"},
        "nochar_full_pos_unweighted": {"val_loss": "1.0312", "bpj": "0.4978", "chrf": "2.85", "kobalt": "38.89%", "hs_len": "22.0%"},
        "nochar_coarse_pos_new_weighted": {"val_loss": "0.7984", "bpj": "0.5003", "chrf": "3.08", "kobalt": "44.44%", "hs_len": "25.0%"},
        "nochar_coarse_pos_unweighted": {"val_loss": "1.0189", "bpj": "0.4969", "chrf": "3.35", "kobalt": "36.11%", "hs_len": "25.0%"},
        "nochar_full_pos_charhead_h11": {"val_loss": "0.7757", "bpj": "0.5621", "chrf": "5.02", "kobalt": "41.67%", "hs_len": "22.0%"},
        "nochar_full_pos_charhead_unweighted": {"val_loss": "1.1512", "bpj": "0.5696", "chrf": "4.31", "kobalt": "36.11%", "hs_len": "26.0%"},
        "nochar_coarse_pos_charhead_h11": {"val_loss": "0.7635", "bpj": "0.5603", "chrf": "4.86", "kobalt": "44.44%", "hs_len": "25.0%"},
        "nochar_coarse_pos_charhead_unweighted": {"val_loss": "1.1398", "bpj": "0.5709", "chrf": "4.03", "kobalt": "44.44%", "hs_len": "26.0%"},
    }

    for m_key, cfg in MODELS_CONFIG.items():
        m_data = results.get(m_key, {})
        best_data = m_data.get("best_ckpt", m_data.get("20_epochs", {}))
        v_loss = best_data.get("val_loss", "-") if best_data else "-"
        bpj = best_data.get("seq2seq", {}).get("bpj", "-") if best_data else "-"
        chrf = best_data.get("seq2seq", {}).get("chrf", "-") if best_data else "-"
        kb = best_data.get("kobalt_cute_hangul", {}).get("overall_accuracy", "-") if best_data else "-"
        if isinstance(kb, (int, float)):
            kb = f"{kb:.1f}%"
        hs_len = best_data.get("ko_hellaswag", {}).get("length", "-") if best_data else "-"
        if isinstance(hs_len, (int, float)):
            hs_len = f"{hs_len*100:.1f}%"
        pm = prior_metrics.get(m_key, {"val_loss": "N/A", "bpj": "N/A", "chrf": "N/A", "kobalt": "N/A", "hs_len": "N/A"})
        lines.append(f"| {cfg['display_name']} | `{pm['val_loss']}` | `{v_loss}` | `{pm['bpj']}` | `{bpj}` | `{pm['chrf']}` | `{chrf}` | `{pm['kobalt']}` | `{kb}` | `{pm['hs_len']}` | `{hs_len}` |")

    full_md = "\n".join(lines) + "\n"
    out_file = os.path.join(REPO_ROOT, output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"\nGenerated comprehensive report: {out_file}")
    return out_file


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 20-Epoch Muon Runner & Benchmark Suite")
    parser.add_argument("--action", choices=["train", "eval", "report", "all"], default="all",
                        help="Action to execute: 'train', 'eval', 'report', or 'all'")
    parser.add_argument("--models", nargs="+", choices=list(MODELS_CONFIG.keys()), default=None,
                        help="Specific model keys to process")
    parser.add_argument("--family", choices=["baselines", "three_hot", "hangul_byte", "nochar", "charhead"], default=None,
                        help="Family group to process")
    parser.add_argument("--max_stage", choices=MILESTONE_LIST, default=None,
                        help="Train/eval only up to specified milestone (e.g. '5_epochs')")
    parser.add_argument("--fast_eval", action="store_true", help="Run fast downstream evaluations")
    parser.add_argument("--results_file", type=str, default="results_muon_20ep_all.json",
                        help="JSON path for aggregated benchmark results")
    parser.add_argument("--report_file", type=str, default="report/comprehensive_muon_20ep_comparison_report.md",
                        help="Markdown report path")
    args = parser.parse_args()

    # Determine target model keys
    if args.models:
        target_keys = args.models
    elif args.family:
        family_map = {
            "baselines": ["baseline_character", "baseline_char_bpe", "baseline_pure_byte"],
            "three_hot": ["three_hot_ind_char_byte", "three_hot_cond_char_byte"],
            "hangul_byte": ["mc_full_pos_weighted_byte", "mc_full_pos_unweighted_byte", "mc_coarse_pos_weighted_byte", "mc_coarse_pos_unweighted_byte"],
            "nochar": ["nochar_full_pos_new_weighted", "nochar_full_pos_unweighted", "nochar_coarse_pos_new_weighted", "nochar_coarse_pos_unweighted"],
            "charhead": ["nochar_full_pos_charhead_h11", "nochar_full_pos_charhead_unweighted", "nochar_coarse_pos_charhead_h11", "nochar_coarse_pos_charhead_unweighted"],
        }
        target_keys = family_map[args.family]
    else:
        target_keys = list(MODELS_CONFIG.keys())

    # Load existing results if present
    results_path = os.path.join(REPO_ROOT, args.results_file)
    master_results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                master_results = json.load(f)
        except Exception:
            pass

    # 1. Training Phase
    if args.action in ("train", "all"):
        print("\n" + "=" * 80)
        print(f" EXECUTION PHASE 1: TRAINING ({len(target_keys)} models)")
        print("=" * 80)
        for k in target_keys:
            try:
                train_single_model(k, MODELS_CONFIG[k], max_stage=args.max_stage)
                if args.action == "all":
                    if k in master_results and all(m in master_results[k] for m in [*MILESTONE_LIST, "best_ckpt"]):
                        print(f"[{k}] All milestone evaluations already completed. Skipping.")
                    else:
                        print(f"\n[Immediate Eval] Running downstream evaluations for {k}...")
                        m_res = evaluate_model_milestones(k, fast_mode=args.fast_eval, existing_results=master_results.get(k, {}))
                        if m_res:
                            master_results[k] = m_res
                            with open(results_path, "w", encoding="utf-8") as f:
                                json.dump(master_results, f, ensure_ascii=False, indent=2)
                            build_markdown_report(master_results, output_path=args.report_file)
            except Exception as e:
                print(f"ERROR: Training failed for {k}: {e}")

    # 2. Evaluation Phase
    if args.action in ("eval", "all"):
        print("\n" + "=" * 80)
        print(f" EXECUTION PHASE 2: DOWNSTREAM EVALUATIONS ({len(target_keys)} models)")
        print("=" * 80)
        for k in target_keys:
            if k in master_results and all(m in master_results[k] for m in [*MILESTONE_LIST, "best_ckpt"]):
                print(f"[{k}] All evaluations already up-to-date in results. Skipping.")
                continue
            m_res = evaluate_model_milestones(k, fast_mode=args.fast_eval, existing_results=master_results.get(k, {}))
            if m_res:
                master_results[k] = m_res
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(master_results, f, ensure_ascii=False, indent=2)

    # 3. Report Generation Phase
    if args.action in ("report", "all") and master_results:
        print("\n" + "=" * 80)
        print(f" EXECUTION PHASE 3: REPORT GENERATION")
        print("=" * 80)
        build_markdown_report(master_results, output_path=args.report_file)


if __name__ == "__main__":
    main()
