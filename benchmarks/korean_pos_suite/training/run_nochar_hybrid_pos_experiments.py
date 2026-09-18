#!/usr/bin/env python3
"""Training and Evaluation Pipeline for 24-Lane Hangul Hybrid Factorizer (No Char Stream Overhead).

Trains 4 models across 10 epochs on the 24-lane hybrid corpus:
1. nochar_coarse_pos_weighted (17 coarse POS tags, weighted loss: w_bpe=1.0, w_pos=0.5, w_struct=0.05)
2. nochar_coarse_pos_unweighted (17 coarse POS tags, unweighted loss: w=1.0 for all 24 lanes)
3. nochar_full_pos_weighted (46 Sejong POS tags, weighted loss: w_bpe=1.0, w_pos=0.5, w_struct=0.05)
4. nochar_full_pos_unweighted (46 Sejong POS tags, unweighted loss: w=1.0 for all 24 lanes)

Saves checkpoints at 3, 5, and 10 epochs, then evaluates all saved checkpoints on:
- Four Benchmark Capabilities (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
- Ko-HellaSwag Zero-Shot Reasoning
- Phonetic Slang & Typo Resilience
- Vocabulary Tail Perplexity (Deciles 1 to 10)
- OOV / Unicode Robustness Stress Test
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
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
MAX_EVAL_EXAMPLES = 200

# 23 Common Factor Lanes
COMMON_LANES = [
    "korean_pos_mc_nochar/script_bpe",
    "korean_pos_mc_nochar/choseong",
    "korean_pos_mc_nochar/jungseong",
    "korean_pos_mc_nochar/jongseong",
    "korean_pos_mc_nochar/jung_base1",
    "korean_pos_mc_nochar/jung_base2",
    "korean_pos_mc_nochar/jung_has_w",
    "korean_pos_mc_nochar/jung_has_y",
    "korean_pos_mc_nochar/jung_has_i",
    "korean_pos_mc_nochar/jong_base1",
    "korean_pos_mc_nochar/jong_base2",
    "korean_pos_mc_nochar/jong_base3",
    "korean_pos_mc_nochar/choseong_tense",
    "korean_pos_mc_nochar/choseong_aspirated",
    "korean_pos_mc_nochar/choseong_nasal_liquid",
    "korean_pos_mc_nochar/choseong_place",
    "korean_pos_mc_nochar/jung_height",
    "korean_pos_mc_nochar/jung_backness",
    "korean_pos_mc_nochar/jung_round",
    "korean_pos_mc_nochar/jong_complex",
    "korean_pos_mc_nochar/has_batchim",
    "korean_pos_mc_nochar/syllable_index_mod",
    "korean_pos_mc_nochar/codepoint_mod",
]

EXPERIMENTS = {
    "nochar_coarse_pos_weighted": {
        "display_name": "No-Char Coarse POS (Weighted, ws=0.05, wpos=0.5)",
        "out_dir": "./out_mc_nochar_coarse_pos_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "pos_mode": "coarse",
        "char_head": False,
    },
    "nochar_coarse_pos_unweighted": {
        "display_name": "No-Char Coarse POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_coarse_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "pos_mode": "coarse",
        "char_head": False,
    },
    "nochar_full_pos_weighted": {
        "display_name": "No-Char Full POS (Weighted, ws=0.05, wpos=0.5)",
        "out_dir": "./out_mc_nochar_full_pos_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "pos_mode": "full",
        "char_head": False,
    },
    "nochar_full_pos_unweighted": {
        "display_name": "No-Char Full POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_full_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "pos_mode": "full",
        "char_head": False,
    },
    "nochar_coarse_pos_charhead_weighted": {
        "display_name": "No-Char Input + Char LM Head Coarse POS (Weighted, ws=0.05, wpos=0.5, wchar=1.0)",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5", "--char_loss_weight", "1.0"],
        "pos_mode": "coarse",
        "char_head": True,
    },
    "nochar_coarse_pos_charhead_unweighted": {
        "display_name": "No-Char Input + Char LM Head Coarse POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0", "--char_loss_weight", "1.0"],
        "pos_mode": "coarse",
        "char_head": True,
    },
    "nochar_full_pos_charhead_weighted": {
        "display_name": "No-Char Input + Char LM Head Full POS (Weighted, ws=0.05, wpos=0.5, wchar=1.0)",
        "out_dir": "./out_mc_nochar_full_pos_charhead_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5", "--char_loss_weight", "1.0"],
        "pos_mode": "full",
        "char_head": True,
    },
    "nochar_full_pos_charhead_unweighted": {
        "display_name": "No-Char Input + Char LM Head Full POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_full_pos_charhead_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0", "--char_loss_weight", "1.0"],
        "pos_mode": "full",
        "char_head": True,
    },
}


def get_epoch_iterations() -> Dict[str, int]:
    """Calculate exact iteration counts for 3, 5, 10, 15, and 20 epochs from train.bin."""
    train_bin = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "script_bpe", "train.bin")
    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Training dataset not found at {train_bin}. Run prepare_hybrid_pos_lanes.py first.")

    total_train_tokens = os.path.getsize(train_bin) // 2  # uint16
    tokens_per_iter = BATCH_SIZE * BLOCK_SIZE  # 16,384
    iters_per_epoch = total_train_tokens // tokens_per_iter

    iters_3ep = iters_per_epoch * 3
    iters_5ep = iters_per_epoch * 5
    iters_10ep = iters_per_epoch * 10
    iters_15ep = iters_per_epoch * 15
    iters_20ep = iters_per_epoch * 20

    print(f"Dataset Size: {total_train_tokens:,} training tokens.")
    print(f"Tokens/iter: {tokens_per_iter:,} (batch={BATCH_SIZE}, block={BLOCK_SIZE})")
    print(f"Iterations/epoch: {iters_per_epoch:,}")
    print(f"Milestones: 3ep={iters_3ep:,} | 5ep={iters_5ep:,} | 10ep={iters_10ep:,} | 15ep={iters_15ep:,} | 20ep={iters_20ep:,}")

    return {
        "iters_per_epoch": iters_per_epoch,
        "3_epochs": iters_3ep,
        "5_epochs": iters_5ep,
        "10_epochs": iters_10ep,
        "15_epochs": iters_15ep,
        "20_epochs": iters_20ep,
    }


def train_single_model(exp_key: str, cfg: Dict[str, Any], epoch_iters: Dict[str, int], out_dir_suffix: str = "_muon"):
    base_out_dir = cfg["out_dir"]
    out_dir = (base_out_dir + out_dir_suffix) if out_dir_suffix else base_out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Preserve existing 10-epoch ckpt.pt as 10_epochs.pt if not already saved
    prev_ckpt = os.path.join(out_dir, "ckpt.pt")
    ckpt_10ep = os.path.join(out_dir, "10_epochs.pt")
    if os.path.exists(prev_ckpt) and not os.path.exists(ckpt_10ep):
        try:
            ckpt_data = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
            if ckpt_data.get("iter_num", 0) >= epoch_iters["10_epochs"]:
                shutil.copyfile(prev_ckpt, ckpt_10ep)
                print(f"[{exp_key}] Preserved existing 10-epoch checkpoint as {ckpt_10ep}")
        except Exception as e:
            print(f"[{exp_key}] Warning checking ckpt.pt: {e}")

    lanes = [*COMMON_LANES, cfg["pos_lane"]]
    extra_args = []
    if cfg.get("char_head", False):
        lanes.append("korean_pos_mc_nochar/char")
        extra_args.extend(["--output_only_datasets", "korean_pos_mc_nochar/char"])

    primary_dataset = "korean_pos_mc_nochar/script_bpe"

    milestones = [
        ("3_epochs", epoch_iters["3_epochs"], "3_epochs.pt"),
        ("5_epochs", epoch_iters["5_epochs"], "5_epochs.pt"),
        ("10_epochs", epoch_iters["10_epochs"], "10_epochs.pt"),
        ("15_epochs", epoch_iters["15_epochs"], "15_epochs.pt"),
        ("20_epochs", epoch_iters["20_epochs"], "20_epochs.pt"),
    ]

    print("\n" + "=" * 80)
    print(f" TRAINING: {exp_key.upper()} ({cfg['display_name']})")
    print(f" Output Directory: {out_dir}")
    print(f" Lanes: {len(lanes)} total (Output-only char head: {cfg.get('char_head', False)})")
    print("=" * 80)

    for stage_name, target_iters, ckpt_filename in milestones:
        stage_ckpt = os.path.join(out_dir, ckpt_filename)
        if os.path.exists(stage_ckpt):
            try:
                ckpt_data = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                curr_iter = ckpt_data.get("iter_num", 0)
                if curr_iter >= target_iters:
                    print(f"[{exp_key}] Milestone {stage_name} ({ckpt_filename}) already at {curr_iter} >= {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{exp_key}] Running stage {stage_name} up to iter {target_iters:,}...")

        cmd = [
            sys.executable,
            "train.py",
            "--dataset", primary_dataset,
            "--training_mode", "multicontext",
            "--multicontext",
            "--multicontext_datasets", *lanes,
            *extra_args,
            *cfg["loss_args"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", str(epoch_iters["20_epochs"]),
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

        # Resume if previous checkpoint exists
        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {exp_key} at stage {stage_name} with exit code {res.returncode}")

        # Save milestone checkpoint
        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{exp_key}] Saved milestone checkpoint: {stage_ckpt}")

        print(f"[{exp_key}] Stage {stage_name} finished in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 24-lane hybrid Hangul factorizer training & evaluation with Muon.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()), default=None,
                        help="Select specific experiment(s) to run")
    parser.add_argument("--mode", choices=["all", "nochar_only", "charhead_only"], default="all",
                        help="Filter experiment group when --experiments is not specified (default: all)")
    parser.add_argument("--parallel", action="store_true", help="Run all selected experiments concurrently")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluations only")
    parser.add_argument("--out_dir_suffix", type=str, default="_muon",
                        help="Suffix appended to out_dir (default: '_muon' to isolate Muon runs from AdamW checkpoints)")
    args = parser.parse_args()

    epoch_iters = get_epoch_iterations()
    if args.experiments:
        exp_keys = args.experiments
    elif args.mode == "nochar_only":
        exp_keys = [k for k, v in EXPERIMENTS.items() if not v.get("char_head", False)]
    elif args.mode == "charhead_only":
        exp_keys = [k for k, v in EXPERIMENTS.items() if v.get("char_head", False)]
    else:
        exp_keys = list(EXPERIMENTS.keys())

    if not args.eval_only:
        if args.parallel and len(exp_keys) > 1:
            procs = []
            for key in exp_keys:
                cmd = [sys.executable, __file__, "--experiments", key]
                if args.out_dir_suffix:
                    cmd.extend(["--out_dir_suffix", args.out_dir_suffix])
                print(f"Launching background worker for {key}...")
                p = subprocess.Popen(cmd, cwd=REPO_ROOT)
                procs.append((key, p))
            for key, p in procs:
                ret = p.wait()
                if ret != 0:
                    raise RuntimeError(f"Parallel worker for {key} exited with code {ret}")
        else:
            for key in exp_keys:
                train_single_model(key, EXPERIMENTS[key], epoch_iters, out_dir_suffix=args.out_dir_suffix)

    print("\n==========================================================================")
    print(" ALL MODEL TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print("==========================================================================")
