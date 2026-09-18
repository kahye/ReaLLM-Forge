#!/usr/bin/env python3
"""Train and Evaluate 20-Epoch Hangul Hybrid Factorizer (No-Char) with Winning Loss Weights (C06_jamo_100_parity).

Trains two models in parallel for 20 epochs (45,240 iterations):
1. nochar_coarse_pos_new_weighted (17 coarse POS tags, w_bpe=1.0, w_jamo=1.0, w_aux=0.05, w_pos=0.5)
2. nochar_full_pos_new_weighted (46 Sejong POS tags, w_bpe=1.0, w_jamo=1.0, w_aux=0.05, w_pos=0.5)

Saves milestone checkpoints at 3, 5, 10, 15, and 20 epochs, plus best validation loss checkpoints (best_ckpt.pt).
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
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
EVAL_INTERVAL = 1000

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

# Winning weights: w_bpe=1.0, w_jamo=1.0 (choseong, jungseong, jongseong), w_aux=0.05 (19 lanes), w_pos=0.5
WINNING_WEIGHTS = [1.0, 1.0, 1.0, 1.0] + [0.05] * 19 + [0.5]

EXPERIMENTS = {
    "nochar_coarse_pos_new_weighted": {
        "display_name": "No-Char Coarse POS (New Winning Weights: wbpe=1.0, wjamo=1.0, wpos=0.5, waux=0.05)",
        "out_dir": "./out_mc_nochar_coarse_pos_new_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "pos_mode": "coarse",
    },
    "nochar_full_pos_new_weighted": {
        "display_name": "No-Char Full POS (New Winning Weights: wbpe=1.0, wjamo=1.0, wpos=0.5, waux=0.05)",
        "out_dir": "./out_mc_nochar_full_pos_new_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "pos_mode": "full",
    },
}


def get_epoch_iterations() -> Dict[str, int]:
    train_bin = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "script_bpe", "train.bin")
    total_train_tokens = os.path.getsize(train_bin) // 2
    tokens_per_iter = BATCH_SIZE * BLOCK_SIZE
    iters_per_epoch = total_train_tokens // tokens_per_iter
    return {
        "iters_per_epoch": iters_per_epoch,
        "3_epochs": iters_per_epoch * 3,
        "5_epochs": iters_per_epoch * 5,
        "10_epochs": iters_per_epoch * 10,
        "15_epochs": iters_per_epoch * 15,
        "20_epochs": iters_per_epoch * 20,
    }


def train_single_model(exp_key: str, cfg: Dict[str, Any], epoch_iters: Dict[str, int]):
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    prev_ckpt = os.path.join(out_dir, "ckpt.pt")
    lanes = [*COMMON_LANES, cfg["pos_lane"]]
    primary_dataset = "korean_pos_mc_nochar/script_bpe"

    milestones = [
        ("3_epochs", epoch_iters["3_epochs"], "3_epochs.pt"),
        ("5_epochs", epoch_iters["5_epochs"], "5_epochs.pt"),
        ("10_epochs", epoch_iters["10_epochs"], "10_epochs.pt"),
        ("15_epochs", epoch_iters["15_epochs"], "15_epochs.pt"),
        ("20_epochs", epoch_iters["20_epochs"], "20_epochs.pt"),
    ]

    print(f"\n================================================================================")
    print(f" TRAINING: {exp_key} ({cfg['display_name']})")
    print(f" Output Dir: {out_dir}")
    print(f"================================================================================")

    for stage_name, target_iters, ckpt_filename in milestones:
        stage_ckpt = os.path.join(out_dir, ckpt_filename)
        if os.path.exists(stage_ckpt):
            try:
                ckpt_data = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                if ckpt_data.get("iter_num", 0) >= target_iters:
                    print(f"[{exp_key}] Milestone {stage_name} already completed at iter {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{exp_key}] Running stage {stage_name} up to iter {target_iters:,}...")
        cmd = [
            sys.executable,
            "train.py",
            "--dataset",
            primary_dataset,
            "--training_mode",
            "multicontext",
            "--multicontext",
            "--multicontext_datasets",
            *lanes,
            "--multicontext_loss_weights",
            *[str(w) for w in WINNING_WEIGHTS],
            "--batch_size",
            str(BATCH_SIZE),
            "--block_size",
            str(BLOCK_SIZE),
            "--max_iters",
            str(target_iters),
            "--eval_interval",
            str(EVAL_INTERVAL),
            "--eval_iters",
            str(EVAL_ITERS),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters",
            str(epoch_iters["20_epochs"]),
            "--always_save_checkpoint",
            "--dropout",
            "0.1",
            "--device",
            DEVICE,
            "--out_dir",
            out_dir,
        ]

        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        log_file = os.path.join(out_dir, f"train_{stage_name}.log")
        with open(log_file, "w") as lf:
            res = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {exp_key} at {stage_name}. Check {log_file}")

        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{exp_key}] Saved milestone checkpoint: {stage_ckpt} in {(t1 - t0)/60:.1f}m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    epoch_iters = get_epoch_iterations()

    if args.experiment:
        train_single_model(args.experiment, EXPERIMENTS[args.experiment], epoch_iters)
    elif args.parallel:
        procs = []
        for key in EXPERIMENTS:
            cmd = [sys.executable, __file__, "--experiment", key]
            print(f"Launching parallel worker for {key}...")
            p = subprocess.Popen(cmd, cwd=REPO_ROOT)
            procs.append((key, p))
        for key, p in procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Worker {key} failed with exit code {ret}")
        print("\nALL PARALLEL 20-EPOCH TRAINING RUNS COMPLETED SUCCESSFULLY!")
    else:
        for key in EXPERIMENTS:
            train_single_model(key, EXPERIMENTS[key], epoch_iters)


if __name__ == "__main__":
    main()
