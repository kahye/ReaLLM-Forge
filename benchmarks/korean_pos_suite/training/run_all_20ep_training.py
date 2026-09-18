#!/usr/bin/env python3
"""Train all 7 models (3 Baselines + 4 Hangul Factorizer with Character Stream and Byte Fallback)
to 20 epochs on the 59M token dataset using modern architecture and optimizer flags:
  --optimizer muon
  --weight_decay 0.0
  --use_qk_norm
  --use_qk_norm_scale
  --use_rotary_embeddings
  --no-use_abs_pos_embeddings

Monitors validation loss and saves checkpoints at 10 epochs, 15 epochs, 20 epochs,
and best validation loss (`best_ckpt.pt`).
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
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50

COMMON_LANES = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
]

MODELS: Dict[str, Dict[str, Any]] = {
    "baseline_character": {
        "display_name": "Baseline Character Tokenizer (V=4,924)",
        "out_dir": "./out_baseline_korean_pos",
        "dataset": "korean_pos_mc/char",
        "training_mode": "single",
        "multicontext": False,
        "loss_args": [],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "baseline_char_bpe": {
        "display_name": "Baseline Byte-Fallback BPE (V=4,924)",
        "out_dir": "./out_baseline_char_bpe",
        "dataset": "korean_pos_mc/char_bpe",
        "training_mode": "single",
        "multicontext": False,
        "loss_args": [],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "baseline_pure_byte": {
        "display_name": "Baseline Pure Byte Tokenizer (V=256)",
        "out_dir": "./out_baseline_pure_byte",
        "dataset": "korean_pos_mc/byte",
        "training_mode": "single",
        "multicontext": False,
        "loss_args": [],
        "iters_5ep": 28108,
        "iters_10ep": 56217,
        "iters_15ep": 84325,
        "iters_20ep": 112434,
    },
    "full_pos_weighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "out_dir": "./out_mc_full_pos_weighted_byte",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "multicontext": True,
        "pos_lane": "korean_pos_mc/pos_full",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "full_pos_unweighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "out_dir": "./out_mc_full_pos_unweighted_byte",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "multicontext": True,
        "pos_lane": "korean_pos_mc/pos_full",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "coarse_pos_weighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "out_dir": "./out_mc_coarse_pos_weighted_byte",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "multicontext": True,
        "pos_lane": "korean_pos_mc/pos",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "coarse_pos_unweighted_byte": {
        "display_name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "out_dir": "./out_mc_coarse_pos_unweighted_byte",
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "multicontext": True,
        "pos_lane": "korean_pos_mc/pos",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
}


def train_single_model(key: str, cfg: Dict[str, Any], out_dir_suffix: str = "_muon"):
    base_out_dir = cfg["out_dir"]
    out_dir = (base_out_dir + out_dir_suffix) if out_dir_suffix else base_out_dir
    os.makedirs(out_dir, exist_ok=True)

    prev_ckpt = os.path.join(out_dir, "ckpt.pt")
    ckpt_5ep = os.path.join(out_dir, "5_epochs.pt")
    ckpt_10ep = os.path.join(out_dir, "10_epochs.pt")

    # Preserve existing 5 or 10-epoch checkpoint if present
    if not os.path.exists(ckpt_5ep):
        exact_5_file = os.path.join(out_dir, f"{cfg['iters_5ep']}.pt")
        if os.path.exists(exact_5_file):
            shutil.copyfile(exact_5_file, ckpt_5ep)
            print(f"[{key}] Copied {exact_5_file} -> {ckpt_5ep}")

    if not os.path.exists(ckpt_10ep):
        exact_10_file = os.path.join(out_dir, f"{cfg['iters_10ep']}.pt")
        if os.path.exists(exact_10_file):
            shutil.copyfile(exact_10_file, ckpt_10ep)
            print(f"[{key}] Copied {exact_10_file} -> {ckpt_10ep}")

    milestones = [
        ("5_epochs", cfg["iters_5ep"], "5_epochs.pt"),
        ("10_epochs", cfg["iters_10ep"], "10_epochs.pt"),
        ("15_epochs", cfg["iters_15ep"], "15_epochs.pt"),
        ("20_epochs", cfg["iters_20ep"], "20_epochs.pt"),
    ]

    print("\n" + "=" * 80)
    print(f" TRAINING: {key.upper()} ({cfg['display_name']})")
    print(f" Output Directory: {out_dir}")
    print(f" Target 20 Epochs: {cfg['iters_20ep']:,} iterations")
    print("=" * 80)

    for stage_name, target_iters, ckpt_filename in milestones:
        stage_ckpt = os.path.join(out_dir, ckpt_filename)
        if os.path.exists(stage_ckpt):
            try:
                d = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                curr_iter = d.get("iter_num", 0)
                if curr_iter >= target_iters:
                    print(f"[{key}] Milestone {stage_name} ({ckpt_filename}) already at {curr_iter} >= {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{key}] Running stage {stage_name} up to iter {target_iters:,}...")

        cmd = [
            sys.executable,
            "train.py",
            "--dataset", cfg["dataset"],
            "--training_mode", cfg["training_mode"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", str(cfg["iters_20ep"]),
            "--eval_interval", "1000",
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
            lanes = [*COMMON_LANES, cfg["pos_lane"], cfg["char_lane"]]
            cmd.extend(["--multicontext", "--multicontext_datasets", *lanes])
            cmd.extend(cfg["loss_args"])

        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {key} at stage {stage_name} with exit code {res.returncode}")

        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{key}] Saved milestone checkpoint: {stage_ckpt}")

        print(f"[{key}] Stage {stage_name} finished in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")


def main():
    parser = argparse.ArgumentParser(description="Train 3 Baselines + 4 Byte-POS Hangul models to 20 epochs with Muon.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=None,
                        help="Select specific model(s) to train")
    parser.add_argument("--group", choices=["all", "baselines", "hangul"], default=None,
                        help="Select group of models to train: 'baselines' (3 baseline models), 'hangul' (4 Hangul factorizer models), or 'all'")
    parser.add_argument("--parallel", action="store_true", help="Run selected models concurrently")
    parser.add_argument("--max_parallel", type=int, default=4, help="Max concurrent processes")
    parser.add_argument("--out_dir_suffix", type=str, default="_muon",
                        help="Suffix appended to out_dir (default: '_muon' to isolate Muon runs from AdamW baseline checkpoints)")
    args = parser.parse_args()

    if args.models:
        keys = args.models
    elif args.group == "baselines":
        keys = ["baseline_character", "baseline_char_bpe", "baseline_pure_byte"]
    elif args.group == "hangul":
        keys = ["full_pos_weighted_byte", "full_pos_unweighted_byte", "coarse_pos_weighted_byte", "coarse_pos_unweighted_byte"]
    else:
        keys = list(MODELS.keys())

    if args.parallel and len(keys) > 1:
        active = []
        pending = list(keys)
        completed = []

        while pending or active:
            while pending and len(active) < args.max_parallel:
                k = pending.pop(0)
                suffix_str = args.out_dir_suffix if args.out_dir_suffix else ""
                log_path = os.path.join(REPO_ROOT, f"train_20ep_{k}{suffix_str}.log")
                f_log = open(log_path, "a", encoding="utf-8")
                cmd = [sys.executable, __file__, "--models", k]
                if args.out_dir_suffix:
                    cmd.extend(["--out_dir_suffix", args.out_dir_suffix])
                print(f"Launching background worker for {k} (log: {log_path})...")
                p = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f_log, stderr=subprocess.STDOUT)
                active.append((k, p, f_log, log_path))

            time.sleep(2)
            still_active = []
            for k, p, f_log, log_path in active:
                ret = p.poll()
                if ret is not None:
                    f_log.close()
                    if ret != 0:
                        print(f"ERROR: Worker for {k} exited with code {ret}. Check {log_path}")
                        raise RuntimeError(f"Parallel worker for {k} failed with code {ret}")
                    else:
                        print(f"SUCCESS: Worker for {k} completed.")
                        completed.append(k)
                else:
                    still_active.append((k, p, f_log, log_path))
            active = still_active
    else:
        for k in keys:
            train_single_model(k, MODELS[k], out_dir_suffix=args.out_dir_suffix)

    print("\n==========================================================================")
    print(" ALL 20-EPOCH TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
