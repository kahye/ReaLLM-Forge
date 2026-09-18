#!/usr/bin/env python3
"""Train Three-Hot Independent (Song et al., 2018) and Three-Hot Conditional (Cognetta et al., EACL 2023)
on the 59-Million Training Token Corpus (both Char/Byte Stream and No-Char BPE) up to 20 Epochs
using modern architecture and optimizer flags:
  --optimizer muon
  --weight_decay 0.0
  --use_qk_norm
  --use_qk_norm_scale
  --use_rotary_embeddings
  --no-use_abs_pos_embeddings

Saves milestone checkpoints at 3, 5, 10, 15, and 20 epochs, plus best_ckpt.pt tracking best validation loss.
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
import os
import shutil
import subprocess
import sys
import time
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50

EXPERIMENTS = {
    "three_hot_ind_char_byte": {
        "display_name": "Three-Hot Independent + Char/Byte Stream (Song 2018)",
        "out_dir": "./out_three_hot_ind_char_byte",
        "dataset": "korean_pos_mc/char_byte",
        "lanes": [
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ],
        "extra_args": [],
        "iters_per_epoch": 3633,
    },
    "three_hot_cond_char_byte": {
        "display_name": "Three-Hot Conditional + Char/Byte Stream (Cognetta 2023)",
        "out_dir": "./out_three_hot_cond_char_byte",
        "dataset": "korean_pos_mc/char_byte",
        "lanes": [
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ],
        "extra_args": ["--three_hot_conditional"],
        "iters_per_epoch": 3633,
    },
    "three_hot_ind_nochar": {
        "display_name": "Three-Hot Independent No-Char BPE (Song 2018)",
        "out_dir": "./out_three_hot_ind_nochar",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "lanes": [
            "korean_pos_mc_nochar/script_bpe",
            "korean_pos_mc_nochar/choseong",
            "korean_pos_mc_nochar/jungseong",
            "korean_pos_mc_nochar/jongseong",
        ],
        "extra_args": [],
        "iters_per_epoch": 2262,
    },
    "three_hot_cond_nochar": {
        "display_name": "Three-Hot Conditional No-Char BPE (Cognetta 2023)",
        "out_dir": "./out_three_hot_cond_nochar",
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "lanes": [
            "korean_pos_mc_nochar/script_bpe",
            "korean_pos_mc_nochar/choseong",
            "korean_pos_mc_nochar/jungseong",
            "korean_pos_mc_nochar/jongseong",
        ],
        "extra_args": ["--three_hot_conditional"],
        "iters_per_epoch": 2262,
    },
}


def train_single_model(exp_key: str, cfg: dict, out_dir_suffix: str = "_muon"):
    base_out_dir = cfg["out_dir"]
    out_dir = (base_out_dir + out_dir_suffix) if out_dir_suffix else base_out_dir
    os.makedirs(out_dir, exist_ok=True)
    ipe = cfg["iters_per_epoch"]
    max_iters_20ep = ipe * 20

    milestones = [
        ("3_epochs", ipe * 3, "3_epochs.pt"),
        ("5_epochs", ipe * 5, "5_epochs.pt"),
        ("10_epochs", ipe * 10, "10_epochs.pt"),
        ("15_epochs", ipe * 15, "15_epochs.pt"),
        ("20_epochs", ipe * 20, "20_epochs.pt"),
    ]

    print(f"\n==========================================================================")
    print(f" TRAINING: {exp_key} ({cfg['display_name']})")
    print(f" Output Dir: {out_dir}")
    print(f" Iterations/epoch: {ipe} | 20 Epochs = {max_iters_20ep:,} iterations")
    print(f"==========================================================================")

    prev_ckpt = os.path.join(out_dir, "ckpt.pt")

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
            "--dataset", cfg["dataset"],
            "--training_mode", "multicontext",
            "--multicontext",
            "--multicontext_datasets", *cfg["lanes"],
            *cfg["extra_args"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", str(max_iters_20ep),
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

        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {exp_key} at stage {stage_name} with exit code {res.returncode}")

        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{exp_key}] Saved milestone checkpoint: {stage_ckpt}")

        print(f"[{exp_key}] Stage {stage_name} finished in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Three-Hot models with Muon, QK-Norm, and RoPE.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()), default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--out_dir_suffix", type=str, default="_muon",
                        help="Suffix appended to out_dir (default: '_muon' to isolate Muon runs from AdamW checkpoints)")
    args = parser.parse_args()

    exp_keys = args.experiments or list(EXPERIMENTS.keys())
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
            train_single_model(key, EXPERIMENTS[key], out_dir_suffix=args.out_dir_suffix)

    print("\n==========================================================================")
    print(" ALL THREE-HOT 59M TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print("==========================================================================")
