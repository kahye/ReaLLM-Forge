#!/usr/bin/env python3
"""Exhaustive Loss Weight Search for 24-Lane Hangul Factorizer without Character Stream (No-Char).

Executes a structured grid search over 16 loss-weight configurations spanning:
- Tier 0: script_bpe (Byte-fallback BPE + <hangul>)
- Tier 1: Primary Jamo (choseong, jungseong, jongseong)
- Tier 2: Sub-character Jamo decomposition (jung_base1..2, jung_has_w/y/i, jong_base1..3)
- Tier 3: Articulatory & Phonological features (tense, aspirated, nasal_liquid, place, height, backness, round, complex, batchim)
- Tier 4: Modular arithmetic checksums (syllable_index_mod, codepoint_mod)
- Tier 5: Morphosyntactic POS tags (pos)

For each configuration, fine-tunes from the 10-epoch checkpoint for 300 iterations (4.9M tokens)
and evaluates exact validation cross-entropy loss across all 24 individual lanes.
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
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from gpt_conf import GPTConfig
from model import GPT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LANES = [
    "korean_pos_mc_nochar/script_bpe",            # 0: Tier 0 (BPE + Byte Fallback)
    "korean_pos_mc_nochar/choseong",              # 1: Tier 1 (Primary Jamo Onset)
    "korean_pos_mc_nochar/jungseong",             # 2: Tier 1 (Primary Jamo Nucleus)
    "korean_pos_mc_nochar/jongseong",             # 3: Tier 1 (Primary Jamo Coda)
    "korean_pos_mc_nochar/jung_base1",            # 4: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jung_base2",            # 5: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jung_has_w",            # 6: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jung_has_y",            # 7: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jung_has_i",            # 8: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jong_base1",            # 9: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jong_base2",            # 10: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/jong_base3",            # 11: Tier 2 (Sub-Jamo)
    "korean_pos_mc_nochar/choseong_tense",        # 12: Tier 3 (Phonological)
    "korean_pos_mc_nochar/choseong_aspirated",    # 13: Tier 3 (Phonological)
    "korean_pos_mc_nochar/choseong_nasal_liquid", # 14: Tier 3 (Phonological)
    "korean_pos_mc_nochar/choseong_place",        # 15: Tier 3 (Phonological)
    "korean_pos_mc_nochar/jung_height",           # 16: Tier 3 (Phonological)
    "korean_pos_mc_nochar/jung_backness",         # 17: Tier 3 (Phonological)
    "korean_pos_mc_nochar/jung_round",            # 18: Tier 3 (Phonological)
    "korean_pos_mc_nochar/jong_complex",          # 19: Tier 3 (Phonological)
    "korean_pos_mc_nochar/has_batchim",           # 20: Tier 3 (Phonological)
    "korean_pos_mc_nochar/syllable_index_mod",    # 21: Tier 4 (Modular Hash)
    "korean_pos_mc_nochar/codepoint_mod",         # 22: Tier 4 (Modular Hash)
    "korean_pos_mc_nochar/pos",                   # 23: Tier 5 (POS Tags)
]

LANE_SHORT_NAMES = [l.split("/")[-1] for l in LANES]


def build_weight_vector(
    w_bpe: float,
    w_jamo: float,
    w_subjamo: float,
    w_phon: float,
    w_mod: float,
    w_pos: float,
) -> List[float]:
    weights = []
    for idx, name in enumerate(LANE_SHORT_NAMES):
        if idx == 0:
            weights.append(w_bpe)
        elif idx in (1, 2, 3):
            weights.append(w_jamo)
        elif 4 <= idx <= 11:
            weights.append(w_subjamo)
        elif 12 <= idx <= 20:
            weights.append(w_phon)
        elif idx in (21, 22):
            weights.append(w_mod)
        elif idx == 23:
            weights.append(w_pos)
    return weights


# Define 16 search space configurations
SEARCH_CONFIGS = {
    # 1. Baselines
    "C01_uniform_1.0": {
        "desc": "Uniform Unweighted Baseline (all 24 lanes w=1.0)",
        "tiers": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    },
    "C02_legacy_heuristic_005": {
        "desc": "Legacy Heuristic (w_bpe=1.0, w_pos=0.5, all 22 struct/jamo=0.05)",
        "tiers": (1.0, 0.05, 0.05, 0.05, 0.05, 0.5),
    },
    # 2. Primary Jamo Weight Sweep (w_jamo in [0.10, 0.25, 0.50, 1.00, 1.50, 2.00])
    "C03_jamo_010": {
        "desc": "Primary Jamo Sweep: w_jamo=0.10 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.10, 0.05, 0.05, 0.05, 0.5),
    },
    "C04_jamo_025": {
        "desc": "Primary Jamo Sweep: w_jamo=0.25 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.25, 0.05, 0.05, 0.05, 0.5),
    },
    "C05_jamo_050": {
        "desc": "Primary Jamo Sweep: w_jamo=0.50 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.50, 0.05, 0.05, 0.05, 0.5),
    },
    "C06_jamo_100_parity": {
        "desc": "Primary Jamo Sweep: w_jamo=1.00 Parity (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5),
    },
    "C07_jamo_150_boost": {
        "desc": "Primary Jamo Sweep: w_jamo=1.50 Boost (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.50, 0.05, 0.05, 0.05, 0.5),
    },
    "C08_jamo_200_heavy": {
        "desc": "Primary Jamo Sweep: w_jamo=2.00 Heavy (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 2.00, 0.05, 0.05, 0.05, 0.5),
    },
    # 3. Auxiliary Structural & Phonological Tier Decomposition Sweep
    "C09_aux_zero_ablation": {
        "desc": "Zero Auxiliary Ablation (bpe=1.0, jamo=1.0, pos=0.5, aux=0.0)",
        "tiers": (1.0, 1.00, 0.00, 0.00, 0.00, 0.5),
    },
    "C10_hierarchical_decay": {
        "desc": "Hierarchical Tier Decay (bpe=1.0, jamo=1.0, subjamo=0.15, phon=0.08, mod=0.02, pos=0.5)",
        "tiers": (1.0, 1.00, 0.15, 0.08, 0.02, 0.5),
    },
    "C11_hierarchical_balanced": {
        "desc": "Hierarchical Balanced (bpe=1.0, jamo=0.85, subjamo=0.20, phon=0.10, mod=0.03, pos=0.4)",
        "tiers": (1.0, 0.85, 0.20, 0.10, 0.03, 0.4),
    },
    "C12_aux_moderate_025": {
        "desc": "Moderate Auxiliary Regularization (bpe=1.0, jamo=1.0, subjamo=0.25, phon=0.25, mod=0.10, pos=0.5)",
        "tiers": (1.0, 1.00, 0.25, 0.25, 0.10, 0.5),
    },
    # 4. POS Tag Supervision Sweep (with Hierarchical Decay base)
    "C13_pos_010_light": {
        "desc": "Light POS Supervision: w_pos=0.10 (bpe=1.0, jamo=1.0, subjamo=0.15, phon=0.08, mod=0.02)",
        "tiers": (1.0, 1.00, 0.15, 0.08, 0.02, 0.10),
    },
    "C14_pos_025_moderate": {
        "desc": "Moderate POS Supervision: w_pos=0.25 (bpe=1.0, jamo=1.0, subjamo=0.15, phon=0.08, mod=0.02)",
        "tiers": (1.0, 1.00, 0.15, 0.08, 0.02, 0.25),
    },
    "C15_pos_100_heavy": {
        "desc": "Heavy POS Supervision: w_pos=1.00 (bpe=1.0, jamo=1.0, subjamo=0.15, phon=0.08, mod=0.02)",
        "tiers": (1.0, 1.00, 0.15, 0.08, 0.02, 1.00),
    },
    # 5. Information-Theoretic Entropy-Proportional Weighting
    "C16_entropy_proportional": {
        "desc": "Entropy/Log-Vocab Proportional Weighting (scaled by lane log2(V_i))",
        "custom_weights": [
            1.00,  # script_bpe (V=4113, log2=12.0)
            0.75,  # choseong (V=22, log2=4.5)
            0.78,  # jungseong (V=24, log2=4.6)
            0.82,  # jongseong (V=31, log2=5.0)
            0.35,  # jung_base1
            0.20,  # jung_base2
            0.12,  # jung_has_w
            0.12,  # jung_has_y
            0.12,  # jung_has_i
            0.35,  # jong_base1
            0.15,  # jong_base2
            0.05,  # jong_base3
            0.15,  # choseong_tense
            0.15,  # choseong_aspirated
            0.18,  # choseong_nasal_liquid
            0.25,  # choseong_place
            0.22,  # jung_height
            0.22,  # jung_backness
            0.15,  # jung_round
            0.15,  # jong_complex
            0.15,  # has_batchim
            0.10,  # syllable_index_mod
            0.10,  # codepoint_mod
            0.45,  # pos (V=20)
        ],
    },
}


def evaluate_per_lane_val_loss(ckpt_path: str, eval_iters: int = 50) -> Dict[str, float]:
    """Evaluate exact validation cross-entropy loss for each of the 24 lanes."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = ckpt["model_args"]
    config = GPTConfig(**gptconf)
    model = GPT(config)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    val_data = [
        np.memmap(os.path.join(REPO_ROOT, "data", l, "val.bin"), dtype=np.uint16, mode="r")
        for l in LANES
    ]
    block_size = 256
    batch_size = 64

    losses = {i: [] for i in range(len(LANES))}
    torch.manual_seed(1337)
    with torch.no_grad():
        for _ in range(eval_iters):
            ix = torch.randint(len(val_data[0]) - block_size, (batch_size,))
            x_list, y_list = [], []
            for d in val_data:
                x = torch.stack([torch.from_numpy((d[i : i + block_size]).astype(np.int64)) for i in ix]).to(DEVICE)
                y = torch.stack([torch.from_numpy((d[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]).to(DEVICE)
                x_list.append(x)
                y_list.append(y)
            _, lane_losses = model(x_list, targets=y_list)
            for i, l in enumerate(lane_losses):
                losses[i].append(l.item())

    return {LANE_SHORT_NAMES[i]: float(np.mean(losses[i])) for i in range(len(LANES))}


def run_sweep():
    base_ckpt = os.path.join(REPO_ROOT, "out_mc_nochar_coarse_pos_unweighted", "5_epochs.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    ckpt_info = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    start_iter = ckpt_info.get("iter_num", 11311)
    target_iters = start_iter + 300  # 300 iterations (~4.9M tokens) fine-tuning per configuration

    results = {}
    out_root = "/tmp/nochar_loss_weight_sweep_v2"
    os.makedirs(out_root, exist_ok=True)

    for cfg_key, cfg in SEARCH_CONFIGS.items():
        if "custom_weights" in cfg:
            weights = cfg["custom_weights"]
        else:
            weights = build_weight_vector(*cfg["tiers"])

        run_dir = os.path.join(out_root, cfg_key)
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dest = os.path.join(run_dir, "ckpt.pt")
        ckpt_data = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        ckpt_data["best_val_loss"] = torch.tensor(float("inf"))
        torch.save(ckpt_data, ckpt_dest)

        print(f"\n>>> Running {cfg_key}: {cfg['desc']}")
        cmd = [
            sys.executable,
            "train.py",
            "--dataset",
            "korean_pos_mc_nochar/script_bpe",
            "--training_mode",
            "multicontext",
            "--multicontext",
            "--multicontext_datasets",
            *LANES,
            "--multicontext_loss_weights",
            *[str(w) for w in weights],
            "--batch_size",
            "64",
            "--block_size",
            "256",
            "--max_iters",
            str(target_iters),
            "--eval_interval",
            "300",
            "--eval_iters",
            "20",
            "--always_save_checkpoint",
            "--init_from",
            "resume",
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters",
            "45241",
            "--dropout",
            "0.1",
            "--device",
            DEVICE,
            "--out_dir",
            run_dir,
        ]

        t0 = time.time()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elapsed = time.time() - t0

        lane_losses = evaluate_per_lane_val_loss(os.path.join(run_dir, "ckpt.pt"), eval_iters=50)

        # Compute aggregate metrics
        recon_loss = (
            lane_losses["script_bpe"]
            + lane_losses["choseong"]
            + lane_losses["jungseong"]
            + lane_losses["jongseong"]
        )
        surface_pos_loss = recon_loss + lane_losses["pos"]
        mean_24_loss = float(np.mean(list(lane_losses.values())))
        weighted_loss = sum(w * lane_losses[name] for w, name in zip(weights, LANE_SHORT_NAMES)) / sum(weights)

        results[cfg_key] = {
            "description": cfg["desc"],
            "weights": dict(zip(LANE_SHORT_NAMES, weights)),
            "lane_losses": lane_losses,
            "metrics": {
                "recon_loss": recon_loss,
                "surface_pos_loss": surface_pos_loss,
                "mean_24_loss": mean_24_loss,
                "weighted_val_loss": weighted_loss,
                "script_bpe_loss": lane_losses["script_bpe"],
                "primary_jamo_sum": lane_losses["choseong"] + lane_losses["jungseong"] + lane_losses["jongseong"],
                "pos_loss": lane_losses["pos"],
            },
            "elapsed_sec": elapsed,
        }
        print(
            f"    Done in {elapsed:.1f}s | Recon Loss: {recon_loss:.4f} | "
            f"BPE: {lane_losses['script_bpe']:.4f} | JamoSum: {results[cfg_key]['metrics']['primary_jamo_sum']:.4f} | "
            f"POS: {lane_losses['pos']:.4f} | Mean24: {mean_24_loss:.4f}"
        )

        # Save incremental results
        with open(os.path.join(REPO_ROOT, "exhaustive_loss_weight_search_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\nAll 16 exhaustive search configurations completed!")


if __name__ == "__main__":
    run_sweep()
