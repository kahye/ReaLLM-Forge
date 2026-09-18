#!/usr/bin/env python3
"""Exhaustive Loss Weight Search for 25-Lane Output-Only Char Target Lane Hangul Factorizer + Byte-Fallback BPE.

Executes a structured grid search over 16 loss-weight configurations spanning 7 functional tiers:
- Tier 0: script_bpe (Byte-fallback BPE + <hangul>, Lane 0)
- Tier 1: Primary Jamo (choseong, jungseong, jongseong, Lanes 1..3)
- Tier 2: Sub-character Jamo decomposition (jung_base1..2, jung_has_w/y/i, jong_base1..3, Lanes 4..11)
- Tier 3: Articulatory & Phonological features (Lanes 12..20)
- Tier 4: Modular arithmetic checksums (syllable_index_mod, codepoint_mod, Lanes 21..22)
- Tier 5: Morphosyntactic POS tags (pos, Lane 23)
- Tier 6: Output-Only Character/Syllable LM Head (char, Lane 24, output_only_lanes=[24], V=11,173)

For each configuration, fine-tunes from the warm-started 25-lane checkpoint for 300 iterations
and evaluates exact validation cross-entropy loss across all 25 individual lanes.
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
    "korean_pos_mc_nochar/char",                  # 24: Tier 6 (Output-Only Char LM Head)
]

LANE_SHORT_NAMES = [l.split("/")[-1] for l in LANES]


def build_weight_vector(
    w_bpe: float,
    w_jamo: float,
    w_subjamo: float,
    w_phon: float,
    w_mod: float,
    w_pos: float,
    w_char: float,
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
        elif idx == 24:
            weights.append(w_char)
    return weights


# Define 16 structured search space configurations across the 7 functional tiers
SEARCH_CONFIGS = {
    # 1. Baselines
    "H01_uniform_1.0": {
        "desc": "Uniform Unweighted Baseline (all 25 lanes w=1.0)",
        "tiers": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    },
    "H02_legacy_char_dominant": {
        "desc": "Legacy Heuristic (w_bpe=1.0, w_char=1.0, w_pos=0.5, all 22 jamo/struct=0.05)",
        "tiers": (1.0, 0.05, 0.05, 0.05, 0.05, 0.5, 1.0),
    },
    # 2. Output-Only Char Head Weight Sweep (with Jamo Parity w_jamo=1.0, w_aux=0.05, w_pos=0.5)
    "H03_char_010_light": {
        "desc": "Char Head Sweep: w_char=0.10 Light Supervision (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5, 0.10),
    },
    "H04_char_025_mild": {
        "desc": "Char Head Sweep: w_char=0.25 Mild Supervision (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5, 0.25),
    },
    "H05_char_050_balanced": {
        "desc": "Char Head Sweep: w_char=0.50 Balanced Supervision (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5, 0.50),
    },
    "H06_char_100_parity": {
        "desc": "Char Head Sweep: w_char=1.00 Full Parity (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5, 1.00),
    },
    "H07_char_150_boost": {
        "desc": "Char Head Sweep: w_char=1.50 Strong Char Supervision (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.50, 0.05, 0.05, 0.05, 0.5, 1.50),
    },
    "H08_char_200_heavy": {
        "desc": "Char Head Sweep: w_char=2.00 Heavy Char Supervision (bpe=1.0, jamo=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 1.00, 0.05, 0.05, 0.05, 0.5, 2.00),
    },
    # 3. Primary Jamo Weight Sweep under Fixed Char Parity (w_char=1.00)
    "H09_jamo_025_char_100": {
        "desc": "Primary Jamo Sweep: w_jamo=0.25 with w_char=1.00 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.25, 0.05, 0.05, 0.05, 0.5, 1.00),
    },
    "H10_jamo_050_char_100": {
        "desc": "Primary Jamo Sweep: w_jamo=0.50 with w_char=1.00 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.50, 0.05, 0.05, 0.05, 0.5, 1.00),
    },
    "H11_jamo_050_char_050": {
        "desc": "Symmetric Dual-Target Balance: w_jamo=0.50, w_char=0.50 (bpe=1.0, pos=0.5, aux=0.05)",
        "tiers": (1.0, 0.50, 0.05, 0.05, 0.05, 0.5, 0.50),
    },
    # 4. Auxiliary Tier Decomposition & Ablation
    "H12_aux_zero_ablation": {
        "desc": "Zero Auxiliary Ablation: w_aux=0.00 (bpe=1.0, jamo=1.0, pos=0.5, char=1.0)",
        "tiers": (1.0, 1.00, 0.00, 0.00, 0.00, 0.5, 1.00),
    },
    "H13_hierarchical_decay": {
        "desc": "Hierarchical Tier Decay (bpe=1.0, jamo=1.0, subjamo=0.15, phon=0.08, mod=0.02, pos=0.5, char=1.0)",
        "tiers": (1.0, 1.00, 0.15, 0.08, 0.02, 0.5, 1.00),
    },
    "H14_hierarchical_char_075": {
        "desc": "Hierarchical Balanced + Char 0.75 (bpe=1.0, jamo=0.85, subjamo=0.15, phon=0.08, mod=0.02, pos=0.4, char=0.75)",
        "tiers": (1.0, 0.85, 0.15, 0.08, 0.02, 0.4, 0.75),
    },
    # 5. Information-Theoretic Entropy-Proportional & Synthesized Pareto Vectors
    "H15_entropy_proportional": {
        "desc": "Entropy/Log-Vocab Proportional Weighting (scaled by lane log2(V_i))",
        "custom_weights": [
            1.00,  # 0: script_bpe (V=4113)
            0.75,  # 1: choseong (V=22)
            0.78,  # 2: jungseong (V=24)
            0.82,  # 3: jongseong (V=31)
            0.30,  # 4: jung_base1
            0.18,  # 5: jung_base2
            0.10,  # 6: jung_has_w
            0.10,  # 7: jung_has_y
            0.10,  # 8: jung_has_i
            0.30,  # 9: jong_base1
            0.12,  # 10: jong_base2
            0.05,  # 11: jong_base3
            0.12,  # 12: choseong_tense
            0.12,  # 13: choseong_aspirated
            0.15,  # 14: choseong_nasal_liquid
            0.20,  # 15: choseong_place
            0.18,  # 16: jung_height
            0.18,  # 17: jung_backness
            0.12,  # 18: jung_round
            0.12,  # 19: jong_complex
            0.12,  # 20: has_batchim
            0.08,  # 21: syllable_index_mod
            0.08,  # 22: codepoint_mod
            0.45,  # 23: pos (V=20)
            1.12,  # 24: char (V=11173, log2=13.45)
        ],
    },
    "H16_pareto_synthesized": {
        "desc": "Synthesized Pareto Optimal (bpe=1.0, jamo=0.85, subjamo=0.10, phon=0.05, mod=0.02, pos=0.45, char=0.85)",
        "tiers": (1.0, 0.85, 0.10, 0.05, 0.02, 0.45, 0.85),
    },
}


def prepare_warmstart_25lane_checkpoint(src_24lane_ckpt: str, dst_25lane_ckpt: str) -> int:
    """Create a 25-lane checkpoint with output_only_lanes=[24] from the 24-lane checkpoint.

    Warm-starts `transformer.lm_head_24.weight` by composing the learned Jamo embeddings
    (`wte_1[c] + wte_2[v] + wte_3[f]`) for all 11,172 Hangul syllables so that `lm_head_24`
    begins fine-tuning aligned with the transformer's internal representation space.
    """
    ckpt = torch.load(src_24lane_ckpt, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    # Append vocab size 11173 for lane 24 (char) and mark lane 24 as output-only
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

    # Compose warm-start weight matrix for lm_head_24 (shape: 11173 x n_embd)
    wte_1 = state_dict["transformer.wte_1.weight"]  # choseong (22 x d)
    wte_2 = state_dict["transformer.wte_2.weight"]  # jungseong (24 x d)
    wte_3 = state_dict["transformer.wte_3.weight"]  # jongseong (31 x d)
    n_embd = wte_1.shape[1]

    lm_head_24_w = torch.zeros((11173, n_embd), dtype=wte_1.dtype)
    # Index 0 is <PAD>: initialize with mean of PAD embeddings
    lm_head_24_w[0] = (wte_1[0] + wte_2[0] + wte_3[0]) / 3.0

    # Indices 1..11172 correspond to Hangul syllables (c in 1..19, v in 1..21, f in 0..27)
    idx = 1
    for c in range(1, 20):
        for v in range(1, 22):
            for f in range(0, 28):
                lm_head_24_w[idx] = (wte_1[c] + wte_2[v] + wte_3[f]) / 3.0
                idx += 1

    state_dict["transformer.lm_head_24.weight"] = lm_head_24_w
    ckpt["best_val_loss"] = torch.tensor(float("inf"))
    # Clear optimizer state since parameter count changed
    if "optimizer" in ckpt:
        del ckpt["optimizer"]

    torch.save(ckpt, dst_25lane_ckpt)
    return ckpt.get("iter_num", 11311)


def evaluate_per_lane_val_loss(ckpt_path: str, eval_iters: int = 50) -> Dict[str, float]:
    """Evaluate exact validation cross-entropy loss for each of the 25 lanes."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = ckpt["model_args"]
    config = GPTConfig(**gptconf)
    model = GPT(config)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
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
    base_24lane_ckpt = os.path.join(REPO_ROOT, "out_mc_nochar_coarse_pos_unweighted", "5_epochs.pt")
    if not os.path.exists(base_24lane_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {base_24lane_ckpt}")

    out_root = "/tmp/charhead_loss_weight_sweep"
    os.makedirs(out_root, exist_ok=True)
    warm_25lane_base = os.path.join(out_root, "base_25lane_warm.pt")
    start_iter = prepare_warmstart_25lane_checkpoint(base_24lane_ckpt, warm_25lane_base)
    target_iters = start_iter + 300  # 300 iterations (~4.9M tokens) fine-tuning per configuration

    results = {}
    output_json_path = os.path.join(REPO_ROOT, "exhaustive_charhead_loss_weight_search_results.json")
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for cfg_key, cfg in SEARCH_CONFIGS.items():
        if cfg_key in results:
            print(f"Skipping {cfg_key} (already in results JSON).")
            continue

        if "custom_weights" in cfg:
            weights = cfg["custom_weights"]
        else:
            weights = build_weight_vector(*cfg["tiers"])

        run_dir = os.path.join(out_root, cfg_key)
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dest = os.path.join(run_dir, "ckpt.pt")
        shutil.copyfile(warm_25lane_base, ckpt_dest)

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
            "--output_only_datasets",
            "korean_pos_mc_nochar/char",
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
        jamo_recon_loss = (
            lane_losses["script_bpe"]
            + lane_losses["choseong"]
            + lane_losses["jungseong"]
            + lane_losses["jongseong"]
        )
        char_surface_loss = lane_losses["script_bpe"] + lane_losses["char"]
        hybrid_total_loss = char_surface_loss + (
            lane_losses["choseong"] + lane_losses["jungseong"] + lane_losses["jongseong"]
        ) + lane_losses["pos"]
        mean_25_loss = float(np.mean(list(lane_losses.values())))
        weighted_loss = sum(w * lane_losses[name] for w, name in zip(weights, LANE_SHORT_NAMES)) / sum(weights)

        results[cfg_key] = {
            "description": cfg["desc"],
            "weights": dict(zip(LANE_SHORT_NAMES, weights)),
            "lane_losses": lane_losses,
            "metrics": {
                "char_surface_loss": char_surface_loss,
                "jamo_recon_loss": jamo_recon_loss,
                "hybrid_total_loss": hybrid_total_loss,
                "mean_25_loss": mean_25_loss,
                "weighted_val_loss": weighted_loss,
                "script_bpe_loss": lane_losses["script_bpe"],
                "char_loss": lane_losses["char"],
                "primary_jamo_sum": lane_losses["choseong"] + lane_losses["jungseong"] + lane_losses["jongseong"],
                "pos_loss": lane_losses["pos"],
            },
            "elapsed_sec": elapsed,
        }
        print(
            f"    Done in {elapsed:.1f}s | CharSurface: {char_surface_loss:.4f} (Char: {lane_losses['char']:.4f}) | "
            f"JamoRecon: {jamo_recon_loss:.4f} | BPE: {lane_losses['script_bpe']:.4f} | Mean25: {mean_25_loss:.4f}"
        )

        # Save incremental results
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

    # Clean up temporary checkpoints in /tmp
    shutil.rmtree(out_root, ignore_errors=True)
    print("\nAll 16 exhaustive search configurations for output-only char head completed!")


if __name__ == "__main__":
    run_sweep()
