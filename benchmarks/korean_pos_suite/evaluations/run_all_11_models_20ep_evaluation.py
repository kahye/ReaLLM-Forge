#!/usr/bin/env python3
"""Evaluate all saved checkpoints (10_epochs, 15_epochs, 20_epochs, best_ckpt) across all 11 models
on all downstream benchmarks:
1. Four Capabilities (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
2. Ko-HellaSwag Zero-Shot Reasoning (raw, length_norm, prior_norm, unigram_norm)
3. Phonetic Slang & Typo Resilience
4. Vocabulary Tail Perplexity (Deciles 1-10)
5. OOV / Unicode Stress Test
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
import subprocess
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_EXAMPLES = 100

ALL_MODELS = [
    # 1. Baselines
    {
        "id": "baseline_character",
        "name": "Baseline Character Tokenizer (V=4,924)",
        "out_dir": "out_baseline_korean_pos",
        "type": "baseline",
        "oov_id": "baseline_character",
    },
    {
        "id": "baseline_char_bpe",
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "out_dir": "out_baseline_char_bpe",
        "type": "baseline",
        "oov_id": "byte_fallback_bpe",
    },
    {
        "id": "baseline_pure_byte",
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "out_dir": "out_baseline_pure_byte",
        "type": "baseline",
        "oov_id": "pure_byte",
    },
    # 2. Hangul Factorizer with Character Stream and Byte Fallback (4 Combinations)
    {
        "id": "full_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "out_dir": "out_mc_full_pos_weighted_byte",
        "type": "byte_pos",
        "oov_id": "byte_fallback_full_pos_weighted",
    },
    {
        "id": "full_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "out_dir": "out_mc_full_pos_unweighted_byte",
        "type": "byte_pos",
        "oov_id": "byte_fallback_full_pos_unweighted",
    },
    {
        "id": "coarse_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "out_dir": "out_mc_coarse_pos_weighted_byte",
        "type": "byte_pos",
        "oov_id": "byte_fallback_coarse_pos_weighted",
    },
    {
        "id": "coarse_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "out_dir": "out_mc_coarse_pos_unweighted_byte",
        "type": "byte_pos",
        "oov_id": "byte_fallback_coarse_pos_unweighted",
    },
    # 3. Hangul Factorizer without Character Stream and Byte Fallback (4 Combinations)
    {
        "id": "nochar_full_pos_weighted",
        "name": "Hangul Factorizer No-Char (Full POS, Weighted)",
        "out_dir": "out_mc_nochar_full_pos_weighted",
        "type": "hybrid",
        "oov_id": "nochar_full_pos_weighted",
    },
    {
        "id": "nochar_full_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Full POS, Unweighted)",
        "out_dir": "out_mc_nochar_full_pos_unweighted",
        "type": "hybrid",
        "oov_id": "nochar_full_pos_unweighted",
    },
    {
        "id": "nochar_coarse_pos_weighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Weighted)",
        "out_dir": "out_mc_nochar_coarse_pos_weighted",
        "type": "hybrid",
        "oov_id": "nochar_coarse_pos_weighted",
    },
    {
        "id": "nochar_coarse_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Unweighted)",
        "out_dir": "out_mc_nochar_coarse_pos_unweighted",
        "type": "hybrid",
        "oov_id": "nochar_coarse_pos_unweighted",
    },
]

CHECKPOINT_STAGES = [
    ("10_epochs", "10_epochs.pt"),
    ("15_epochs", "15_epochs.pt"),
    ("20_epochs", "20_epochs.pt"),
    ("best_ckpt", "best_ckpt.pt"),
]


def eval_single_checkpoint(model_info: Dict[str, Any], stage_name: str, ckpt_path: str) -> Dict[str, Any]:
    res = {}
    try:
        d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        res["iter_num"] = int(d.get("iter_num", 0))
        val_loss = d.get("best_val_loss", 0.0)
        res["best_val_loss"] = float(val_loss.item() if hasattr(val_loss, "item") else val_loss)
    except Exception as e:
        print(f"    Warning reading metadata from {ckpt_path}: {e}")

    tag = f"{model_info['id']}_{stage_name}"

    # 1. Four Capabilities
    t4 = f"/tmp/eval_4cap_{tag}.json"
    cmd_4cap = [
        sys.executable, "benchmarks/run_four_capability_evals.py",
        "--ckpts", f"{tag}:{ckpt_path}",
        "--device", DEVICE,
        "--max_examples", str(MAX_EXAMPLES),
        "--output_json", t4,
    ]
    subprocess.run(cmd_4cap, cwd=REPO_ROOT, check=False)
    if os.path.exists(t4):
        with open(t4, "r", encoding="utf-8") as f:
            res["four_capabilities"] = json.load(f).get(tag, {})
        os.remove(t4)

    # 2. Ko-HellaSwag
    ths = f"/tmp/eval_hs_{tag}.json"
    cmd_hs = [
        sys.executable, "benchmarks/run_ko_hellaswag.py",
        "--ckpt_path", ckpt_path,
        "--device", DEVICE,
        "--eval_all_norms",
        "--max_examples", str(MAX_EXAMPLES),
        "--output_json", ths,
    ]
    subprocess.run(cmd_hs, cwd=REPO_ROOT, check=False)
    if os.path.exists(ths):
        with open(ths, "r", encoding="utf-8") as f:
            res["ko_hellaswag"] = json.load(f).get("accuracies", {})
        os.remove(ths)

    # 3. Phonetic Slang & Typo Resilience (run on best_ckpt and 20_epochs)
    if stage_name in ("best_ckpt", "20_epochs", "15_epochs"):
        tslang = f"/tmp/eval_slang_{tag}.json"
        is_mc = model_info["type"] in ("byte_pos", "hybrid")
        cmd_slang = [
            sys.executable, "benchmarks/run_phonetic_slang_eval.py",
            "--device", DEVICE,
            "--max_examples", str(MAX_EXAMPLES),
            "--output_json", tslang,
        ]
        cmd_slang.extend(["--mc_ckpt" if is_mc else "--base_ckpt", ckpt_path])
        subprocess.run(cmd_slang, cwd=REPO_ROOT, check=False)
        if os.path.exists(tslang):
            with open(tslang, "r", encoding="utf-8") as f:
                res["phonetic_slang"] = json.load(f)
            os.remove(tslang)

        # 4. Vocabulary Tail Perplexity
        ttail = f"/tmp/eval_tail_{tag}.json"
        cmd_tail = [
            sys.executable, "benchmarks/run_vocab_tail_perplexity.py",
            "--device", DEVICE,
            "--max_examples", "50",
            "--sample_syllables_per_decile", "50",
            "--output_json", ttail,
        ]
        cmd_tail.extend(["--mc_ckpt" if is_mc else "--base_ckpt", ckpt_path])
        subprocess.run(cmd_tail, cwd=REPO_ROOT, check=False)
        if os.path.exists(ttail):
            with open(ttail, "r", encoding="utf-8") as f:
                res["vocab_tail"] = json.load(f)
            os.remove(ttail)

    return res


def main():
    parser = argparse.ArgumentParser(description="Run full downstream evaluation across all 11 models & checkpoints.")
    parser.add_argument("--output_json", type=str, default="results_all_11_models_20ep_comprehensive.json")
    parser.add_argument("--models", nargs="+", default=None, help="Filter by model ID(s)")
    args = parser.parse_args()

    out_path = os.path.join(REPO_ROOT, args.output_json)
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    models_to_eval = [m for m in ALL_MODELS if not args.models or m["id"] in args.models]

    for m in models_to_eval:
        m_id = m["id"]
        if m_id not in all_results:
            all_results[m_id] = {
                "name": m["name"],
                "type": m["type"],
                "out_dir": m["out_dir"],
                "checkpoints": {},
            }

        for stage_name, ckpt_file in CHECKPOINT_STAGES:
            ckpt_path = os.path.join(REPO_ROOT, m["out_dir"], ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"[{m_id}] Checkpoint {ckpt_path} not found. Skipping {stage_name}.")
                continue

            if stage_name in all_results[m_id]["checkpoints"] and "four_capabilities" in all_results[m_id]["checkpoints"][stage_name]:
                print(f"[{m_id}] Stage {stage_name} already evaluated. Skipping.")
                continue

            print(f"\n>>> Evaluating {m['name']} @ {stage_name} ({ckpt_path})...")
            stage_res = eval_single_checkpoint(m, stage_name, ckpt_path)
            all_results[m_id]["checkpoints"][stage_name] = stage_res

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all 11 models 20-epoch evaluation results to {out_path}")


if __name__ == "__main__":
    main()
