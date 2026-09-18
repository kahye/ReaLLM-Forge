#!/usr/bin/env python3
"""Comprehensive Evaluation Suite for Best Validation Checkpoints across All 11 Models.

Evaluates:
1. Baseline character tokenizer (V=4,924)
2. Baseline bytefallback BPE (V=4,924)
3. Baseline pure byte tokenizer (V=256)
4. Hangul factorizer with character stream and bytefallback (Full POS, Weighted)
5. Hangul factorizer with character stream and bytefallback (Full POS, Unweighted)
6. Hangul factorizer with character stream and bytefallback (Coarse POS, Weighted)
7. Hangul factorizer with character stream and bytefallback (Coarse POS, Unweighted)
8. Hangul factorizer without character stream and bytefallback BPE (Full POS, Weighted)
9. Hangul factorizer without character stream and bytefallback BPE (Full POS, Unweighted)
10. Hangul factorizer without character stream and bytefallback BPE (Coarse POS, Weighted)
11. Hangul factorizer without character stream and bytefallback BPE (Coarse POS, Unweighted)

Downstream Benchmarks:
1. Four Capabilities: KLUE-NER, KLUE-DP, NSMC (80% noise), UnSmile (80% noise), KorMedMCQA
2. Ko-HellaSwag Zero-Shot Reasoning (Raw, Length-Norm, Prior-Norm, Unigram-Norm)
3. Phonetic Slang & Typo Resilience (Clean vs 80% Corrupted)
4. Vocabulary Tail Perplexity (Deciles 1 to 10)
5. OOV & Unicode Stress-Test Prompt Evaluation
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
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EVAL_TARGETS = [
    # 1. Baseline Models
    {
        "id": "baseline_character",
        "name": "Baseline Character Tokenizer (V=4,924)",
        "ckpt": "out_baseline_korean_pos/best_ckpt.pt",
        "fallback_ckpt": "out_baseline_korean_pos/ckpt.pt",
        "type": "baseline",
        "oov_id": "baseline_character",
    },
    {
        "id": "baseline_char_bpe",
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "ckpt": "out_baseline_char_bpe/best_ckpt.pt",
        "fallback_ckpt": "out_baseline_char_bpe/ckpt.pt",
        "type": "baseline",
        "oov_id": "byte_fallback_bpe",
    },
    {
        "id": "baseline_pure_byte",
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "ckpt": "out_baseline_pure_byte/best_ckpt.pt",
        "fallback_ckpt": "out_baseline_pure_byte/ckpt.pt",
        "type": "baseline",
        "oov_id": "pure_byte",
    },

    # 2. Hangul Factorizer with Character Stream and Byte Fallback (4 Combinations)
    {
        "id": "byte_fallback_full_pos_weighted",
        "name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "ckpt": "out_mc_full_pos_weighted_byte/best_ckpt.pt",
        "fallback_ckpt": "out_mc_full_pos_weighted_byte/ckpt.pt",
        "type": "byte_pos",
        "oov_id": "byte_fallback_full_pos_weighted",
    },
    {
        "id": "byte_fallback_full_pos_unweighted",
        "name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "ckpt": "out_mc_full_pos_unweighted_byte/best_ckpt.pt",
        "fallback_ckpt": "out_mc_full_pos_unweighted_byte/ckpt.pt",
        "type": "byte_pos",
        "oov_id": "byte_fallback_full_pos_unweighted",
    },
    {
        "id": "byte_fallback_coarse_pos_weighted",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "ckpt": "out_mc_coarse_pos_weighted_byte/best_ckpt.pt",
        "fallback_ckpt": "out_mc_coarse_pos_weighted_byte/ckpt.pt",
        "type": "byte_pos",
        "oov_id": "byte_fallback_coarse_pos_weighted",
    },
    {
        "id": "byte_fallback_coarse_pos_unweighted",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "ckpt": "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt",
        "fallback_ckpt": "out_mc_coarse_pos_unweighted_byte/ckpt.pt",
        "type": "byte_pos",
        "oov_id": "byte_fallback_coarse_pos_unweighted",
    },

    # 3. Hangul Factorizer without Character Stream and Byte Fallback BPE (4 Combinations)
    {
        "id": "nochar_full_pos_weighted",
        "name": "Hangul Factorizer No-Char (Full POS, Weighted)",
        "ckpt": "out_mc_nochar_full_pos_weighted/best_ckpt.pt",
        "fallback_ckpt": "out_mc_nochar_full_pos_weighted/ckpt.pt",
        "type": "hybrid",
        "oov_id": "nochar_full_pos_weighted",
    },
    {
        "id": "nochar_full_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Full POS, Unweighted)",
        "ckpt": "out_mc_nochar_full_pos_unweighted/best_ckpt.pt",
        "fallback_ckpt": "out_mc_nochar_full_pos_unweighted/ckpt.pt",
        "type": "hybrid",
        "oov_id": "nochar_full_pos_unweighted",
    },
    {
        "id": "nochar_coarse_pos_weighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Weighted)",
        "ckpt": "out_mc_nochar_coarse_pos_weighted/best_ckpt.pt",
        "fallback_ckpt": "out_mc_nochar_coarse_pos_weighted/ckpt.pt",
        "type": "hybrid",
        "oov_id": "nochar_coarse_pos_weighted",
    },
    {
        "id": "nochar_coarse_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Unweighted)",
        "ckpt": "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt",
        "fallback_ckpt": "out_mc_nochar_coarse_pos_unweighted/ckpt.pt",
        "type": "hybrid",
        "oov_id": "nochar_coarse_pos_unweighted",
    },
]


def resolve_ckpt(target: Dict[str, Any]) -> str | None:
    if os.path.exists(target["ckpt"]):
        return target["ckpt"]
    fallback = target.get("fallback_ckpt")
    if fallback and os.path.exists(fallback):
        return fallback
    return None


def run_benchmark_evaluations(
    max_examples: int = 100,
    output_file: str = "eval_results_best_checkpoints.json",
    targets_filter: List[str] | None = None,
):
    results = {}

    targets = [t for t in EVAL_TARGETS if not targets_filter or t["id"] in targets_filter]

    print("==========================================================================")
    print(" STARTING BEST CHECKPOINTS DOWNSTREAM EVALUATION SUITE")
    print(f" Targets: {len(targets)} models")
    print(f" Max Examples per benchmark: {max_examples}")
    print(f" Device: {DEVICE}")
    print("==========================================================================")

    for idx, target in enumerate(targets, 1):
        target_id = target["id"]
        target_name = target["name"]
        ckpt_path = resolve_ckpt(target)

        if not ckpt_path:
            print(f"[{idx}/{len(targets)}] Checkpoint not found for {target_name}. Skipping.")
            continue

        print(f"\n[{idx}/{len(targets)}] Evaluating: {target_name} ({ckpt_path})")
        target_res = {
            "name": target_name,
            "ckpt": ckpt_path,
            "type": target["type"],
        }

        # 1. Four Capability Evaluations (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
        print("  -> Running Four Capabilities...")
        temp_4cap = f"/tmp/eval_4cap_{target_id}.json"
        cmd_4cap = [
            sys.executable, "benchmarks/run_four_capability_evals.py",
            "--ckpts", f"{target_id}:{ckpt_path}",
            "--device", DEVICE,
            "--max_examples", str(max_examples),
            "--output_json", temp_4cap,
        ]
        try:
            subprocess.run(cmd_4cap, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_4cap):
                with open(temp_4cap, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["four_capabilities"] = data.get(target_id, {})
                os.remove(temp_4cap)
        except Exception as e:
            print(f"     Failed Four Capabilities: {e}")

        # 2. Ko-HellaSwag Zero-Shot Reasoning
        print("  -> Running Ko-HellaSwag Zero-Shot Reasoning...")
        temp_hs = f"/tmp/eval_hs_{target_id}.json"
        cmd_hs = [
            sys.executable, "benchmarks/run_ko_hellaswag.py",
            "--ckpt_path", ckpt_path,
            "--device", DEVICE,
            "--eval_all_norms",
            "--max_examples", str(max_examples),
            "--output_json", temp_hs,
        ]
        try:
            subprocess.run(cmd_hs, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_hs):
                with open(temp_hs, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["ko_hellaswag"] = data.get("accuracies", {})
                os.remove(temp_hs)
        except Exception as e:
            print(f"     Failed Ko-HellaSwag: {e}")

        # 3. Phonetic Slang & Typo Resilience (Clean vs 80% Corrupted)
        print("  -> Running Phonetic Slang & Typo Resilience...")
        temp_slang = f"/tmp/eval_slang_{target_id}.json"
        is_mc = target["type"] in ["hybrid", "byte_pos"]
        cmd_slang = [
            sys.executable, "benchmarks/run_phonetic_slang_eval.py",
            "--device", DEVICE,
            "--max_examples", str(max_examples),
            "--output_json", temp_slang,
        ]
        if is_mc:
            cmd_slang.extend(["--mc_ckpt", ckpt_path])
        else:
            cmd_slang.extend(["--base_ckpt", ckpt_path])

        try:
            subprocess.run(cmd_slang, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_slang):
                with open(temp_slang, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["phonetic_slang"] = data
                os.remove(temp_slang)
        except Exception as e:
            print(f"     Failed Phonetic Slang: {e}")

        # 4. Vocabulary Tail Perplexity
        print("  -> Running Vocabulary Tail Perplexity...")
        temp_tail = f"/tmp/eval_tail_{target_id}.json"
        cmd_tail = [
            sys.executable, "benchmarks/run_vocab_tail_perplexity.py",
            "--device", DEVICE,
            "--max_examples", str(min(max_examples, 50)),
            "--sample_syllables_per_decile", "50",
            "--output_json", temp_tail,
        ]
        if is_mc:
            cmd_tail.extend(["--mc_ckpt", ckpt_path])
        else:
            cmd_tail.extend(["--base_ckpt", ckpt_path])

        try:
            subprocess.run(cmd_tail, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_tail):
                with open(temp_tail, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["vocab_tail"] = data
                os.remove(temp_tail)
        except Exception as e:
            print(f"     Failed Vocab Tail: {e}")

        results[target_id] = target_res

        # Incremental save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. OOV & Unicode Stress-Test across all 11 models
    print("\n  -> Running OOV & Unicode Stress-Test across all models...")
    oov_target_ids = [t["oov_id"] for t in targets]
    temp_oov = f"/tmp/eval_oov_all.json"
    cmd_oov = [
        sys.executable, "benchmarks/run_oov_evaluations.py",
        "--models", *oov_target_ids,
        "--device", DEVICE,
        "--max_new_tokens", "25",
        "--output_json", temp_oov,
    ]
    try:
        subprocess.run(cmd_oov, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_oov):
            with open(temp_oov, "r", encoding="utf-8") as f:
                oov_data = json.load(f)
            results["oov_evaluations"] = oov_data.get("model_evaluations", {})
            os.remove(temp_oov)
    except Exception as e:
        print(f"     Failed OOV evaluation: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================================================")
    print(f" ALL EVALUATIONS COMPLETED! Results saved to {output_file}")
    print(f"==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete downstream evaluation suite on best checkpoints.")
    parser.add_argument("--max_examples", type=int, default=100, help="Max examples per benchmark")
    parser.add_argument("--output_file", type=str, default="eval_results_best_checkpoints.json", help="Output JSON path")
    parser.add_argument("--targets", nargs="+", default=None, help="Specific target model IDs to run")
    args = parser.parse_args()

    run_benchmark_evaluations(max_examples=args.max_examples, output_file=args.output_file, targets_filter=args.targets)
