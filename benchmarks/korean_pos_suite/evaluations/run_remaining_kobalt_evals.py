#!/usr/bin/env python3
"""Evaluate remaining baseline/factorizer models on KoBALT & CUTE-Hangul and merge into results_kobalt_cute_hangul_eval.json."""
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
import math
import os
import pickle
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from run_kobalt_cute_hangul_eval import (
    TASK_DATASETS,
    load_gpt_model,
    score_candidate_mc24,
    score_candidate_mc25,
)
from hangul_factorizer import HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_25_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
    tok_25_coarse = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")

    out_json = os.path.join(REPO_ROOT, "results_kobalt_cute_hangul_eval.json")
    with open(out_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    extra_models = {
        "mc_25_full_unweighted": {
            "name": "25-Lane Factorizer + Char Stream (Full POS, Unweighted)",
            "path": "out_mc_full_pos_unweighted_byte/best_ckpt.pt",
            "type": "mc_25",
            "pos_mode": "full",
        },
        "mc_25_coarse_weighted": {
            "name": "25-Lane Factorizer + Char Stream (Coarse POS, Weighted)",
            "path": "out_mc_coarse_pos_weighted_byte/best_ckpt.pt",
            "type": "mc_25",
            "pos_mode": "coarse",
        },
        "mc_25_coarse_unweighted": {
            "name": "25-Lane Factorizer + Char Stream (Coarse POS, Unweighted)",
            "path": "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt",
            "type": "mc_25",
            "pos_mode": "coarse",
        },
        "mc_24_nochar_full_unweighted": {
            "name": "24-Lane No-Char (Full POS, Unweighted)",
            "path": "out_mc_nochar_full_pos_unweighted/best_ckpt.pt",
            "type": "mc_24",
            "pos_mode": "full",
        },
        "mc_24_nochar_coarse_unweighted": {
            "name": "24-Lane No-Char (Coarse POS, Unweighted)",
            "path": "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt",
            "type": "mc_24",
            "pos_mode": "coarse",
        },
    }

    for model_id, mcfg in extra_models.items():
        if model_id in results:
            print(f"[Skip] {model_id} already in {out_json}")
            continue

        ckpt_path = os.path.join(REPO_ROOT, mcfg["path"])
        if not os.path.exists(ckpt_path):
            print(f"[Skip] Missing checkpoint: {ckpt_path}")
            continue

        print(f"Evaluating {mcfg['name']} ({model_id})...")
        model, ckpt_config = load_gpt_model(ckpt_path, device)
        m_type = mcfg["type"]
        pos_mode = mcfg["pos_mode"]

        stois_25 = []
        if m_type == "mc_25":
            lane_names = ckpt_config.get("multicontext_datasets", [])
            for lp in lane_names:
                with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
                    stois_25.append(pickle.load(f)["stoi"])

        model_scores = {}
        total_correct = 0
        total_items = 0

        for task_name, items in TASK_DATASETS.items():
            correct = 0
            for item in items:
                ctx = item["ctx"]
                cands = item["candidates"]
                ans = item["answer"]

                cand_scores = []
                for cand in cands:
                    if m_type == "mc_24":
                        tok_24 = tok_24_coarse if pos_mode == "coarse" else tok_24_full
                        s = score_candidate_mc24(model, tok_24, ctx, cand, device)
                    elif m_type == "mc_25":
                        tok_25 = tok_25_coarse if pos_mode == "coarse" else tok_25_full
                        s = score_candidate_mc25(model, tok_25, stois_25, ctx, cand, device)
                    else:
                        s = -math.inf
                    cand_scores.append(s)

                pred = int(max(range(len(cand_scores)), key=lambda i: cand_scores[i]))
                if pred == ans:
                    correct += 1

            acc = correct / max(1, len(items))
            model_scores[task_name] = {
                "accuracy": round(acc * 100.0, 2),
                "correct": correct,
                "total": len(items),
            }
            total_correct += correct
            total_items += len(items)

        overall_acc = total_correct / max(1, total_items)
        model_scores["overall_accuracy"] = round(overall_acc * 100.0, 2)
        print(f"  => Overall Accuracy: {overall_acc*100.0:.1f}%")
        results[model_id] = {
            "name": mcfg["name"],
            "scores": model_scores,
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Updated", out_json)


if __name__ == "__main__":
    main()
