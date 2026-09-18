#!/usr/bin/env python3
"""
Evaluate ALL 3,009 KorMedMCQA official test questions across all 34 models
(17 AdamW + 17 Muon best_ckpt).
Saves to results_kormedmcqa_all_3009_models.json.
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
import sys
import time
from typing import Any, Dict, List
import torch
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_kormedmcqa_1000_all_34_models import (
    ADAMW_MODELS,
    MUON_MODELS,
    build_tokenizer_map,
    pre_tokenize_kormed,
    evaluate_mcqa_batched,
    _load_checkpoint,
)


def load_all_kormedmcqa_test() -> List[Dict[str, Any]]:
    questions = []
    counts = {}
    for subj in ["doctor", "dentist", "nurse", "pharm"]:
        ds = load_dataset("sean0042/KorMedMCQA", subj, split="test")
        c = 0
        for row in ds:
            q = row.get("question", "")
            opts = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", ""), row.get("E", "")]
            opts = [o for o in opts if o]
            raw_ans = row.get("answer")
            if q and opts and raw_ans is not None:
                ans_idx = int(raw_ans) - 1
                if 0 <= ans_idx < len(opts):
                    questions.append({
                        "subject": subj,
                        "question": q,
                        "options": opts,
                        "ans_idx": ans_idx,
                    })
                    c += 1
        counts[subj] = c
    print(f"Loaded ALL {len(questions)} valid KorMedMCQA test questions across 4 subjects: {counts}")
    return questions


def main():
    out_path = os.path.join(REPO_ROOT, "results_kormedmcqa_all_3009_models.json")
    cache_path = "/tmp/kormed_all_3009_pretok_cache.pt"

    questions = load_all_kormedmcqa_test()
    tok_map = build_tokenizer_map()

    if os.path.exists(cache_path):
        print(f"Loading pre-tokenized KorMedMCQA cache from {cache_path}...")
        cache = torch.load(cache_path, weights_only=False)
    else:
        print(f"Pre-tokenizing {len(questions)} KorMedMCQA questions...")
        cache = pre_tokenize_kormed(questions, tok_map, block_size=256)
        torch.save(cache, cache_path)

    all_results = {"adamw": {}, "muon": {}, "num_examples": len(questions)}

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== EVALUATING {suite_name.upper()} ON ALL {len(questions)} KORMEDMCQA TEST QUESTIONS ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            t0 = time.time()
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            model = _load_checkpoint(ckpt_path, "cuda")
            pre_items = cache[tok_key]
            res = evaluate_mcqa_batched(model, pre_items, tok_key, device="cuda", batch_size=32)
            all_results[suite_name][m_id] = res
            dt = time.time() - t0
            for mode_name, m_dict in res.items():
                print(f"  [{suite_name.upper()}] {m_id:38s} ({mode_name:11s}) -> Acc: {m_dict['accuracy']*100:.2f}% ({dt:.1f}s)")
            del model
            torch.cuda.empty_cache()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full KorMedMCQA (N={len(questions)}) results to {out_path}")


if __name__ == "__main__":
    main()
