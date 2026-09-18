#!/usr/bin/env python3
"""Evaluate the 4 Three-Hot Tokenizer 59M 20ep models on Ko-HellaSwag, Phonetic Slang, Vocabulary Tail, and Seq2Seq Translation."""
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
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS = [
    {
        "id": "three_hot_ind_char_byte",
        "name": "Three-Hot Ind Char-Byte (Song 2018)",
        "ckpt": "out_three_hot_ind_char_byte/best_ckpt.pt",
        "seq2seq_type": "mc_25",
    },
    {
        "id": "three_hot_cond_char_byte",
        "name": "Three-Hot Cond Char-Byte (Cognetta 2023)",
        "ckpt": "out_three_hot_cond_char_byte/best_ckpt.pt",
        "seq2seq_type": "mc_25",
    },
    {
        "id": "three_hot_ind_nochar",
        "name": "Three-Hot Ind No-Char BPE (Song 2018)",
        "ckpt": "out_three_hot_ind_nochar/best_ckpt.pt",
        "seq2seq_type": "mc_24",
    },
    {
        "id": "three_hot_cond_nochar",
        "name": "Three-Hot Cond No-Char BPE (Cognetta 2023)",
        "ckpt": "out_three_hot_cond_nochar/best_ckpt.pt",
        "seq2seq_type": "mc_24",
    },
]

def main():
    # Load 4-capability results
    with open("eval_results_three_hot_four_capabilities.json", "r", encoding="utf-8") as f:
        four_cap_data = json.load(f)

    master_results = {}
    for m in MODELS:
        mid = m["id"]
        ckpt = m["ckpt"]
        print(f"\n====================================================")
        print(f" Evaluating {m['name']} ({ckpt})")
        print(f"====================================================")

        res = {
            "name": m["name"],
            "ckpt": ckpt,
            "four_capabilities": four_cap_data.get(ckpt, four_cap_data.get(mid, {})),
        }

        # 1. Ko-HellaSwag
        hs_json = f"/tmp/hs_{mid}.json"
        cmd_hs = [
            sys.executable, "benchmarks/run_ko_hellaswag.py",
            "--ckpt_path", ckpt,
            "--device", "cuda",
            "--eval_all_norms",
            "--max_examples", "100",
            "--output_json", hs_json,
        ]
        print(" -> Running Ko-HellaSwag...")
        subprocess.run(cmd_hs, cwd=REPO_ROOT, check=True)
        with open(hs_json, "r", encoding="utf-8") as f:
            res["ko_hellaswag"] = json.load(f).get("accuracies", {})

        # 2. Phonetic Slang
        slang_json = f"/tmp/slang_{mid}.json"
        cmd_slang = [
            sys.executable, "benchmarks/run_phonetic_slang_eval.py",
            "--mc_ckpt", ckpt,
            "--device", "cuda",
            "--max_examples", "100",
            "--output_json", slang_json,
        ]
        print(" -> Running Phonetic Slang Resilience...")
        subprocess.run(cmd_slang, cwd=REPO_ROOT, check=True)
        with open(slang_json, "r", encoding="utf-8") as f:
            res["phonetic_slang"] = json.load(f)

        # 3. Vocabulary Tail Perplexity
        tail_json = f"/tmp/tail_{mid}.json"
        cmd_tail = [
            sys.executable, "benchmarks/run_vocab_tail_perplexity.py",
            "--mc_ckpt", ckpt,
            "--device", "cuda",
            "--max_examples", "50",
            "--sample_syllables_per_decile", "50",
            "--output_json", tail_json,
        ]
        print(" -> Running Vocabulary Tail Perplexity...")
        subprocess.run(cmd_tail, cwd=REPO_ROOT, check=True)
        with open(tail_json, "r", encoding="utf-8") as f:
            res["vocab_tail"] = json.load(f)

        master_results[mid] = res

    with open("eval_results_three_hot_59m_all.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2, ensure_ascii=False)
    print("\nSaved eval_results_three_hot_59m_all.json!")

if __name__ == "__main__":
    main()
