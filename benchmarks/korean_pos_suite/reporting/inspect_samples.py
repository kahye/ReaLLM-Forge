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

with open("results_seq2seq_ko_to_en_all_15_best_checkpoints.json", "r") as f:
    ko_en = json.load(f)

models_to_check = [
    "baseline_character",
    "baseline_char_bpe",
    "baseline_pure_byte",
    "mc_full_pos_weighted_byte",
    "mc_nochar_full_pos_new_weighted",
    "nochar_full_pos_charhead_h11",
]

for s_idx in [0, 1, 3, 4, 10, 14]:
    print("="*100)
    print(f"SAMPLE #{s_idx}")
    ko = ko_en["samples_metadata"][s_idx]["ko"]
    en = ko_en["samples_metadata"][s_idx]["en"]
    print(f"KO PROMPT: {ko}")
    print(f"EN REF   : {en}")
    print("-"*100)
    for mid in models_to_check:
        m = ko_en["models"][mid]
        rec = m["per_sample"][s_idx]
        print(f"[{mid:<32}]: {rec['response']}")
        print(f"    BLEU: {rec['bleu']:.2f} | chrF: {rec['chrf']:.2f} | BPC: {rec['bpc']:.4f}")
