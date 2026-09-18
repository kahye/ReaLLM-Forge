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

for mid, mdata in ko_en["models"].items():
    records = mdata["per_sample"]
    # Best & worst BPC (lower is better)
    best_bpc = min(records, key=lambda x: x["bpc"])
    worst_bpc = max(records, key=lambda x: x["bpc"])
    # Best chrF (higher is better)
    best_chrf = max(records, key=lambda x: x["chrf"])
    worst_chrf = min(records, key=lambda x: x["chrf"])
    # Best BLEU (higher is better)
    best_bleu = max(records, key=lambda x: x["bleu"])
    worst_bleu = min(records, key=lambda x: x["bleu"])
    
    print(f"[{mid}]")
    print(f"  Best BPC : #{best_bpc['sample_idx']} ({best_bpc['bpc']:.4f}) | Worst BPC : #{worst_bpc['sample_idx']} ({worst_bpc['bpc']:.4f})")
    print(f"  Best chrF: #{best_chrf['sample_idx']} ({best_chrf['chrf']:.2f}) | Worst chrF: #{worst_chrf['sample_idx']} ({worst_chrf['chrf']:.2f})")
    print(f"  Best BLEU: #{best_bleu['sample_idx']} ({best_bleu['bleu']:.2f}) | Worst BLEU: #{worst_bleu['sample_idx']} ({worst_bleu['bleu']:.2f})")
