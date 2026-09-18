#!/usr/bin/env python3
"""Evaluate 3, 5, 10, 15, 20 Epoch and Best Checkpoints for No-Char Hybrid POS models on 4 Capabilities and Ko-HellaSwag."""
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
DEVICE = "cuda:0"
MAX_EXAMPLES = 100

MODELS = {
    "nochar_full_pos_weighted": "out_mc_nochar_full_pos_weighted",
    "nochar_full_pos_unweighted": "out_mc_nochar_full_pos_unweighted",
    "nochar_coarse_pos_weighted": "out_mc_nochar_coarse_pos_weighted",
    "nochar_coarse_pos_unweighted": "out_mc_nochar_coarse_pos_unweighted",
}

EPOCHS = [
    ("3_epoch", "3_epochs.pt"),
    ("5_epoch", "5_epochs.pt"),
    ("10_epoch", "10_epochs.pt"),
    ("15_epoch", "15_epochs.pt"),
    ("20_epoch", "20_epochs.pt"),
    ("best", "best_ckpt.pt"),
]

def main():
    results = {}
    for m_id, out_dir in MODELS.items():
        results[m_id] = {}
        for ep_name, ckpt_file in EPOCHS:
            ckpt_path = os.path.join(out_dir, ckpt_file)
            if not os.path.exists(ckpt_path):
                continue
            print(f"Evaluating {m_id} @ {ep_name} ({ckpt_path})...")
            res_entry = {}
            t4 = f"/tmp/temp_4cap_{m_id}_{ep_name}.json"
            subprocess.run([
                sys.executable, "benchmarks/run_four_capability_evals.py",
                "--ckpts", f"{ep_name}:{ckpt_path}",
                "--device", DEVICE,
                "--max_examples", str(MAX_EXAMPLES),
                "--output_json", t4
            ], cwd=REPO_ROOT, check=True)
            if os.path.exists(t4):
                with open(t4) as f:
                    res_entry["four_capabilities"] = json.load(f).get(ep_name, {})
                os.remove(t4)

            ths = f"/tmp/temp_hs_{m_id}_{ep_name}.json"
            subprocess.run([
                sys.executable, "benchmarks/run_ko_hellaswag.py",
                "--ckpt_path", ckpt_path,
                "--device", DEVICE,
                "--eval_all_norms",
                "--max_examples", str(MAX_EXAMPLES),
                "--output_json", ths
            ], cwd=REPO_ROOT, check=True)
            if os.path.exists(ths):
                with open(ths) as f:
                    res_entry["ko_hellaswag"] = json.load(f).get("accuracies", {})
                os.remove(ths)

            results[m_id][ep_name] = res_entry
            with open("results_nochar_epoch_progression.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved progression results to results_nochar_epoch_progression.json")

if __name__ == "__main__":
    main()
