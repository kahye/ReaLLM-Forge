#!/usr/bin/env python3
"""Update both AdamW and Muon Best Validation Loss reports with 1,000-example KorMedMCQA results."""
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRAIN_AA3 = "/usr/local/google/home/kahye/.gemini/jetski/brain/aa3fd194-fa6f-447c-8cca-e9dd3112a744"
BRAIN_CUR = "/usr/local/google/home/kahye/.gemini/jetski/brain/b9392c59-2440-49a3-a8be-3f1713ac5f7e"

with open(os.path.join(REPO_ROOT, "results_kormedmcqa_1000_all_models.json"), "r", encoding="utf-8") as f:
    km_1000 = json.load(f)

ROW_TO_MODEL = [
    ("Baseline Character Tokenizer (V=4,924)", "baseline_character", "primary"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "baseline_char_bpe", "primary"),
    ("Baseline Pure Byte Tokenizer (V=256)", "baseline_pure_byte", "primary"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "three_hot_ind_char_byte", "primary"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "three_hot_cond_char_byte", "primary"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "mc_full_pos_weighted_byte", "primary"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "mc_full_pos_unweighted_byte", "primary"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "mc_coarse_pos_weighted_byte", "primary"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "mc_coarse_pos_unweighted_byte", "primary"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "nochar_full_pos_new_weighted", "primary"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "nochar_full_pos_unweighted", "primary"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "nochar_coarse_pos_new_weighted", "primary"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "nochar_coarse_pos_unweighted", "primary"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITH `lm_head_24`)", "nochar_full_pos_charhead_h11", "charhead_24"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT `lm_head_24`, Greedy)", "nochar_full_pos_charhead_h11", "primary"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT `lm_head_24`)", "nochar_full_pos_charhead_h11", "primary"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT `lm_head_24`, Joint Beam Search)", "nochar_full_pos_charhead_h11", "jamo_lvt"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITH `lm_head_24`)", "nochar_full_pos_charhead_unweighted", "charhead_24"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT `lm_head_24`, Greedy)", "nochar_full_pos_charhead_unweighted", "primary"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT `lm_head_24`)", "nochar_full_pos_charhead_unweighted", "primary"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT `lm_head_24`, Joint Beam Search)", "nochar_full_pos_charhead_unweighted", "jamo_lvt"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITH `lm_head_24`)", "nochar_coarse_pos_charhead_h11", "charhead_24"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT `lm_head_24`, Greedy)", "nochar_coarse_pos_charhead_h11", "primary"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT `lm_head_24`)", "nochar_coarse_pos_charhead_h11", "primary"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT `lm_head_24`, Joint Beam Search)", "nochar_coarse_pos_charhead_h11", "jamo_lvt"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITH `lm_head_24`)", "nochar_coarse_pos_charhead_unweighted", "charhead_24"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT `lm_head_24`, Greedy)", "nochar_coarse_pos_charhead_unweighted", "primary"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT `lm_head_24`)", "nochar_coarse_pos_charhead_unweighted", "primary"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT `lm_head_24`, Joint Beam Search)", "nochar_coarse_pos_charhead_unweighted", "jamo_lvt"),
]


def update_markdown_report(md_text: str, opt_key: str) -> str:
    opt_data = km_1000[opt_key]
    lines = md_text.splitlines()
    new_lines = []
    in_sec2 = False
    in_sec41 = False

    for line in lines:
        if line.startswith("## 2. Master Overview Table"):
            in_sec2 = True
            in_sec41 = False
        elif line.startswith("## 3."):
            in_sec2 = False
        elif line.startswith("### 4.1. Syntactic & Entity Perplexity"):
            in_sec41 = True
            in_sec2 = False
            line = "### 4.1. Syntactic & Entity Perplexity (`KLUE-NER` & `KLUE-DP`) and Domain QA (`KorMedMCQA`, `N=1,000` Balanced Examples)"
        elif line.startswith("### 4.2.") or line.startswith("## 5."):
            in_sec41 = False

        if (in_sec2 or in_sec41) and line.startswith("| **"):
            matched = None
            for label, m_id, mode_key in ROW_TO_MODEL:
                prefix = f"| **{label}** |"
                if line.startswith(prefix):
                    matched = (label, m_id, mode_key)
                    break

            if matched:
                label, m_id, mode_key = matched
                acc = opt_data[m_id][mode_key]["accuracy"]
                km_str = f"**`{acc*100:.1f}%`**"

                parts = [p.strip() for p in line.split("|")]
                if in_sec2 and len(parts) >= 12:
                    # Table 2 columns:
                    # 1: Model, 2: Ckpt, 3: Val Loss, 4: BPJ, 5: chrF, 6: NER PPL, 7: DP PPL, 8: HS Raw, 9: HS Len, 10: KorMed, 11: KoBALT
                    parts[10] = km_str
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                elif in_sec41 and len(parts) >= 8:
                    # Table 4.1 columns:
                    # 1: Model, 2: Ckpt, 3: NER LogProb, 4: NER PPL, 5: DP LogProb, 6: DP PPL, 7: KorMedMCQA Acc
                    parts[7] = km_str
                    line = "| " + " | ".join(parts[1:-1]) + " |"

        if opt_key == "adamw" and "reaches peak medical QA domain performance (**`24.0%` KorMedMCQA**, higher than the 22.0% at 15_epochs)" in line:
            line = line.replace(
                "reaches peak medical QA domain performance (**`24.0%` KorMedMCQA**, higher than the 22.0% at 15_epochs)",
                "achieves strong medical QA domain performance (**`21.2%` KorMedMCQA on 1,000 balanced examples**, while `mc_full_pos_unweighted_byte` reaches **`22.7%`** and `nochar_full_pos_charhead_h11` reaches **`22.5%`**)"
            )

        new_lines.append(line)

    return "\n".join(new_lines) + "\n"


def main():
    for opt_key, filename in [
        ("adamw", "adamw_best_validation_loss_checkpoints_comparison_report.md"),
        ("muon", "muon_best_validation_loss_checkpoints_comparison_report.md"),
    ]:
        src_path = os.path.join(REPO_ROOT, "report", filename)
        with open(src_path, "r", encoding="utf-8") as f:
            orig_md = f.read()

        updated_md = update_markdown_report(orig_md, opt_key)

        target_paths = [
            os.path.join(BRAIN_AA3, filename),
            os.path.join(BRAIN_CUR, filename),
            os.path.join(REPO_ROOT, "report", filename),
            os.path.join(REPO_ROOT, "updatereport", filename),
        ]
        for tp in target_paths:
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            with open(tp, "w", encoding="utf-8") as f:
                f.write(updated_md)
            print(f"Updated {tp}")


if __name__ == "__main__":
    main()
