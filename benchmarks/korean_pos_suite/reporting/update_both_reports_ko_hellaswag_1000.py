#!/usr/bin/env python3
"""Update both AdamW and Muon Best Validation Loss reports with 1,000-example Ko-HellaSwag results."""
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
import re
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRAIN_AA3 = "/usr/local/google/home/kahye/.gemini/jetski/brain/aa3fd194-fa6f-447c-8cca-e9dd3112a744"

with open(os.path.join(REPO_ROOT, "results_ko_hellaswag_1000_all_models.json"), "r", encoding="utf-8") as f:
    hs_1000 = json.load(f)

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
    opt_data = hs_1000[opt_key]
    lines = md_text.splitlines()
    new_lines = []
    in_sec2 = False
    in_sec42 = False

    for line in lines:
        if line.startswith("## 2. Master Overview Table"):
            in_sec2 = True
            in_sec42 = False
        elif line.startswith("## 3."):
            in_sec2 = False
        elif line.startswith("### 4.2. Zero-Shot Commonsense Reasoning"):
            in_sec42 = True
            in_sec2 = False
            line = "### 4.2. Zero-Shot Commonsense Reasoning (`Ko-HellaSwag` Across All 4 Normalizations, `N=1,000` Validation Examples)"
        elif line.startswith("## 5.") or line.startswith("### 4.3"):
            in_sec42 = False

        if (in_sec2 or in_sec42) and line.startswith("| **"):
            matched = None
            for label, m_id, mode_key in ROW_TO_MODEL:
                prefix = f"| **{label}** |"
                if line.startswith(prefix):
                    matched = (label, m_id, mode_key)
                    break

            if matched:
                label, m_id, mode_key = matched
                d = opt_data[m_id][mode_key]
                raw_str = f"**`{d['none']*100:.1f}%`**"
                len_str = f"**`{d['length']*100:.1f}%`**"
                prior_str = f"`{d['prior']*100:.1f}%`"
                unigram_str = f"`{d['unigram']*100:.1f}%`"

                parts = [p.strip() for p in line.split("|")]
                # parts[0] is empty before first |, parts[-1] is empty after last |
                if in_sec2 and len(parts) >= 12:
                    # Table 2 columns:
                    # 1: Model, 2: Ckpt, 3: Val Loss, 4: BPJ, 5: chrF, 6: NER PPL, 7: DP PPL, 8: HS Raw, 9: HS Len, 10: KorMed, 11: KoBALT
                    parts[8] = raw_str
                    parts[9] = len_str
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                elif in_sec42 and len(parts) >= 7:
                    # Table 4.2 columns:
                    # 1: Model, 2: Ckpt, 3: Raw Acc, 4: Length-Norm, 5: Prior-Norm, 6: Unigram-Norm
                    parts[3] = raw_str
                    parts[4] = len_str
                    parts[5] = prior_str
                    parts[6] = unigram_str
                    line = "| " + " | ".join(parts[1:-1]) + " |"

        # Also update strategic conclusion bullet if it mentions old 33.0% / 25.0%
        if opt_key == "adamw" and "strong commonsense reasoning (**`33.0%` raw / `25.0%` length-normalized Ko-HellaSwag**)" in line:
            line = line.replace(
                "(**`33.0%` raw / `25.0%` length-normalized Ko-HellaSwag**)",
                "(**`26.0%` length-normalized / `28.8%` prior-normalized Ko-HellaSwag on 1,000 examples**)"
            )

        new_lines.append(line)

    return "\n".join(new_lines) + "\n"


def main():
    for opt_key, filename in [
        ("adamw", "adamw_best_validation_loss_checkpoints_comparison_report.md"),
        ("muon", "muon_best_validation_loss_checkpoints_comparison_report.md"),
    ]:
        src_path = os.path.join(BRAIN_AA3, filename)
        with open(src_path, "r", encoding="utf-8") as f:
            orig_md = f.read()

        updated_md = update_markdown_report(orig_md, opt_key)

        target_paths = [
            src_path,
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
