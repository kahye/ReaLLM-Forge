#!/usr/bin/env python3
"""Update both AdamW and Muon Best Validation Loss reports with 1,000-example Extreme Noise Resilience (NSMC 80% & UnSmile 80%) results."""
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRAIN_AA3 = "/usr/local/google/home/kahye/.gemini/jetski/brain/aa3fd194-fa6f-447c-8cca-e9dd3112a744"
BRAIN_CUR = "/usr/local/google/home/kahye/.gemini/jetski/brain/b9392c59-2440-49a3-a8be-3f1713ac5f7e"

with open(os.path.join(REPO_ROOT, "results_noise_resilience_1000_all_models.json"), "r", encoding="utf-8") as f:
    noise_1000 = json.load(f)

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
    opt_data = noise_1000[opt_key]
    lines = md_text.splitlines()
    new_lines = []
    in_sec6 = False

    for line in lines:
        if line.startswith("## 6. Extreme Noise Resilience"):
            in_sec6 = True
            line = "## 6. Extreme Noise Resilience (`NSMC 80%` & `UnSmile 80%`, `N=1,000` Examples Each) Evaluation"
        elif line.startswith("## 7."):
            in_sec6 = False

        if in_sec6 and line.startswith("| **"):
            matched = None
            for label, m_id, mode_key in ROW_TO_MODEL:
                prefix = f"| **{label}** |"
                if line.startswith(prefix):
                    matched = (label, m_id, mode_key)
                    break

            if matched:
                label, m_id, mode_key = matched
                m_entry = opt_data[m_id][mode_key]
                nsmc_clean = m_entry["nsmc"]["clean_logprob"]
                nsmc_deg = m_entry["nsmc"]["degradation"]
                unsmile_clean = m_entry["unsmile"]["clean_logprob"]
                unsmile_deg = m_entry["unsmile"]["degradation"]

                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    parts[3] = f"`{nsmc_clean:.4f}`"
                    parts[4] = f"**`{nsmc_deg:.2f}`**"
                    parts[5] = f"`{unsmile_clean:.4f}`"
                    parts[6] = f"**`{unsmile_deg:.2f}`**"
                    line = "| " + " | ".join(parts[1:-1]) + " |"

        new_lines.append(line)

    return "\n".join(new_lines) + "\n"


def main():
    targets = [
        (os.path.join(REPO_ROOT, "report/adamw_best_validation_loss_checkpoints_comparison_report.md"), "adamw"),
        (os.path.join(REPO_ROOT, "report/muon_best_validation_loss_checkpoints_comparison_report.md"), "muon"),
        (os.path.join(REPO_ROOT, "updatereport/adamw_best_validation_loss_checkpoints_comparison_report.md"), "adamw"),
        (os.path.join(REPO_ROOT, "updatereport/muon_best_validation_loss_checkpoints_comparison_report.md"), "muon"),
        (os.path.join(BRAIN_AA3, "adamw_best_validation_loss_checkpoints_comparison_report.md"), "adamw"),
        (os.path.join(BRAIN_AA3, "muon_best_validation_loss_checkpoints_comparison_report.md"), "muon"),
        (os.path.join(BRAIN_CUR, "adamw_best_validation_loss_checkpoints_comparison_report.md"), "adamw"),
        (os.path.join(BRAIN_CUR, "muon_best_validation_loss_checkpoints_comparison_report.md"), "muon"),
    ]

    for path, opt_key in targets:
        if not os.path.exists(path):
            print(f"Skipping non-existent: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        new_txt = update_markdown_report(txt, opt_key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_txt)
        print(f"Updated ({opt_key}): {path}")


if __name__ == "__main__":
    main()
