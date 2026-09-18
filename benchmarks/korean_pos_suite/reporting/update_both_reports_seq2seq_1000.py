#!/usr/bin/env python3
"""Update both AdamW and Muon Best Validation Loss reports with 1,000-example Seq2Seq Translation Benchmark results."""
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
BRAIN_CUR = "/usr/local/google/home/kahye/.gemini/jetski/brain/4583b51e-b3c3-4937-a684-b9625cb44b6e"

with open(os.path.join(REPO_ROOT, "results_seq2seq_1000_all_models.json"), "r", encoding="utf-8") as f:
    s2s_1000 = json.load(f)

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
    opt_data = s2s_1000[opt_key]
    lines = md_text.splitlines()
    new_lines = []
    in_sec2 = False
    in_sec3 = False

    for line in lines:
        if line.startswith("## 2. Master Overview Table"):
            in_sec2 = True
            in_sec3 = False
        elif line.startswith("## 3. Detailed English-to-Korean Seq2Seq Translation Benchmark"):
            in_sec2 = False
            in_sec3 = True
            line = "## 3. Detailed English-to-Korean Seq2Seq Translation Benchmark (`N=1,000` Parallel Test Sentences)"
        elif line.startswith("## 4."):
            in_sec3 = False

        if (in_sec2 or in_sec3) and line.startswith("| **"):
            matched = None
            for label, m_id, mode_key in ROW_TO_MODEL:
                prefix = f"| **{label}** |"
                if line.startswith(prefix):
                    matched = (label, m_id, mode_key)
                    break

            if matched:
                label, m_id, mode_key = matched
                m_entry = opt_data[m_id][mode_key]
                bpj = m_entry["bpj"]
                t_loss = m_entry["target_loss"]
                t_ppl = m_entry["target_ppl"]
                bleu = m_entry["bleu"]
                chrf = m_entry["chrf"]
                lat = m_entry["latency_ms"]

                parts = [p.strip() for p in line.split("|")]
                if in_sec2 and len(parts) >= 12:
                    if "WITHOUT" in label:
                        parts[4] = f"**`{bpj:.4f}`**"
                    else:
                        parts[4] = f"`{bpj:.4f}`"
                    parts[5] = f"**`{chrf:.2f}`**"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                elif in_sec3 and len(parts) >= 9:
                    if "WITHOUT" in label:
                        parts[3] = f"**`{bpj:.4f}`**"
                        parts[4] = f"**`{t_loss:.4f}`**"
                        parts[5] = f"**`{t_ppl:.2f}`**"
                    else:
                        parts[3] = f"`{bpj:.4f}`"
                        parts[4] = f"`{t_loss:.4f}`"
                        parts[5] = f"`{t_ppl:.2f}`"
                    parts[6] = f"**`{bleu:.2f}`**"
                    parts[7] = f"**`{chrf:.2f}`**"
                    parts[8] = f"`{lat:.1f} ms`"
                    line = "| " + " | ".join(parts[1:-1]) + " |"

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
