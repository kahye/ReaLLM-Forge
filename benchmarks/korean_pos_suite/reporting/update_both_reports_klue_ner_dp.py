#!/usr/bin/env python3
"""
Update both AdamW and Muon Best Validation Loss reports across all 8 locations
with stabilized 3-seed canonical KLUE-NER & KLUE-DP probing results (N_train=3,000, N_val=1,000)
and ensure all Section 2 columns (KLUE-NER, KLUE-DP, Ko-HellaSwag Raw/Len-Norm, KorMedMCQA, KoBALT)
are accurately aligned.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRAIN_AA3 = "/usr/local/google/home/kahye/.gemini/jetski/brain/aa3fd194-fa6f-447c-8cca-e9dd3112a744"
BRAIN_CUR = "/usr/local/google/home/kahye/.gemini/jetski/brain/b9392c59-2440-49a3-a8be-3f1713ac5f7e"

with open(os.path.join(REPO_ROOT, "results_klue_ner_dp_adamw_val_loss.json"), "r", encoding="utf-8") as f:
    ner_dp_adamw = {r["model_name"]: r for r in json.load(f)}

with open(os.path.join(REPO_ROOT, "results_klue_ner_dp_muon_val_loss.json"), "r", encoding="utf-8") as f:
    ner_dp_muon = {r["model_name"]: r for r in json.load(f)}

with open(os.path.join(REPO_ROOT, "results_ko_hellaswag_1000_all_models.json"), "r", encoding="utf-8") as f:
    hs_1000 = json.load(f)

with open(os.path.join(REPO_ROOT, "results_kormedmcqa_1000_all_models.json"), "r", encoding="utf-8") as f:
    km_1000 = json.load(f)

with open(os.path.join(REPO_ROOT, "results_kobalt_cute_1000_all_models.json"), "r", encoding="utf-8") as f:
    kb_1000 = json.load(f)

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


def lookup_probe_row(probe_dict: dict, label: str) -> dict:
    if label in probe_dict:
        return probe_dict[label]
    alt = label.replace("(WITHOUT `lm_head_24`)", "(WITHOUT `lm_head_24`, Greedy)")
    if alt in probe_dict:
        return probe_dict[alt]
    raise KeyError(f"Could not find probe row for {label}")


def update_markdown_report(md_text: str, opt_key: str) -> str:
    probe_dict = ner_dp_adamw if opt_key == "adamw" else ner_dp_muon
    hs_opt = hs_1000[opt_key]
    km_opt = km_1000[opt_key]
    kb_opt = kb_1000[opt_key]

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
        elif line.startswith("### 4.1."):
            in_sec41 = True
            in_sec2 = False
            line = "### 4.1. Sequence Labeling & Syntactic Parsing Probing (`KLUE-NER` F1 & `KLUE-DP` UAS/LAS, `N_train=3,000, N_val=1,000`) and Domain QA (`KorMedMCQA`, `N=1,000` Balanced Examples)"
        elif line.startswith("### 4.2.") or line.startswith("## 5."):
            in_sec41 = False

        if in_sec41 and line.startswith("> **Evaluation Methodology (Approach 1"):
            line = (
                "> **Evaluation Methodology (Approach 1 — Canonical Frozen-Backbone Compositional Probing):** "
                "Because `KLUE-NER` is a sequence labeling task and `KLUE-DP` is a graph-structured dependency parsing task, "
                "evaluating raw sentence perplexity conflates vocabulary size with structural accuracy. We evaluate canonical frozen-backbone "
                "contextual representations (`ln_f`), compositional factor embeddings, and explicit POS syntactic projections (`lane_23`) "
                "trained on `N_train=3,000` examples and evaluated on `N_val=1,000` validation examples using 3-seed averaging (`[42, 43, 44]`) "
                "with Cosine Annealing LR and gradient clipping: a 1-layer BiLSTM sequence tagger evaluated via strict **Entity-Level Macro F1 (%)** "
                "and **Character-Level Macro F1 (%)** across all 13 BIO classes on `klue/ner`, and a **Biaffine Dependency Parser (Dozat & Manning, 2017)** "
                "over eojeol representations evaluated via **Unlabeled Attachment Score (UAS %)** and **Labeled Attachment Score (LAS %)** across all 37 syntactic relations on `klue/dp`. "
                "Because `WITH lm_head_24`, `WITHOUT lm_head_24, Greedy`, and `WITHOUT lm_head_24, Joint Beam Search` share the exact same frozen checkpoint backbone, they share identical canonical probing scores."
            )

        if (in_sec2 or in_sec41) and line.startswith("| **"):
            matched = None
            for label, m_id, mode_key in ROW_TO_MODEL:
                prefix = f"| **{label}** |"
                if line.startswith(prefix):
                    matched = (label, m_id, mode_key)
                    break

            if matched:
                label, m_id, mode_key = matched
                p_row = lookup_probe_row(probe_dict, label)
                ent_f1 = p_row["ner_entity_f1"]
                char_f1 = p_row["ner_char_f1"]
                uas = p_row["dp_uas"]
                las = p_row["dp_las"]

                hs_raw = hs_opt[m_id][mode_key]["none"] * 100.0
                hs_len = hs_opt[m_id][mode_key]["length"] * 100.0
                km_acc = km_opt[m_id][mode_key]["accuracy"] * 100.0
                kb_acc = kb_opt[m_id][mode_key]["overall_accuracy"]

                parts = [p.strip() for p in line.split("|")]
                if in_sec2 and len(parts) >= 14:
                    # Table 2 columns:
                    # 1: Model, 2: Ckpt, 3: Val Loss, 4: BPJ, 5: chrF,
                    # 6: NER Ent F1, 7: NER Char F1, 8: DP UAS, 9: DP LAS,
                    # 10: HS Raw, 11: HS Len-Norm, 12: KorMedMCQA, 13: KoBALT & CUTE
                    parts[6] = f"`{ent_f1:.2f}%`"
                    parts[7] = f"`{char_f1:.2f}%`"
                    parts[8] = f"`{uas:.2f}%`"
                    parts[9] = f"`{las:.2f}%`"
                    parts[10] = f"**`{hs_raw:.1f}%`**"
                    parts[11] = f"**`{hs_len:.1f}%`**"
                    parts[12] = f"**`{km_acc:.1f}%`**"
                    parts[13] = f"**`{kb_acc:.2f}%`**"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                elif in_sec41 and len(parts) >= 8:
                    # Table 4.1 columns:
                    # 1: Model, 2: Ckpt, 3: NER Ent F1, 4: NER Char F1, 5: DP UAS, 6: DP LAS, 7: KorMedMCQA Acc
                    parts[3] = f"`{ent_f1:.2f}%`"
                    parts[4] = f"`{char_f1:.2f}%`"
                    parts[5] = f"`{uas:.2f}%`"
                    parts[6] = f"`{las:.2f}%`"
                    parts[7] = f"**`{km_acc:.1f}%`**"
                    line = "| " + " | ".join(parts[1:-1]) + " |"

        new_lines.append(line)

    return "\n".join(new_lines) + "\n"


def main():
    report_targets = [
        ("adamw", [
            os.path.join(REPO_ROOT, "report/adamw_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(REPO_ROOT, "updatereport/adamw_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(BRAIN_AA3, "adamw_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(BRAIN_CUR, "adamw_best_validation_loss_checkpoints_comparison_report.md"),
        ]),
        ("muon", [
            os.path.join(REPO_ROOT, "report/muon_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(REPO_ROOT, "updatereport/muon_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(BRAIN_AA3, "muon_best_validation_loss_checkpoints_comparison_report.md"),
            os.path.join(BRAIN_CUR, "muon_best_validation_loss_checkpoints_comparison_report.md"),
        ]),
    ]

    for opt_key, paths in report_targets:
        src_path = paths[0]
        with open(src_path, "r", encoding="utf-8") as f:
            updated_md = update_markdown_report(f.read(), opt_key)
        for p in paths:
            if os.path.exists(os.path.dirname(p)):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(updated_md)
                print(f"Updated {p}")


if __name__ == "__main__":
    main()
