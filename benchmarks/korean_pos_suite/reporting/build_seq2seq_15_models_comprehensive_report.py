#!/usr/bin/env python3
"""Generates the Comprehensive English-to-Korean Seq2Seq Translation Benchmark Report for All 15 Models.

Reads `results_seq2seq_all_15_best_checkpoints_detailed.json` and writes:
1. Summary comparison table of all 15 models across BPJ, Target Loss, Target PPL, BLEU, Jamo chrF, and Latency.
2. Detailed model-by-model analysis for all 15 models showing the exact samples where each model achieves its
   Highest Score and Lowest Score on BPJ, Jamo chrF, and Sentence BLEU.
3. Complete Side-by-Side Translation Response Tables showing the outputs and metrics of all 15 models on
   every highlighted test sample for direct comparison.
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
from typing import Any, Dict, List, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_JSON = os.path.join(REPO_ROOT, "results_seq2seq_all_15_best_checkpoints_detailed.json")

MODEL_ORDER = [
    "baseline_character",
    "baseline_char_bpe",
    "baseline_pure_byte",
    "mc_full_pos_weighted_byte",
    "mc_full_pos_unweighted_byte",
    "mc_coarse_pos_weighted_byte",
    "mc_coarse_pos_unweighted_byte",
    "mc_nochar_full_pos_new_weighted",
    "mc_nochar_full_pos_unweighted",
    "mc_nochar_coarse_pos_new_weighted",
    "mc_nochar_coarse_pos_unweighted",
    "nochar_full_pos_charhead_h11",
    "nochar_full_pos_charhead_unweighted",
    "nochar_coarse_pos_charhead_h11",
    "nochar_coarse_pos_charhead_unweighted",
]


def build_markdown() -> str:
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data: Dict[str, Any] = data["models"]
    samples_meta: List[Dict[str, Any]] = data["samples_metadata"]

    lines: List[str] = []
    lines.append("# Comprehensive English-to-Korean Seq2Seq Translation Benchmark Report (All 15 Models)")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report evaluates the **best validation loss checkpoints (`best_ckpt.pt`)** across **all 15 trained models** "
        "spanning four architectural families on the **English-to-Korean Seq2Seq Translation Benchmark**:"
    )
    lines.append("1. **Monolithic Baselines (3 models)**: Character Tokenizer ($V=4,924$), Byte-Fallback BPE ($V=4,924$), Pure UTF-8 Byte ($V=256$).")
    lines.append("2. **25-Lane Hangul Factorizer + Char Stream (`mc_25`, 4 models)**: Full POS vs. Coarse POS $\\times$ Weighted ($w_{\\text{factors}}=0.05$) vs. Unweighted.")
    lines.append("3. **24-Lane No-Char Hangul Factorizer + Byte-Fallback BPE (`mc_24`, 4 models)**: Full POS vs. Coarse POS $\\times$ Newly Weighted ($w_{\\text{jamo}}=1.0$) vs. Unweighted.")
    lines.append("4. **Output-Only Target Lane Hangul Factorizer (`charhead`, 4 models)**: 24 input lanes + 25th output-only syllable head ($V=11,173$) $\\times$ Newly Weighted H11 ($w_{\\text{jamo}}=0.5, w_{\\text{char}}=0.5$) vs. Unweighted.")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Key Takeaways Across All 15 Models**")
    lines.append("> * **Latency Parity Achieved**: Following the $O(1)$ integer syllable recomposition and newline early-stopping fixes, all 24-lane no-char (`359–366 ms/sample`) and output-only target lane (`364–384 ms/sample`) factorizers match the generation speed of the 25-lane with-char models (`361–381 ms/sample`), eliminating the previous $2\\times$ latency gap.")
    lines.append("> * **Best Generative Translation Quality Among Factorizers**: The **Output-Only Target Lane Factorizer (`nochar_full_pos_charhead_h11`)** achieves the highest translation BLEU (**0.32**) and Compatibility-Jamo `chrF` (**5.02**) among all factorized architectures, outperforming standard 24-lane (`chrF 3.08`) and 25-lane (`chrF 3.75`) models while retaining $3\\times$ better information density (**0.5621 BPJ**) than Character/BPE baselines (**1.66–1.68 BPJ**).")
    lines.append("> * **Best Information Density (Teacher-Forced Compression)**: The **25-Lane With-Char (`mc_full_pos_unweighted_byte`: 0.4761 BPJ)** and **24-Lane No-Char (`mc_nochar_coarse_pos_new_weighted`: 0.5003 BPJ)** models achieve the lowest Bits Per Jamo across the 100-sentence test set.")
    lines.append("")

    # Section 1: Overall Corpus Comparison Table
    lines.append("---")
    lines.append("")
    lines.append("## 1. Overall Corpus-Level Benchmark Comparison (All 15 Best Checkpoints)")
    lines.append("")
    lines.append("| Model ID | Model Name | Family | Best Iter | BPJ (Bits/Jamo) $\\downarrow$ | Target Loss $\\downarrow$ | Target PPL $\\downarrow$ | Corpus BLEU $\\uparrow$ | Jamo `chrF` $\\uparrow$ | Latency (ms) $\\downarrow$ |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mid in MODEL_ORDER:
        if mid not in models_data:
            continue
        m = models_data[mid]
        cm = m["corpus_metrics"]
        lines.append(
            f"| `{mid}` | {m['name']} | {m['family']} | {m['iter_num']:,} | "
            f"**{cm['bpj']:.4f}** | {cm['target_loss']:.4f} | {cm['target_ppl']:.2f} | "
            f"{cm['bleu']:.2f} | **{cm['chrf']:.2f}** | {cm['latency_ms']:.1f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Collect highlighted sample indices across models
    highlighted_sample_indices: Set[int] = set()

    # Section 2: Per-Model High/Low Metric Analysis
    lines.append("## 2. Model-by-Model Metric Analysis (Best & Worst Examples per Metric)")
    lines.append("")
    lines.append(
        "For each of the 15 models, we identify the specific test sentences where that model achieves its **Best (Highest Quality)** "
        "and **Worst (Lowest Quality)** performance across **BPJ (Bits Per Jamo)**, **Compatibility-Jamo `chrF`**, and **Sentence BLEU**. "
        "Full 15-model side-by-side translation outputs for every referenced sample index (`Sample #0` to `Sample #19`) are provided in **Section 3**."
    )
    lines.append("")

    for idx_m, mid in enumerate(MODEL_ORDER, 1):
        if mid not in models_data:
            continue
        m = models_data[mid]
        cm = m["corpus_metrics"]
        per_sample: List[Dict[str, Any]] = m["per_sample"]

        # Best/Worst BPJ (Lower BPJ is better)
        best_bpj_item = min(per_sample, key=lambda x: x["bpj"])
        worst_bpj_item = max(per_sample, key=lambda x: x["bpj"])

        # Best/Worst chrF (Higher chrF is better)
        best_chrf_item = max(per_sample, key=lambda x: x["chrf"])
        worst_chrf_item = min(per_sample, key=lambda x: x["chrf"])

        # Best/Worst BLEU (Higher BLEU is better)
        best_bleu_item = max(per_sample, key=lambda x: x["bleu"])
        worst_bleu_item = min(per_sample, key=lambda x: x["bleu"])

        for item in (best_bpj_item, worst_bpj_item, best_chrf_item, worst_chrf_item, best_bleu_item, worst_bleu_item):
            highlighted_sample_indices.add(item["sample_idx"])

        lines.append(f"### 2.{idx_m} `{mid}` — {m['name']}")
        lines.append(f"- **Family**: {m['family']} | **Checkpoint**: `{m['ckpt_path']}` (Iter {m['iter_num']:,})")
        lines.append(
            f"- **Corpus Averages**: BPJ = `{cm['bpj']:.4f}` | Target PPL = `{cm['target_ppl']:.2f}` | "
            f"Corpus BLEU = `{cm['bleu']:.2f}` | Jamo `chrF` = `{cm['chrf']:.2f}` | Latency = `{cm['latency_ms']:.1f} ms`"
        )
        lines.append("")
        lines.append("| Metric | Best Score Sample (High Performance) | Worst Score Sample (Low Performance) |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(
            f"| **BPJ (Bits/Jamo)**<br>*(Lower is better)* | "
            f"**Sample #{best_bpj_item['sample_idx']}** (`BPJ = {best_bpj_item['bpj']:.4f}`, `PPL = {best_bpj_item['target_ppl']:.2f}`)<br>"
            f"• **EN**: *\"{best_bpj_item['en']}\"*<br>"
            f"• **Model Output**: `{best_bpj_item['response']}` | "
            f"**Sample #{worst_bpj_item['sample_idx']}** (`BPJ = {worst_bpj_item['bpj']:.4f}`, `PPL = {worst_bpj_item['target_ppl']:.2f}`)<br>"
            f"• **EN**: *\"{worst_bpj_item['en']}\"*<br>"
            f"• **Model Output**: `{worst_bpj_item['response']}` |"
        )
        lines.append(
            f"| **Jamo `chrF`**<br>*(Higher is better)* | "
            f"**Sample #{best_chrf_item['sample_idx']}** (`chrF = {best_chrf_item['chrf']:.2f}`, `BLEU = {best_chrf_item['bleu']:.2f}`)<br>"
            f"• **EN**: *\"{best_chrf_item['en']}\"*<br>"
            f"• **Model Output**: `{best_chrf_item['response']}` | "
            f"**Sample #{worst_chrf_item['sample_idx']}** (`chrF = {worst_chrf_item['chrf']:.2f}`, `BLEU = {worst_chrf_item['bleu']:.2f}`)<br>"
            f"• **EN**: *\"{worst_chrf_item['en']}\"*<br>"
            f"• **Model Output**: `{worst_chrf_item['response']}` |"
        )
        lines.append(
            f"| **Sentence BLEU**<br>*(Higher is better)* | "
            f"**Sample #{best_bleu_item['sample_idx']}** (`BLEU = {best_bleu_item['bleu']:.2f}`, `chrF = {best_bleu_item['chrf']:.2f}`)<br>"
            f"• **EN**: *\"{best_bleu_item['en']}\"*<br>"
            f"• **Model Output**: `{best_bleu_item['response']}` | "
            f"**Sample #{worst_bleu_item['sample_idx']}** (`BLEU = {worst_bleu_item['bleu']:.2f}`, `chrF = {worst_bleu_item['chrf']:.2f}`)<br>"
            f"• **EN**: *\"{worst_bleu_item['en']}\"*<br>"
            f"• **Model Output**: `{worst_bleu_item['response']}` |"
        )
        lines.append("")

    # Section 3: Side-by-Side Comparison Tables for All Highlighted Examples
    lines.append("---")
    lines.append("")
    lines.append("## 3. Side-by-Side Translation Responses from All 15 Models on Highlighted Benchmark Samples")
    lines.append("")
    lines.append(
        "To enable direct qualitative and quantitative comparison across all 15 models, the tables below present the exact generated "
        "Korean translation alongside sentence-level **BPJ**, **Sentence BLEU**, and **Jamo `chrF`** for every test sample highlighted in Section 2."
    )
    lines.append("")

    sorted_indices = sorted(list(highlighted_sample_indices))
    for s_idx in sorted_indices:
        s_meta = samples_meta[s_idx]
        lines.append(f"### Sample #{s_idx}")
        lines.append(f"- **Source English (`en`)**: `{s_meta['en']}`")
        lines.append(f"- **Reference Korean (`ko`)**: `{s_meta['ko']}`")
        lines.append("")
        lines.append("| Model ID | Family | Generated Korean Translation (`response`) | Sentence BPJ $\\downarrow$ | Sentence BLEU $\\uparrow$ | Jamo `chrF` $\\uparrow$ |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")

        for mid in MODEL_ORDER:
            if mid not in models_data:
                continue
            m = models_data[mid]
            rec = m["per_sample"][s_idx]
            resp_clean = rec["response"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{mid}` | {m['family']} | `{resp_clean}` | "
                f"{rec['bpj']:.4f} | {rec['bleu']:.2f} | **{rec['chrf']:.2f}** |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    report_md = build_markdown()
    out_workspace = os.path.join(REPO_ROOT, "report", "seq2seq_15_models_translation_benchmark_report.md")
    os.makedirs(os.path.dirname(out_workspace), exist_ok=True)
    with open(out_workspace, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved workspace report to: {out_workspace}")

    artifact_dir = "/usr/local/google/home/kahye/.gemini/jetski/brain/8b07e71e-c0ef-4ce6-8de5-063a8903dcd0"
    os.makedirs(artifact_dir, exist_ok=True)
    out_artifact = os.path.join(artifact_dir, "seq2seq_15_models_translation_benchmark_report.md")
    with open(out_artifact, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved artifact report to: {out_artifact}")


if __name__ == "__main__":
    main()
