#!/usr/bin/env python3
"""Generate comprehensive comparative report for Output-Only Target Lane Hangul Factorizer (H11) vs all best checkpoints."""
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


def load_json(filename):
    path = os.path.join(REPO_ROOT, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    charhead_metrics = load_json("results_charhead_h11_20ep_all_metrics.json")
    all_11_best = load_json("results_all_11_models_best_checkpoints.json")
    new_nochar_epochs = load_json("results_new_weighted_nochar_all_epochs.json")
    kobalt_all = load_json("results_kobalt_cute_hangul_eval.json")
    seq2seq_old = load_json("results_seq2seq_translation_all_checkpoints.json")
    seq2seq_new_nochar = load_json("results_new_weighted_nochar_seq2seq.json")

    # Best validation losses for each model
    best_val_losses = {
        "baseline_character": (1.4720, "best_ckpt.pt"),
        "baseline_char_bpe": (1.4712, "best_ckpt.pt"),
        "baseline_pure_byte": (1.1328, "best_ckpt.pt"),
        "mc_full_pos_weighted": (0.8653, "best_ckpt.pt"),
        "mc_full_pos_unweighted": (1.1585, "best_ckpt.pt"),
        "mc_coarse_pos_weighted": (0.8524, "best_ckpt.pt"),
        "mc_coarse_pos_unweighted": (1.1472, "best_ckpt.pt"),
        "mc_nochar_full_pos_unweighted": (1.0312, "best_ckpt.pt"),
        "mc_nochar_coarse_pos_unweighted": (1.0189, "best_ckpt.pt"),
        "mc_nochar_full_pos_new_weighted": (0.8112, "best_ckpt.pt"),
        "mc_nochar_coarse_pos_new_weighted": (0.7984, "best_ckpt.pt"),
        "nochar_full_pos_charhead_h11": (0.7757, "best_ckpt.pt (15_epochs.pt)"),
        "nochar_coarse_pos_charhead_h11": (0.7635, "best_ckpt.pt (15_epochs.pt)"),
    }

    # (logical_id, display_name, kobalt_key, s2s_key, all11_key)
    models_spec = [
        # 1. Baseline Tokenizers
        ("baseline_character", "Baseline Character Tokenizer (`V=4,924`)", "baseline_character", "baseline_character", "baseline_character"),
        ("baseline_char_bpe", "Baseline Byte-Fallback BPE (`V=4,924`)", "baseline_char_bpe", "baseline_char_bpe", "baseline_char_bpe"),
        ("baseline_pure_byte", "Baseline Pure Byte (`V=256`)", "baseline_pure_byte", "baseline_pure_byte", "baseline_pure_byte"),
        # 2. 25-Lane Hangul Factorizer WITH Input Character Stream + Byte-Fallback
        ("mc_full_pos_weighted", "25-Lane + Char Input (Full POS, Weighted)", "mc_25_full_weighted", "mc_full_pos_weighted_byte", "byte_fallback_full_pos_weighted"),
        ("mc_full_pos_unweighted", "25-Lane + Char Input (Full POS, Unweighted)", "mc_25_full_unweighted", "mc_full_pos_unweighted_byte", "byte_fallback_full_pos_unweighted"),
        ("mc_coarse_pos_weighted", "25-Lane + Char Input (Coarse POS, Weighted)", "mc_25_coarse_weighted", "mc_coarse_pos_weighted_byte", "byte_fallback_coarse_pos_weighted"),
        ("mc_coarse_pos_unweighted", "25-Lane + Char Input (Coarse POS, Unweighted)", "mc_25_coarse_unweighted", "mc_coarse_pos_unweighted_byte", "byte_fallback_coarse_pos_unweighted"),
        # 3. 24-Lane Hangul Factorizer WITHOUT Character Stream + Byte-Fallback (Unweighted)
        ("mc_nochar_full_pos_unweighted", "24-Lane No-Char (Full POS, Unweighted)", "mc_24_nochar_full_unweighted", "mc_nochar_full_pos_unweighted", "nochar_full_pos_unweighted"),
        ("mc_nochar_coarse_pos_unweighted", "24-Lane No-Char (Coarse POS, Unweighted)", "mc_24_nochar_coarse_unweighted", "mc_nochar_coarse_pos_unweighted", "nochar_coarse_pos_unweighted"),
        # 4. 24-Lane Hangul Factorizer WITHOUT Character Stream + Byte-Fallback (Newly Weighted)
        ("mc_nochar_full_pos_new_weighted", "24-Lane No-Char (Full POS, Newly Weighted)", "mc_24_nochar_full_new_weighted", "nochar_full_pos_new_weighted", "nochar_full_pos_new_weighted"),
        ("mc_nochar_coarse_pos_new_weighted", "24-Lane No-Char (Coarse POS, Newly Weighted)", "mc_24_nochar_coarse_new_weighted", "nochar_coarse_pos_new_weighted", "nochar_coarse_pos_new_weighted"),
    ]

    lines = []
    lines.append("# Comprehensive Downstream Evaluation Report: Output-Only Target Lane Hangul Factorizer (`H11_jamo_050_char_050`) vs. All Baseline & Multi-Lane Models")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Key Takeaway:** By removing the 11,173-entry Hangul syllable table from the **input embedding (`wte_char`)** while preserving an **output-only character prediction head (`output_only_lanes=[24]`)** trained with symmetric dual-target supervision (`H11_jamo_050_char_050`: $w_{\\text{bpe}}=1.0, w_{\\text{jamo}}=0.5, w_{\\text{char}}=0.5, w_{\\text{pos}}=0.5, w_{\\text{aux}}=0.05$), the **Output-Only Target Lane Hangul Factorizer** achieves:")
    lines.append("> - **Lowest Overall Validation Loss:** **`0.7635`** (Coarse POS) and **`0.7757`** (Full POS), outperforming both the 24-lane no-char models (`0.7984`) and the 25-lane input-char models (`0.8524`).")
    lines.append("> - **Strong KoBALT & CUTE-Hangul Diagnostic Accuracy (`44.44%`):** Ties the highest overall diagnostic score among all weighted models while achieving **`50.0%` on CUTE Jamo Surgery** and **`50.0%` on Choseong Quiz**.")
    lines.append("> - **Zero Input Vocabulary Bloat:** Saves **4.29M input embedding parameters** (`wte_char` eliminated) while retaining full syllable composition supervision at the output LM head.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Architectural & Loss Weight Summary")
    lines.append("")
    lines.append("| Model Family | Input Embeddings | Output LM Heads | Total Params | Loss Weights ($w_{\\text{bpe}}, w_{\\text{jamo}}, w_{\\text{char}}, w_{\\text{pos}}, w_{\\text{aux}}$) |")
    lines.append("| :--- | :--- | :--- | :---: | :--- |")
    lines.append("| **25-Lane + Input Char Stream** | 25 lanes (includes `wte_char`: $11,173 \\times 384$) | 25 LM heads | `20.90M` | Weighted: `(1.0, 1.0, 1.0, 0.5, 0.05)` / Unweighted: all `1.0` |")
    lines.append("| **24-Lane No-Char (Input/Output)** | 24 lanes (no `wte_char`) | 24 LM heads | `12.31M` | Newly Weighted: `(1.0, 1.0, N/A, 0.5, 0.05)` / Unweighted: all `1.0` |")
    lines.append("| **25-Lane Output-Only Char Head (`H11`)** | **24 lanes (NO `wte_char` in input)** | **25 LM heads (`output_only_lanes=[24]`)** | **`16.61M`** | **Winning `H11`: `(1.0, 0.5, 0.5, 0.5, 0.05)`** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Master Comparison at Best Validation Loss Checkpoint (`best_ckpt.pt`)")
    lines.append("")
    lines.append("### 2.1 Validation Loss, Seq2Seq Translation & KoBALT Diagnostic Suite")
    lines.append("")
    lines.append("| Model Name | Best Val Loss | Seq2Seq BPJ $\\downarrow$ | Target PPL $\\downarrow$ | BLEU $\\uparrow$ | chrF $\\uparrow$ | KoBALT Choseong | Allomorphic Particles | Assimilation Rules | Vowel Harmony | CUTE Jamo Surgery | **KoBALT Overall $\\uparrow$** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    def get_s2s_best(s2s_key):
        if s2s_key in seq2seq_old:
            return seq2seq_old[s2s_key]["checkpoints"].get("best", seq2seq_old[s2s_key]["checkpoints"].get("15_epoch", {}))
        if s2s_key in seq2seq_new_nochar:
            return seq2seq_new_nochar[s2s_key].get("best", seq2seq_new_nochar[s2s_key].get("15_epoch", {}))
        return {}

    def format_master_row(display_name, val_loss, s2s, kobalt_scores, is_highlight=False):
        bpj = s2s.get("bpj", 0.0)
        ppl = s2s.get("target_ppl", 0.0)
        bleu = s2s.get("bleu", 0.0)
        chrf = s2s.get("chrf", 0.0)
        c_quiz = kobalt_scores.get("choseong_quiz", {}).get("accuracy", 0.0)
        allo = kobalt_scores.get("allomorphic_particles", {}).get("accuracy", 0.0)
        assim = kobalt_scores.get("phonological_assimilation", {}).get("accuracy", 0.0)
        vowel = kobalt_scores.get("vowel_harmony_ideophones", {}).get("accuracy", 0.0)
        jamo = kobalt_scores.get("cute_jamo_manipulation", {}).get("accuracy", 0.0)
        overall = kobalt_scores.get("overall_accuracy", 0.0)
        name_col = f"**{display_name}**" if is_highlight else display_name
        return (
            f"| {name_col} | `{val_loss:.4f}` | `{bpj:.3f}` | `{ppl:.2f}` | `{bleu:.2f}` | `{chrf:.2f}` | "
            f"`{c_quiz:.1f}%` | `{allo:.1f}%` | `{assim:.1f}%` | `{vowel:.1f}%` | `{jamo:.1f}%` | **`{overall:.2f}%`** |"
        )

    for mid, disp, kkey, s2s_key, _ in models_spec:
        val_loss, _ = best_val_losses[mid]
        s2s = get_s2s_best(s2s_key)
        kscores = kobalt_all.get(kkey, {}).get("scores", {})
        lines.append(format_master_row(disp, val_loss, s2s, kscores, is_highlight=False))

    for h11_id, h11_disp in [
        ("nochar_full_pos_charhead_h11", "25-Lane Output-Only Char Head (Full POS, Winning H11)"),
        ("nochar_coarse_pos_charhead_h11", "25-Lane Output-Only Char Head (Coarse POS, Winning H11)"),
    ]:
        val_loss, _ = best_val_losses[h11_id]
        best_entry = charhead_metrics[h11_id]["best"]
        s2s = best_entry["seq2seq"]
        kscores = best_entry["kobalt_cute_hangul"]
        lines.append(format_master_row(h11_disp, val_loss, s2s, kscores, is_highlight=True))

    lines.append("")
    lines.append("### 2.2 Four Capabilities Benchmark & Ko-HellaSwag Zero-Shot Reasoning")
    lines.append("")
    lines.append("| Model Name | KLUE-NER PPL $\\downarrow$ | KLUE-DP PPL $\\downarrow$ | NSMC OOV Noise $\\Delta$ | UnSmile OOV Noise $\\Delta$ | KorMedMCQA Acc | Ko-HellaSwag Acc |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mid, disp, _, _, all11_key in models_spec:
        if all11_key in all_11_best:
            entry = all_11_best[all11_key]
            fc = entry["four_capabilities"]
            ner_ppl = fc["klue_ner_ppl"]
            dp_ppl = fc["klue_dp_ppl"]
            nsmc_deg = fc["nsmc_degradation"]
            uns_deg = fc["unsmile_degradation"]
            mcqa = fc["kormedqa_acc"] * 100.0
            hs = entry["ko_hellaswag"]["length_norm"] * 100.0
        else:
            entry = new_nochar_epochs[all11_key]["best"]
            fc = entry["four_capabilities"]
            ner_ppl = fc["klue_ner"]["perplexity"]
            dp_ppl = fc["klue_dp"]["perplexity"]
            nsmc_deg = fc["noisy_text"]["nsmc"]["degradation"]
            uns_deg = fc["noisy_text"]["unsmile"]["degradation"]
            mcqa = fc["kormedqa"]["accuracy"] * 100.0
            hs = entry["ko_hellaswag"]["length"] * 100.0
        lines.append(
            f"| {disp} | `{ner_ppl:.2f}` | `{dp_ppl:.2f}` | `+{nsmc_deg:.2f} nats` | `+{uns_deg:.2f} nats` | `{mcqa:.1f}%` | `{hs:.1f}%` |"
        )

    for h11_id, h11_disp in [
        ("nochar_full_pos_charhead_h11", "25-Lane Output-Only Char Head (Full POS, Winning H11)"),
        ("nochar_coarse_pos_charhead_h11", "25-Lane Output-Only Char Head (Coarse POS, Winning H11)"),
    ]:
        best_entry = charhead_metrics[h11_id]["best"]
        fc = best_entry["four_capabilities"]
        ner_ppl = fc["klue_ner"]["perplexity"]
        dp_ppl = fc["klue_dp"]["perplexity"]
        nsmc_deg = fc["noisy_text"]["nsmc"]["degradation"]
        uns_deg = fc["noisy_text"]["unsmile"]["degradation"]
        mcqa = fc["kormedqa"]["accuracy"] * 100.0
        hs = best_entry["ko_hellaswag"]["length"] * 100.0
        lines.append(
            f"| **{h11_disp}** | `{ner_ppl:.2f}` | `{dp_ppl:.2f}` | `+{nsmc_deg:.2f} nats` | `+{uns_deg:.2f} nats` | `{mcqa:.1f}%` | `{hs:.1f}%` |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Training Trajectory Across 20 Epochs (`3, 5, 10, 15, 20, best`)")
    lines.append("")
    lines.append("Below is the milestone checkpoint progression for both **Coarse POS** and **Full POS** versions of the Output-Only Target Lane Hangul Factorizer (`H11_jamo_050_char_050`).")
    lines.append("")
    lines.append("### 3.1 Coarse POS (`nochar_coarse_pos_charhead_h11`) Progression")
    lines.append("")
    lines.append("| Checkpoint | Seq2Seq BPJ $\\downarrow$ | Target PPL $\\downarrow$ | BLEU $\\uparrow$ | chrF $\\uparrow$ | KLUE-NER PPL $\\downarrow$ | KLUE-DP PPL $\\downarrow$ | KoBALT Choseong | CUTE Jamo Surgery | **KoBALT Overall $\\uparrow$** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ckpt_tag in ["3_epoch", "5_epoch", "10_epoch", "15_epoch", "20_epoch", "best"]:
        d = charhead_metrics["nochar_coarse_pos_charhead_h11"][ckpt_tag]
        s2s = d["seq2seq"]
        fc = d["four_capabilities"]
        ks = d["kobalt_cute_hangul"]
        lines.append(
            f"| `{ckpt_tag}` | `{s2s['bpj']:.3f}` | `{s2s['target_ppl']:.2f}` | `{s2s['bleu']:.2f}` | `{s2s['chrf']:.2f}` | "
            f"`{fc['klue_ner']['perplexity']:.2f}` | `{fc['klue_dp']['perplexity']:.2f}` | "
            f"`{ks['choseong_quiz']['accuracy']:.1f}%` | `{ks['cute_jamo_manipulation']['accuracy']:.1f}%` | **`{ks['overall_accuracy']:.2f}%`** |"
        )

    lines.append("")
    lines.append("### 3.2 Full POS (`nochar_full_pos_charhead_h11`) Progression")
    lines.append("")
    lines.append("| Checkpoint | Seq2Seq BPJ $\\downarrow$ | Target PPL $\\downarrow$ | BLEU $\\uparrow$ | chrF $\\uparrow$ | KLUE-NER PPL $\\downarrow$ | KLUE-DP PPL $\\downarrow$ | KoBALT Choseong | CUTE Jamo Surgery | **KoBALT Overall $\\uparrow$** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ckpt_tag in ["3_epoch", "5_epoch", "10_epoch", "15_epoch", "20_epoch", "best"]:
        d = charhead_metrics["nochar_full_pos_charhead_h11"][ckpt_tag]
        s2s = d["seq2seq"]
        fc = d["four_capabilities"]
        ks = d["kobalt_cute_hangul"]
        lines.append(
            f"| `{ckpt_tag}` | `{s2s['bpj']:.3f}` | `{s2s['target_ppl']:.2f}` | `{s2s['bleu']:.2f}` | `{s2s['chrf']:.2f}` | "
            f"`{fc['klue_ner']['perplexity']:.2f}` | `{fc['klue_dp']['perplexity']:.2f}` | "
            f"`{ks['choseong_quiz']['accuracy']:.1f}%` | `{ks['cute_jamo_manipulation']['accuracy']:.1f}%` | **`{ks['overall_accuracy']:.2f}%`** |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Architectural Synthesis & Key Conclusions")
    lines.append("")
    lines.append("1. **Eliminating Input Vocabulary Bloat While Retaining Syllable-Level Supervision:**")
    lines.append("   - Standard 25-lane Hangul factorizers require an $11,173 \\times 384$ input character embedding matrix (`wte_char`), adding 4.29M parameters and creating vulnerability to rare/unseen Hangul syllable combinations at the input layer.")
    lines.append("   - By moving Lane 24 (`char`) to `output_only_lanes=[24]`, the transformer processes **purely compositional Jamo + POS + Byte-Fallback BPE representations at the input**, while the output layer (`lm_head_char`) enforces joint syllable-level and Jamo-level co-prediction.")
    lines.append("2. **Symmetric Dual-Target Loss Weighting (`H11_jamo_050_char_050`) Prevents Gradient Dominance:**")
    lines.append("   - Splitting supervision symmetrically between the composed syllable output head ($w_{\\text{char}}=0.5$) and the decomposed Jamo output heads ($w_{\\text{jamo}}=0.5$) yields the best validation loss (`0.7635` coarse POS / `0.7757` full POS) and improves Seq2Seq translation BLEU/chrF (`0.32 / 5.02` at `best_ckpt.pt`, rising to `0.35 / 5.33` at `20_epochs.pt`).")
    lines.append("3. **Superior Sub-Syllable Manipulation & Linguistic Probing (KoBALT & CUTE-Hangul):**")
    lines.append("   - On the 5-task KoBALT & CUTE-Hangul benchmark, the Output-Only Target Lane Hangul Factorizer (`H11`) reaches **`44.44%` overall accuracy** (and **`47.22%`** at epoch 3), tying the highest score across all weighted models and outperforming both the unweighted 24-lane no-char models (`36.11%`) and the 25-lane input-character models (`41.67%`).")

    os.makedirs(os.path.join(REPO_ROOT, "report"), exist_ok=True)
    report_path = os.path.join(REPO_ROOT, "report", "charhead_h11_vs_all_best_checkpoints_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Report written to:", report_path)


if __name__ == "__main__":
    main()
