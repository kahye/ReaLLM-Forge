#!/usr/bin/env python3
"""Generate comprehensive best-checkpoint comparison report across all 6 model families (15 models total)
following the structure of /google/data/rw/users/ka/kahye/gemma_tokens/best_checkpoints_comparison_report.md
and incorporating all tasks including Seq2Seq Translation and KoBALT & CUTE-Hangul Diagnostic Suite.
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

    # Best iterations and validation losses for all 15 models
    meta_info = {
        "baseline_character": ("62,000", 1.4720),
        "baseline_char_bpe": ("71,000", 1.4768),
        "baseline_pure_byte": ("106,000", 1.0072),
        "mc_full_pos_weighted": ("72,000", 0.5790),
        "mc_full_pos_unweighted": ("65,000", 0.2639),
        "mc_coarse_pos_weighted": ("67,000", 0.5723),
        "mc_coarse_pos_unweighted": ("57,000", 0.2664),
        "mc_nochar_full_pos_new_weighted": ("43,000", 0.8112),
        "mc_nochar_full_pos_unweighted": ("43,000", 1.0312),
        "mc_nochar_coarse_pos_new_weighted": ("45,000", 0.7984),
        "mc_nochar_coarse_pos_unweighted": ("45,000", 1.0189),
        "nochar_full_pos_charhead_h11": ("54,796", 0.7757),
        "nochar_full_pos_charhead_unweighted": ("54,646", 1.1512),
        "nochar_coarse_pos_charhead_h11": ("54,796", 0.7635),
        "nochar_coarse_pos_charhead_unweighted": ("54,646", 1.1398),
    }

    # (logical_id, display_name, kobalt_key, s2s_key, all11_key, is_charhead)
    models_spec = [
        # 1. Baseline Tokenizers
        ("baseline_character", "Baseline Character Tokenizer (`V=4,924`)", "baseline_character", "baseline_character", "baseline_character", False),
        ("baseline_char_bpe", "Baseline Byte-Fallback BPE (`V=4,924`)", "baseline_char_bpe", "baseline_char_bpe", "baseline_char_bpe", False),
        ("baseline_pure_byte", "Baseline Pure Byte Tokenizer (`V=256`)", "baseline_pure_byte", "baseline_pure_byte", "baseline_pure_byte", False),
        # 2. Hangul Factorizer WITH Character Stream & Byte Fallback (25 Streams)
        ("mc_full_pos_weighted", "Hangul Factorizer + Char Stream (Full POS, Weighted)", "mc_25_full_weighted", "mc_full_pos_weighted_byte", "byte_fallback_full_pos_weighted", False),
        ("mc_full_pos_unweighted", "Hangul Factorizer + Char Stream (Full POS, Unweighted)", "mc_25_full_unweighted", "mc_full_pos_unweighted_byte", "byte_fallback_full_pos_unweighted", False),
        ("mc_coarse_pos_weighted", "Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "mc_25_coarse_weighted", "mc_coarse_pos_weighted_byte", "byte_fallback_coarse_pos_weighted", False),
        ("mc_coarse_pos_unweighted", "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "mc_25_coarse_unweighted", "mc_coarse_pos_unweighted_byte", "byte_fallback_coarse_pos_unweighted", False),
        # 3. No-Char Hangul Factorizer WITH Byte-Fallback BPE (24 Streams)
        ("mc_nochar_full_pos_new_weighted", "No-Char Hangul Factorizer (Full POS, Newly Weighted)", "mc_24_nochar_full_new_weighted", "nochar_full_pos_new_weighted", "nochar_full_pos_new_weighted", False),
        ("mc_nochar_full_pos_unweighted", "No-Char Hangul Factorizer (Full POS, Unweighted)", "mc_24_nochar_full_unweighted", "mc_nochar_full_pos_unweighted", "nochar_full_pos_unweighted", False),
        ("mc_nochar_coarse_pos_new_weighted", "No-Char Hangul Factorizer (Coarse POS, Newly Weighted)", "mc_24_nochar_coarse_new_weighted", "nochar_coarse_pos_new_weighted", "nochar_coarse_pos_new_weighted", False),
        ("mc_nochar_coarse_pos_unweighted", "No-Char Hangul Factorizer (Coarse POS, Unweighted)", "mc_24_nochar_coarse_unweighted", "mc_nochar_coarse_pos_unweighted", "nochar_coarse_pos_unweighted", False),
        # 4. Output-Only Target Lane Hangul Factorizer WITH Byte-Fallback BPE (25 Streams, output_only_lanes=[24])
        ("nochar_full_pos_charhead_h11", "Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)", "nochar_full_pos_charhead_h11", "nochar_full_pos_charhead_h11", "nochar_full_pos_charhead_h11", True),
        ("nochar_full_pos_charhead_unweighted", "Output-Only Char Head Factorizer (Full POS, Unweighted)", "nochar_full_pos_charhead_unweighted", "nochar_full_pos_charhead_unweighted", "nochar_full_pos_charhead_unweighted", True),
        ("nochar_coarse_pos_charhead_h11", "Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)", "nochar_coarse_pos_charhead_h11", "nochar_coarse_pos_charhead_h11", "nochar_coarse_pos_charhead_h11", True),
        ("nochar_coarse_pos_charhead_unweighted", "Output-Only Char Head Factorizer (Coarse POS, Unweighted)", "nochar_coarse_pos_charhead_unweighted", "nochar_coarse_pos_charhead_unweighted", "nochar_coarse_pos_charhead_unweighted", True),
    ]

    def get_s2s_best(s2s_key, is_charhead):
        if is_charhead:
            return charhead_metrics[s2s_key]["best"]["seq2seq"]
        if s2s_key in seq2seq_old:
            return seq2seq_old[s2s_key]["checkpoints"].get("best", seq2seq_old[s2s_key]["checkpoints"].get("15_epoch", {}))
        if s2s_key in seq2seq_new_nochar:
            return seq2seq_new_nochar[s2s_key].get("best", seq2seq_new_nochar[s2s_key].get("15_epoch", {}))
        return {}

    def get_kobalt_best(kkey, is_charhead):
        if is_charhead:
            return charhead_metrics[kkey]["best"]["kobalt_cute_hangul"]
        return kobalt_all.get(kkey, {}).get("scores", {})

    def get_fc_and_hs(all11_key, is_charhead):
        if is_charhead:
            entry = charhead_metrics[all11_key]["best"]
            fc = entry["four_capabilities"]
            hs = entry["ko_hellaswag"]
            return {
                "ner_logprob": fc["klue_ner"]["avg_logprob"],
                "ner_ppl": fc["klue_ner"]["perplexity"],
                "dp_logprob": fc["klue_dp"]["avg_logprob"],
                "dp_ppl": fc["klue_dp"]["perplexity"],
                "nsmc_clean": fc["noisy_text"]["nsmc"]["clean_logprob"],
                "nsmc_deg": fc["noisy_text"]["nsmc"]["degradation"],
                "unsmile_clean": fc["noisy_text"]["unsmile"]["clean_logprob"],
                "unsmile_deg": fc["noisy_text"]["unsmile"]["degradation"],
                "mcqa": fc["kormedqa"]["accuracy"] * 100.0,
                "hs_raw": hs.get("none", hs.get("raw", 0.0)) * 100.0,
                "hs_len": hs.get("length", hs.get("length_norm", 0.0)) * 100.0,
                "hs_prior": hs.get("prior", hs.get("prior_length_norm", 0.0)) * 100.0,
                "hs_unigram": hs.get("unigram", hs.get("unigram_length_norm", 0.0)) * 100.0,
            }
        if all11_key in all_11_best:
            entry = all_11_best[all11_key]
            fc = entry["four_capabilities"]
            hs = entry["ko_hellaswag"]
            return {
                "ner_logprob": fc["klue_ner_logprob"],
                "ner_ppl": fc["klue_ner_ppl"],
                "dp_logprob": fc["klue_dp_logprob"],
                "dp_ppl": fc["klue_dp_ppl"],
                "nsmc_clean": fc["nsmc_clean_logprob"],
                "nsmc_deg": fc["nsmc_degradation"],
                "unsmile_clean": fc["unsmile_clean_logprob"],
                "unsmile_deg": fc["unsmile_degradation"],
                "mcqa": fc["kormedqa_acc"] * 100.0,
                "hs_raw": hs["raw"] * 100.0,
                "hs_len": hs["length_norm"] * 100.0,
                "hs_prior": hs["prior_length_norm"] * 100.0,
                "hs_unigram": hs["unigram_length_norm"] * 100.0,
            }
        entry = new_nochar_epochs[all11_key]["best"]
        fc = entry["four_capabilities"]
        hs = entry["ko_hellaswag"]
        return {
            "ner_logprob": fc["klue_ner"]["avg_logprob"],
            "ner_ppl": fc["klue_ner"]["perplexity"],
            "dp_logprob": fc["klue_dp"]["avg_logprob"],
            "dp_ppl": fc["klue_dp"]["perplexity"],
            "nsmc_clean": fc["noisy_text"]["nsmc"]["clean_logprob"],
            "nsmc_deg": fc["noisy_text"]["nsmc"]["degradation"],
            "unsmile_clean": fc["noisy_text"]["unsmile"]["clean_logprob"],
            "unsmile_deg": fc["noisy_text"]["unsmile"]["degradation"],
            "mcqa": fc["kormedqa"]["accuracy"] * 100.0,
            "hs_raw": hs.get("none", hs.get("raw", 0.0)) * 100.0,
            "hs_len": hs.get("length", hs.get("length_norm", 0.0)) * 100.0,
            "hs_prior": hs.get("prior", hs.get("prior_length_norm", 0.0)) * 100.0,
            "hs_unigram": hs.get("unigram", hs.get("unigram_length_norm", 0.0)) * 100.0,
        }

    lines = []
    lines.append("# Comprehensive Evaluation Report: Best Validation Checkpoints (`best_ckpt.pt`) Across All 15 Trained Models & Downstream Tasks")
    lines.append("")
    lines.append("## 1. Executive Summary & Evaluated Architectures")
    lines.append("")
    lines.append("This report compares the **best validation loss checkpoints (`best_ckpt.pt`)** across all **6 model families (15 models total)** trained for 20 epochs on the 59.5M token Korean corpus (`batch_size=64`, `block_size=256`, `n_layer=6`, `n_head=6`, `n_embd=384`). Every model is evaluated on **all downstream tasks**, including the **English-to-Korean Seq2Seq Translation Benchmark** and the **KoBALT & CUTE-Hangul Korean Linguistic Diagnostic Suite**.")
    lines.append("")
    lines.append("### The 6 Evaluated Model Families (15 Models)")
    lines.append("1. **Baseline Character Tokenizer** (`baseline_character`, `V=4,924`, Best Iter: `62,000`)")
    lines.append("2. **Baseline Byte-Fallback BPE** (`baseline_char_bpe`, `V=4,924`, Best Iter: `71,000`)")
    lines.append("3. **Baseline Pure Byte Tokenizer** (`baseline_pure_byte`, `V=256`, Best Iter: `106,000`)")
    lines.append("4. **Hangul Factorizer WITH Character Stream & Byte Fallback (`25 Streams`, 4 Combinations)**:")
    lines.append("   - **Full POS (46 tags), Weighted Loss** (`w_char=1.0, w_pos=0.5, w_struct=0.05`, Best Iter: `72,000`)")
    lines.append("   - **Full POS (46 tags), Unweighted Loss** (`w=1.0` across all 25 lanes, Best Iter: `65,000`)")
    lines.append("   - **Coarse POS (17 tags), Weighted Loss** (`w_char=1.0, w_pos=0.5, w_struct=0.05`, Best Iter: `67,000`)")
    lines.append("   - **Coarse POS (17 tags), Unweighted Loss** (`w=1.0` across all 25 lanes, Best Iter: `57,000`)")
    lines.append("5. **No-Char Hangul Factorizer WITH Byte-Fallback BPE (`24 Streams`, No-Char, 4 Combinations)**:")
    lines.append("   - **No-Char Full POS (46 tags), Newly Weighted Loss** (`w_bpe=1.0, w_jamo=1.0, w_aux=0.05, w_pos=0.5`, Best Iter: `43,000`)")
    lines.append("   - **No-Char Full POS (46 tags), Unweighted Loss** (`w=1.0` across all 24 lanes, Best Iter: `43,000`)")
    lines.append("   - **No-Char Coarse POS (17 tags), Newly Weighted Loss** (`w_bpe=1.0, w_jamo=1.0, w_aux=0.05, w_pos=0.5`, Best Iter: `45,000`)")
    lines.append("   - **No-Char Coarse POS (17 tags), Unweighted Loss** (`w=1.0` across all 24 lanes, Best Iter: `45,000`)")
    lines.append("6. **Output-Only Target Lane Hangul Factorizer WITH Byte-Fallback BPE (`25 Streams`, `output_only_lanes=[24]`, 4 Combinations)**:")
    lines.append("   - **Output-Only Char Head Full POS (46 tags), Newly Weighted (`H11`)** (`w_bpe=1.0, w_jamo=0.5, w_char=0.5, w_pos=0.5, w_aux=0.05`, Best Iter: `54,796`)")
    lines.append("   - **Output-Only Char Head Full POS (46 tags), Unweighted** (`w=1.0` across all 25 lanes, Best Iter: `54,646`)")
    lines.append("   - **Output-Only Char Head Coarse POS (17 tags), Newly Weighted (`H11`)** (`w_bpe=1.0, w_jamo=0.5, w_char=0.5, w_pos=0.5, w_aux=0.05`, Best Iter: `54,796`)")
    lines.append("   - **Output-Only Char Head Coarse POS (17 tags), Unweighted** (`w=1.0` across all 25 lanes, Best Iter: `54,646`)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Summary of Best-Checkpoint Leaders by Capability:**")
    lines.append("> - **Lowest Overall Validation Loss Among No-Input-Char Architectures**: **Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)** achieves **`0.7635`** validation loss (and **`0.7757`** on Full POS), outperforming both 24-lane No-Char models (`0.7984`) and saving **4.29M input parameters** compared to 25-lane input-char models.")
    lines.append("> - **Seq2Seq Translation (`EN -> KO`) Multi-Stream Generation Quality**: **Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)** achieves the highest multi-stream BLEU and chrF (**`0.32` BLEU / `5.02` chrF** at `best_ckpt.pt`), substantially outperforming 24-lane No-Char models (`0.00–0.04` BLEU / `2.83–3.42` chrF).")
    lines.append("> - **KoBALT & CUTE-Hangul Korean Linguistic & Sub-Syllable Diagnostic Suite**: **Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)** achieves **`44.44%`** overall accuracy (`50.0%` CUTE Jamo Surgery, `50.0%` Choseong Quiz, `87.5%` Allomorphic Particles), tying the top score across all weighted models.")
    lines.append("> - **Syntactic & Named Entity Modeling (`KLUE-NER` & `KLUE-DP`)**: **No-Char** and **Output-Only Char Head Factorizers** achieve the lowest perplexity across all architectures (`1.69–1.75` NER PPL / `1.46–1.48` DP PPL), outperforming Baseline Character (`8.35` / `11.51`) and BPE (`8.35` / `11.46`) by **4.8x to 7.8x**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Master Overview Table: Best Validation Checkpoints (`best_ckpt.pt`) Across All Downstream Tasks")
    lines.append("")
    lines.append("| Model Family & Variant | Best Iter | Val Loss | Seq2Seq BPJ (↓) | Seq2Seq chrF (↑) | KLUE-NER PPL (↓) | KLUE-DP PPL (↓) | Ko-HellaSwag Raw (↑) | Ko-HellaSwag Len-Norm (↑) | KorMedMCQA Acc (↑) | KoBALT & CUTE Overall Acc (↑) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        best_iter, val_loss = meta_info[mid]
        s2s = get_s2s_best(s2s_key, is_charhead)
        kscores = get_kobalt_best(kkey, is_charhead)
        fc_hs = get_fc_and_hs(all11_key, is_charhead)
        bpj = s2s.get("bpj", 0.0)
        chrf = s2s.get("chrf", 0.0)
        ner_ppl = fc_hs["ner_ppl"]
        dp_ppl = fc_hs["dp_ppl"]
        hs_raw = fc_hs["hs_raw"]
        hs_len = fc_hs["hs_len"]
        mcqa = fc_hs["mcqa"]
        kobalt_ov = kscores.get("overall_accuracy", 0.0)
        lines.append(
            f"| **{disp}** | `{best_iter}` | `{val_loss:.4f}` | `{bpj:.4f}` | `{chrf:.2f}` | `{ner_ppl:.2f}` | `{dp_ppl:.2f}` | **`{hs_raw:.1f}%`** | **`{hs_len:.1f}%`** | **`{mcqa:.1f}%`** | **`{kobalt_ov:.2f}%`** |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed English-to-Korean Seq2Seq Translation Benchmark (`best_ckpt.pt`)")
    lines.append("")
    lines.append("All 15 models' best validation checkpoints were evaluated on the English-to-Korean Seq2Seq Translation Benchmark (`data/korean_seq2seq_bench/test.jsonl`):")
    lines.append("- **Teacher-Forced Evaluation (`500` pairs)**: Measures exact cross-entropy target loss, Target Perplexity (PPL), and **BPJ (Bits-Per-Jamo)**.")
    lines.append("- **Autoregressive Generation (`100` pairs, `max_new_tokens=40`)**: Measures **BLEU-4**, **chrF** (`char_order=18` jamo character F-score), and **Latency (`ms / sentence`)**.")
    lines.append("")
    lines.append("| Model Family & Variant | Best Iter | Teacher-Forced BPJ (↓) | Target Loss (↓) | Target PPL (↓) | BLEU-4 (↑) | chrF (↑) | Latency (`ms / sent`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        best_iter, _ = meta_info[mid]
        s2s = get_s2s_best(s2s_key, is_charhead)
        bpj = s2s.get("bpj", 0.0)
        t_loss = s2s.get("target_loss", 0.0)
        t_ppl = s2s.get("target_ppl", 0.0)
        bleu = s2s.get("bleu", 0.0)
        chrf = s2s.get("chrf", 0.0)
        lat = s2s.get("latency_ms", 0.0)
        lines.append(
            f"| **{disp}** | `{best_iter}` | **`{bpj:.4f}`** | `{t_loss:.4f}` | `{t_ppl:.2f}` | `{bleu:.2f}` | **`{chrf:.2f}`** | `{lat:.1f} ms` |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Detailed Four Capabilities & Zero-Shot Reasoning Evaluation (`best_ckpt.pt`)")
    lines.append("")
    lines.append("### 4.1. Syntactic & Entity Perplexity (`KLUE-NER` & `KLUE-DP`) and Domain QA (`KorMedMCQA`)")
    lines.append("")
    lines.append("| Model Family & Variant | KLUE-NER Avg LogProb | KLUE-NER PPL (↓) | KLUE-DP Avg LogProb | KLUE-DP PPL (↓) | KorMedMCQA Acc. (↑) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        fc_hs = get_fc_and_hs(all11_key, is_charhead)
        lines.append(
            f"| **{disp}** | `{fc_hs['ner_logprob']:.4f}` | **`{fc_hs['ner_ppl']:.2f}`** | `{fc_hs['dp_logprob']:.4f}` | **`{fc_hs['dp_ppl']:.2f}`** | **`{fc_hs['mcqa']:.1f}%`** |"
        )

    lines.append("")
    lines.append("### 4.2. Zero-Shot Commonsense Reasoning (`Ko-HellaSwag` Across All 4 Normalizations)")
    lines.append("")
    lines.append("| Model Family & Variant | Raw Acc (`none`) | Length-Norm (`length`) | Prior-Norm (`prior`) | Unigram-Norm (`unigram`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        fc_hs = get_fc_and_hs(all11_key, is_charhead)
        lines.append(
            f"| **{disp}** | **`{fc_hs['hs_raw']:.1f}%`** | **`{fc_hs['hs_len']:.1f}%`** | `{fc_hs['hs_prior']:.1f}%` | `{fc_hs['hs_unigram']:.1f}%` |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. KoBALT & CUTE-Hangul Korean Linguistic & Sub-Syllable Diagnostic Suite (`best_ckpt.pt`)")
    lines.append("")
    lines.append("Evaluates 5 literature-grounded Korean sub-syllable & linguistic tasks from `benchmarks/run_kobalt_cute_hangul_eval.py`:")
    lines.append("1. **Choseong Quiz & Sub-Syllable Probing** (`choseong_quiz`)")
    lines.append("2. **Allomorphic Particle Minimal Pairs** (`allomorphic_particles`: `은/는`, `이/가`, `을/를`, `로/으로`)")
    lines.append("3. **Phonological Assimilation Rules** (`phonological_assimilation`: nasalization, liquidization, palatalization, aspiration)")
    lines.append("4. **Vowel Harmony & Ideophones** (`vowel_harmony_ideophones`)")
    lines.append("5. **CUTE-Hangul Jamo Surgery & Orthographic Editing** (`cute_jamo_manipulation`)")
    lines.append("")
    lines.append("| Model Family & Variant | Choseong Quiz | Allomorphic Particles | Assimilation Rules | Vowel Harmony | CUTE Jamo Surgery | **Overall Accuracy (↑)** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        kscores = get_kobalt_best(kkey, is_charhead)
        c_q = kscores.get("choseong_quiz", {}).get("accuracy", 0.0)
        allo = kscores.get("allomorphic_particles", {}).get("accuracy", 0.0)
        assim = kscores.get("phonological_assimilation", {}).get("accuracy", 0.0)
        vow = kscores.get("vowel_harmony_ideophones", {}).get("accuracy", 0.0)
        jamo = kscores.get("cute_jamo_manipulation", {}).get("accuracy", 0.0)
        ov = kscores.get("overall_accuracy", 0.0)
        lines.append(
            f"| **{disp}** | `{c_q:.1f}%` | `{allo:.1f}%` | `{assim:.1f}%` | `{vow:.1f}%` | `{jamo:.1f}%` | **`{ov:.2f}%`** |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Extreme Noise Resilience (`NSMC 80%` & `UnSmile 80%`) Evaluation (`best_ckpt.pt`)")
    lines.append("")
    lines.append("| Model Family & Variant | NSMC Clean LogProb | NSMC 80% Deg. (↓) | UnSmile Clean LogProb | UnSmile 80% Deg. (↓) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for mid, disp, kkey, s2s_key, all11_key, is_charhead in models_spec:
        fc_hs = get_fc_and_hs(all11_key, is_charhead)
        lines.append(
            f"| **{disp}** | `{fc_hs['nsmc_clean']:.4f}` | **`{fc_hs['nsmc_deg']:.2f}`** | `{fc_hs['unsmile_clean']:.4f}` | **`{fc_hs['unsmile_deg']:.2f}`** |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Architectural Conclusions Across All 6 Model Families & 4 Factorizer Combinations")
    lines.append("")
    lines.append("1. **Output-Only Target Lane Hangul Factorizer (`output_only_lanes=[24]`) vs. 25-Stream Input-Char vs. 24-Stream No-Char**:")
    lines.append("   - **Output-Only Char Head (`output_only_lanes=[24]`)** eliminates the 11,173-entry input character embedding matrix (`wte_char`), saving **4.29M input parameters** while retaining full syllable composition supervision at the output LM head (`lm_head_24`).")
    lines.append("   - When trained with symmetric dual-target supervision (**`H11_jamo_050_char_050`**), it achieves the **lowest validation loss (`0.7635` Coarse POS / `0.7757` Full POS)** across all no-input-char architectures and dramatically improves Seq2Seq generation quality (**`0.32` BLEU / `5.02` chrF**) over 24-lane No-Char models (`0.00–0.04` BLEU).")
    lines.append("2. **Full Sejong POS (`46 Tags`) vs. Coarse POS (`17 Tags`)**:")
    lines.append("   - **Full POS (`46 tags`)** excels on Seq2Seq translation generation quality (**`0.32` BLEU / `5.02` chrF**) and domain QA (**`23.0%` KorMedMCQA**).")
    lines.append("   - **Coarse POS (`17 tags`)** achieves stronger zero-shot multiple-choice reasoning (**`25.0%` length-normalized Ko-HellaSwag**) and higher KoBALT sub-syllable diagnostic accuracy (**`44.44%`**).")
    lines.append("3. **Newly Weighted Loss (`H11_jamo_050_char_050`) vs. Unweighted Loss (`w=1.0`)**:")
    lines.append("   - **Newly Weighted Loss (`H11`)** prevents the 19 phonological/vowel-harmony auxiliary lanes from drowning out the primary lexical (`script_bpe`), Jamo (`initial`, `medial`, `final`), and syllable (`char`) heads, improving validation loss by over **33%** (`0.7635` vs. `1.1398`) and boosting Seq2Seq translation BLEU.")

    os.makedirs(os.path.join(REPO_ROOT, "report"), exist_ok=True)
    report_path = os.path.join(REPO_ROOT, "report", "best_checkpoints_comparison_report_all_15_models.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Master 15-model report written to:", report_path)


if __name__ == "__main__":
    main()
