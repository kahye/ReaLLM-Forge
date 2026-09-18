#!/usr/bin/env python3
"""Generate comprehensive markdown report for the Exhaustive Loss Weight Search on the
25-Lane Output-Only Char Target Lane Hangul Factorizer + Byte-Fallback BPE (`output_only_lanes=[24]`).
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
from pathlib import Path

with open("exhaustive_charhead_loss_weight_search_results.json") as f:
    data = json.load(f)

LANES_INFO = [
    ("script_bpe", "Tier 0: Lexical BPE & Byte-Fallback (`V=4113`)"),
    ("choseong", "Tier 1: Primary Jamo Onset (`V=20`)"),
    ("jungseong", "Tier 1: Primary Jamo Nucleus (`V=22`)"),
    ("jongseong", "Tier 1: Primary Jamo Coda (`V=28`)"),
    ("jung_base1", "Tier 2: Sub-Jamo Vowel Base 1"),
    ("jung_base2", "Tier 2: Sub-Jamo Vowel Base 2"),
    ("jung_has_w", "Tier 2: Sub-Jamo Glide +w"),
    ("jung_has_y", "Tier 2: Sub-Jamo Glide +y"),
    ("jung_has_i", "Tier 2: Sub-Jamo Glide +i"),
    ("jong_base1", "Tier 2: Sub-Jamo Coda Base 1"),
    ("jong_base2", "Tier 2: Sub-Jamo Coda Base 2"),
    ("jong_base3", "Tier 2: Sub-Jamo Coda Base 3"),
    ("choseong_tense", "Tier 3: Phonological Onset Tense"),
    ("choseong_aspirated", "Tier 3: Phonological Onset Aspirated"),
    ("choseong_nasal_liquid", "Tier 3: Phonological Onset Nasal/Liquid"),
    ("choseong_place", "Tier 3: Phonological Onset Place"),
    ("jung_height", "Tier 3: Phonological Vowel Height"),
    ("jung_backness", "Tier 3: Phonological Vowel Backness"),
    ("jung_round", "Tier 3: Phonological Vowel Rounding"),
    ("jong_complex", "Tier 3: Phonological Complex Coda"),
    ("has_batchim", "Tier 3: Phonological Batchim Presence"),
    ("syllable_index_mod", "Tier 4: Modular Syllable Index Hash (`mod 17`)"),
    ("codepoint_mod", "Tier 4: Modular Codepoint Hash (`mod 31`)"),
    ("pos", "Tier 5: Morphosyntactic POS Tags (`V=20`)"),
    ("char", "Tier 6: Output-Only Hangul Syllable LM Head (`V=11173`, `output_only_lanes=[24]`)"),
]

lines = []
def add(s=""):
    lines.append(s)

add("# Exhaustive Loss Weight Search Report: 25-Lane Output-Only Character Target Lane Hangul Factorizer (`output_only_lanes=[24]`)")
add()
add("## 1. Executive Summary & Architectural Motivation")
add()
add("This report details the weight search space design, empirical validation loss matrix across 16 configurations, per-lane optimization winners across all 25 lanes, and Pareto-optimal selection criteria for the **25-Lane Output-Only Character Target Lane Hangul Factorizer + Byte-Fallback BPE**.")
add()
add("### Why Decoupling Input Tokenization from Output Character Generation Solves OOV")
add("1. **Zero Input Character Vocabulary Overhead (`wte_24` Omitted):**")
add("   In traditional multi-stream models, summing a character input embedding (`wte_char`) into the hidden state $x = \\sum_i \\text{wte}_i(t_i)$ causes severe out-of-vocabulary (OOV) degradation whenever a rare or unseen Hangul syllable appears in the prompt. By registering Lane 24 (`char`) in `output_only_lanes=[24]`, **no input embedding table `wte_24` is created or summed**. Every Hangul syllable in the prompt is decomposed strictly into the 24 in-vocabulary factor/BPE lanes (`script_bpe`, `choseong`, `jungseong`, `jongseong`, phonological features, and `pos`).")
add("2. **Direct Single-Step Syllable Supervision at Output (`lm_head_24` Included):**")
add("   At the output layer, `lm_head_24` projects the final hidden state $x$ directly onto all $11,172$ modern Hangul syllables (`0xAC00–0xD7A3` + `<PAD>`, $V=11,173$). During training, its cross-entropy loss $\\mathcal{L}_{\\text{char}}$ shapes the transformer's final representation to linearly separate exact syllable identities. During autoregressive decoding, when Lane 0 predicts `<hangul>`, the model can decode the exact syllable from `lm_head_24` and deterministically factorize it into the 24 input lanes for the next step.")
add()
add("> [!IMPORTANT]")
add("> **Key Findings from the 16-Configuration Exhaustive Weight Search:**")
add("> 1. **Global 25-Lane Validation Loss & BPE/POS Winner (`H11_jamo_050_char_050`):**")
add(">    Setting symmetric dual-target supervision **$w_{\\text{bpe}}=1.00, w_{\\text{jamo}}=0.50, w_{\\text{char}}=0.50, w_{\\text{pos}}=0.50, w_{\\text{aux}}=0.05$** achieves the **lowest overall 25-lane mean validation loss (`0.47317`)**, the **#1 lowest `script_bpe` validation loss (`1.49518`)**, and the **#1 lowest `pos` validation loss (`0.44716`)** across the entire search space.")
add("> 2. **Joint Hybrid Generation Winner (`H10_jamo_050_char_100`):**")
add(">    Setting **$w_{\\text{bpe}}=1.00, w_{\\text{jamo}}=0.50, w_{\\text{char}}=1.00, w_{\\text{pos}}=0.50, w_{\\text{aux}}=0.05$** achieves the **#1 lowest Joint Hybrid Total Loss (`5.34024`)** ($\\mathcal{L}_{\\text{script\\_bpe}} + \\mathcal{L}_{\\text{char}} + \\mathcal{L}_{\\text{jamo\\_sum}} + \\mathcal{L}_{\\text{pos}}$), providing the strongest combined character-head and compositional jamo generation accuracy.")
add("> 3. **Primary Jamo Compositional Winner (`H05_char_050_balanced`):**")
add(">    Setting **$w_{\\text{bpe}}=1.00, w_{\\text{jamo}}=1.00, w_{\\text{char}}=0.50, w_{\\text{pos}}=0.50, w_{\\text{aux}}=0.05$** achieves the **#1 lowest Primary Jamo validation loss sum (`1.97508`)** (`choseong=0.74100`, `jungseong=0.76749`).")
add()
add("---")
add()
add("## 2. Structured 7-Tier Weight Search Space Design")
add()
add("To systematically evaluate how the output-only character head (`Tier 6`) interacts with the 24 input factor lanes (`Tiers 0–5`), the 25 lanes are partitioned into **7 Functional Tiers**:")
add()
add("| Tier | Lane Indices | Lane Names | Vocabulary & Role | Searched Weight Range |")
add("| :--- | :---: | :--- | :--- | :---: |")
add("| **Tier 0** | `0` | `script_bpe` | Non-Korean BPE subwords, 256 byte-fallback tokens, `<hangul>` trigger (`V=4113`) | Fixed reference $w_{\\text{bpe}} = 1.00$ |")
add("| **Tier 1** | `1–3` | `choseong`, `jungseong`, `jongseong` | Primary syllabic Jamo constituents (`V=20, 22, 28`) | $w_{\\text{jamo}} \\in [0.05, 1.00]$ |")
add("| **Tier 2** | `4–11` | `jung_base1..2`, `jung_has_w/y/i`, `jong_base1..3` | Sub-character diphthong & complex coda decomposition | $w_{\\text{subjamo}} \\in [0.00, 1.00]$ |")
add("| **Tier 3** | `12–20` | `choseong_tense..place`, `jung_height..round`, `has_batchim` | Articulatory & phonological features | $w_{\\text{phon}} \\in [0.00, 1.00]$ |")
add("| **Tier 4** | `21–22` | `syllable_index_mod`, `codepoint_mod` | Modular arithmetic checksums (`mod 17`, `mod 31`) | $w_{\\text{mod}} \\in [0.00, 1.00]$ |")
add("| **Tier 5** | `23` | `pos` | Morphosyntactic part-of-speech tag supervision (`V=20`) | $w_{\\text{pos}} \\in [0.40, 1.00]$ |")
add("| **Tier 6** | `24` | `char` *(Output-Only)* | Dedicated Hangul syllable output head (`V=11173`, no input `wte`) | $w_{\\text{char}} \\in [0.10, 2.00]$ |")
add()
add("### 2.1 Complete Specification of the 16 Search Space Configurations")
add()
add("| Config ID | Tier Weights $(w_{\\text{bpe}}, w_{\\text{jamo}}, w_{\\text{subjamo}}, w_{\\text{phon}}, w_{\\text{mod}}, w_{\\text{pos}}, w_{\\text{char}})$ | Architectural Hypothesis Tested |")
add("| :--- | :---: | :--- |")
add("| **`H01_uniform_1.0`** | `(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)` | Uniform unweighted baseline across all 25 heads |")
add("| **`H02_legacy_char_dominant`** | `(1.00, 0.05, 0.05, 0.05, 0.05, 0.50, 1.00)` | Legacy heuristic: full weight on `char` (`1.0`), minimal weight on Jamo (`0.05`) |")
add("| **`H03_char_010_light`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 0.10)` | Output-only `char` sweep: light auxiliary character supervision ($w_{\\text{char}}=0.10$) |")
add("| **`H04_char_025_mild`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 0.25)` | Output-only `char` sweep: mild character supervision ($w_{\\text{char}}=0.25$) |")
add("| **`H05_char_050_balanced`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 0.50)` | Output-only `char` sweep: balanced character supervision ($w_{\\text{char}}=0.50$) |")
add("| **`H06_char_100_parity`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 1.00)` | Output-only `char` sweep: full $1:1:1$ parity (`bpe=1.0, jamo=1.0, char=1.0`) |")
add("| **`H07_char_150_boost`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 1.50)` | Output-only `char` sweep: boosted character supervision ($w_{\\text{char}}=1.50$) |")
add("| **`H08_char_200_heavy`** | `(1.00, 1.00, 0.05, 0.05, 0.05, 0.50, 2.00)` | Output-only `char` sweep: heavy character emphasis ($w_{\\text{char}}=2.00$) |")
add("| **`H09_jamo_025_char_100`** | `(1.00, 0.25, 0.05, 0.05, 0.05, 0.50, 1.00)` | Jamo sweep under full `char` parity: $w_{\\text{jamo}}=0.25, w_{\\text{char}}=1.00$ |")
add("| **`H10_jamo_050_char_100`** | `(1.00, 0.50, 0.05, 0.05, 0.05, 0.50, 1.00)` | **Joint Hybrid Winner**: $w_{\\text{jamo}}=0.50, w_{\\text{char}}=1.00$ |")
add("| **`H11_jamo_050_char_050`** | `(1.00, 0.50, 0.05, 0.05, 0.05, 0.50, 0.50)` | **Global Mean-25 & BPE/POS Winner**: symmetric dual-target balance ($0.50 / 0.50$) |")
add("| **`H12_aux_zero_ablation`** | `(1.00, 1.00, 0.00, 0.00, 0.00, 0.50, 1.00)` | Pure primary + char ablation (zeros out all 19 auxiliary structural lanes) |")
add("| **`H13_hierarchical_decay`** | `(1.00, 1.00, 0.15, 0.08, 0.02, 0.50, 1.00)` | Hierarchical tier decay across sub-jamo (`0.15`), phon (`0.08`), mod (`0.02`) |")
add("| **`H14_hierarchical_char_075`** | `(1.00, 0.85, 0.15, 0.08, 0.02, 0.40, 0.75)` | Balanced hierarchical decay with $w_{\\text{char}}=0.75$ |")
add("| **`H15_entropy_proportional`** | Custom $\\propto \\log_2(V_i)$ | Information-theoretic scaling proportional to each lane's vocabulary entropy |")
add("| **`H16_pareto_synthesized`** | `(1.00, 0.85, 0.10, 0.05, 0.02, 0.45, 0.85)` | Synthesized multi-objective Pareto-balanced vector |")
add()
add("---")
add()
add("## 3. Complete Validation Loss Matrix Across All 16 Configurations")
add()
add("The table below reports exact validation cross-entropy losses across key composite metrics and individual heads:")
add("- **Char Surface Loss ($\\mathcal{L}_{\\text{char\\_surface}}$)**: $\\mathcal{L}_{\\text{script\\_bpe}} + \\mathcal{L}_{\\text{char}}$ (direct single-step syllable + BPE generation).")
add("- **Jamo Recon Loss ($\\mathcal{L}_{\\text{jamo\\_recon}}$)**: $\\mathcal{L}_{\\text{script\\_bpe}} + \\mathcal{L}_{\\text{choseong}} + \\mathcal{L}_{\\text{jungseong}} + \\mathcal{L}_{\\text{jongseong}}$.")
add("- **Hybrid Total Loss ($\\mathcal{L}_{\\text{hybrid\\_total}}$)**: $\\mathcal{L}_{\\text{script\\_bpe}} + \\mathcal{L}_{\\text{char}} + \\mathcal{L}_{\\text{jamo\\_sum}} + \\mathcal{L}_{\\text{pos}}$.")
add("- **Mean 25-Lane Loss**: Unweighted arithmetic mean of validation cross-entropy loss across all 25 lanes.")
add()
add("| Config ID | Char Surface $\\downarrow$ | Jamo Recon $\\downarrow$ | Hybrid Total $\\downarrow$ | Mean 25-Lane $\\downarrow$ | `script_bpe` $\\downarrow$ | `char` (Lane 24) $\\downarrow$ | Primary Jamo Sum $\\downarrow$ | `pos` $\\downarrow$ |")
add("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for cfg_key, info in data.items():
    m = info["metrics"]
    cs = f"**{m['char_surface_loss']:.5f}**" if cfg_key == "H02_legacy_char_dominant" else f"{m['char_surface_loss']:.5f}"
    jr = f"**{m['jamo_recon_loss']:.5f}**" if cfg_key == "H03_char_010_light" else f"{m['jamo_recon_loss']:.5f}"
    ht = f"**{m['hybrid_total_loss']:.5f}**" if cfg_key == "H10_jamo_050_char_100" else f"{m['hybrid_total_loss']:.5f}"
    m25 = f"**{m['mean_25_loss']:.5f}**" if cfg_key == "H11_jamo_050_char_050" else f"{m['mean_25_loss']:.5f}"
    bpe = f"**{m['script_bpe_loss']:.5f}**" if cfg_key == "H11_jamo_050_char_050" else f"{m['script_bpe_loss']:.5f}"
    ch = f"**{m['char_loss']:.5f}**" if cfg_key == "H02_legacy_char_dominant" else f"{m['char_loss']:.5f}"
    pj = f"**{m['primary_jamo_sum']:.5f}**" if cfg_key == "H05_char_050_balanced" else f"{m['primary_jamo_sum']:.5f}"
    pos = f"**{m['pos_loss']:.5f}**" if cfg_key == "H11_jamo_050_char_050" else f"{m['pos_loss']:.5f}"
    add(f"| `{cfg_key}` | {cs} | {jr} | {ht} | {m25} | {bpe} | {ch} | {pj} | {pos} |")

add()
add("---")
add()
add("## 4. Winning Weight Selection Per Individual Lane (All 25 Lanes)")
add()
add("To inspect which loss weight configuration produces the lowest validation loss for each individual lane ($0 \\dots 24$), the table below lists the **#1 Winning Configuration**, its **Winning Lane Weight**, and its **Minimum Validation Cross-Entropy Loss** for every lane:")
add()
add("| Lane Index | Lane Name | Functional Tier & Role | Best Configuration | Best Val Loss | Winning Lane Weight |")
add("| :---: | :--- | :--- | :--- | :---: | :---: |")

for idx, (lane_name, lane_desc) in enumerate(LANES_INFO):
    ranked = sorted(data.items(), key=lambda kv: kv[1]["lane_losses"][lane_name])
    best_k, best_v = ranked[0]
    val_loss = best_v["lane_losses"][lane_name]
    w = best_v["weights"][lane_name]
    add(f"| **{idx}** | `{lane_name}` | {lane_desc} | **`{best_k}`** | `{val_loss:.5f}` | `{w}` |")

add()
add("---")
add()
add("## 5. How the Winning Weights Are Selected & Practical Recommendations")
add()
add("### 5.1 Selection Methodology: Balancing Dual-Representation Synergy")
add("Because the 25-lane model predicts both **compositional Jamo factors (`Lanes 1–3`)** and **direct syllable identity (`Lane 24`)** from the shared hidden state $x$, the loss weights govern a multi-task trade-off:")
add("1. **Why Pure Character Dominance (`H02_legacy_char_dominant`, $w_{\\text{jamo}}=0.05, w_{\\text{char}}=1.00$) Is Suboptimal for Factorized Reasoning:**")
add("   Although `H02` achieves the lowest standalone `char` loss (`1.40811`), starving the Primary Jamo heads (`w_jamo=0.05`) degrades Primary Jamo validation loss to `2.00162` (the worst among all regularized runs). If an OOV prompt or rare syllable requires compositional fallback, under-trained Jamo heads reduce accuracy.")
add("2. **Why Symmetric Dual-Target Supervision (`H11_jamo_050_char_050`, $w_{\\text{jamo}}=0.50, w_{\\text{char}}=0.50$) Wins Global Generalization:**")
add("   Assigning equal moderate weight (`0.50`) to both Primary Jamo (`choseong`, `jungseong`, `jongseong`) and the output-only Character Head (`char`) while keeping `script_bpe` at `1.00` creates constructive multi-task regularization. Both representations reinforce the same underlying syllable manifold without dominating the gradient norm, achieving:")
add("   - **#1 Global Mean 25-Lane Validation Loss:** `0.47317`")
add("   - **#1 Non-Korean Byte-Fallback BPE Validation Loss (`script_bpe`):** `1.49518`")
add("   - **#1 Morphosyntactic POS Validation Loss (`pos`):** `0.44716`")
add("3. **Why Joint Hybrid Weighting (`H10_jamo_050_char_100`, $w_{\\text{jamo}}=0.50, w_{\\text{char}}=1.00$) Wins Direct + Compositional Decoding:**")
add("   When direct single-step decoding from `lm_head_24` is the primary generation mechanism (with Jamo heads serving as compositional verification), setting $w_{\\text{char}}=1.00$ alongside $w_{\\text{jamo}}=0.50$ achieves the **#1 lowest Joint Hybrid Total Loss (`5.34024`)** ($\\mathcal{L}_{\\text{script\\_bpe}} + \\mathcal{L}_{\\text{char}} + \\mathcal{L}_{\\text{jamo\\_sum}} + \\mathcal{L}_{\\text{pos}}$).")
add()
add("### 5.2 Recommended Production Weight Configurations")
add()
add("Depending on your downstream decoding mode, use one of the two Pareto-winning configurations below:")
add()
add("#### Option A: Best Overall Generalization & Symmetric Dual-Decoding (`H11_jamo_050_char_050`)")
add("- **Weights:** `w_bpe = 1.0`, `w_jamo = 0.5`, `w_pos = 0.5`, `w_char = 0.5`, `w_struct = 0.05`")
add("- **Command Line:**")
add("  ```bash")
add("  ./venv/bin/python3 train.py \\")
add("    --dataset korean_pos_mc_nochar/script_bpe \\")
add("    --training_mode multicontext --multicontext \\")
add("    --multicontext_datasets korean_pos_mc_nochar/script_bpe ... korean_pos_mc_nochar/pos korean_pos_mc_nochar/char \\")
add("    --output_only_datasets korean_pos_mc_nochar/char \\")
add("    --multicontext_loss_weights 1.0 0.5 0.5 0.5 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.5 0.5")
add("  ```")
add()
add("#### Option B: Best Joint Hybrid & Character-Head Generation (`H10_jamo_050_char_100`)")
add("- **Weights:** `w_bpe = 1.0`, `w_jamo = 0.5`, `w_pos = 0.5`, `w_char = 1.0`, `w_struct = 0.05`")
add("- **Command Line (or via `run_nochar_hybrid_pos_experiments.py`):**")
add("  ```bash")
add("  ./venv/bin/python3 run_nochar_hybrid_pos_experiments.py --experiments nochar_coarse_pos_charhead_weighted")
add("  ```")

report_content = "\n".join(lines)
os.makedirs("report", exist_ok=True)
with open("report/exhaustive_charhead_loss_weight_search_report.md", "w") as f:
    f.write(report_content)

print("Saved report/exhaustive_charhead_loss_weight_search_report.md successfully!")
