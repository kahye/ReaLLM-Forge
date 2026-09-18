#!/usr/bin/env python3
"""Comprehensive English-to-Korean Seq2Seq Translation Benchmark Evaluator for All 15 Models (Best Checkpoints).

Evaluates the 15 best validation loss checkpoints (`best_ckpt.pt`):
- 3 Baselines:
    1. baseline_character (`out_baseline_korean_pos/best_ckpt.pt`)
    2. baseline_char_bpe (`out_baseline_char_bpe/best_ckpt.pt`)
    3. baseline_pure_byte (`out_baseline_pure_byte/best_ckpt.pt`)
- 4 Hangul Factorizer with Character Stream & Byte-Fallback (`mc_25`):
    4. mc_full_pos_weighted_byte (`out_mc_full_pos_weighted_byte/best_ckpt.pt`)
    5. mc_full_pos_unweighted_byte (`out_mc_full_pos_unweighted_byte/best_ckpt.pt`)
    6. mc_coarse_pos_weighted_byte (`out_mc_coarse_pos_weighted_byte/best_ckpt.pt`)
    7. mc_coarse_pos_unweighted_byte (`out_mc_coarse_pos_unweighted_byte/best_ckpt.pt`)
- 4 No-Char Hangul Factorizer with Byte-Fallback BPE (`mc_24`, newly weighted):
    8. mc_nochar_full_pos_new_weighted (`out_mc_nochar_full_pos_new_weighted/best_ckpt.pt`)
    9. mc_nochar_full_pos_unweighted (`out_mc_nochar_full_pos_unweighted/best_ckpt.pt`)
    10. mc_nochar_coarse_pos_new_weighted (`out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt`)
    11. mc_nochar_coarse_pos_unweighted (`out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt`)
- 4 Output-Only Target Lane Hangul Factorizer with Byte-Fallback BPE (`charhead`, newly weighted H11):
    12. nochar_full_pos_charhead_h11 (`out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt`)
    13. nochar_full_pos_charhead_unweighted (`out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt`)
    14. nochar_coarse_pos_charhead_h11 (`out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt`)
    15. nochar_coarse_pos_charhead_unweighted (`out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt`)

Computes:
1. Overall Corpus Metrics: BPJ (Bits Per Jamo), Target Loss, Target Perplexity, Corpus BLEU, Corpus Jamo chrF, Latency (ms).
2. Per-Sample Metrics (for all 20 generation samples across all 15 models):
   - Sentence-level BPJ
   - Sentence-level BLEU
   - Sentence-level Compatibility-Jamo chrF (char_order=18)
   - Generated Korean translation response
"""
from __future__ import annotations
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
import sys
from typing import Any, Dict, List

import sacrebleu
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_seq2seq_translation_benchmark import (
    TEST_JSONL,
    TranslationEvaluator,
    canonicalize,
    load_model_from_ckpt,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS_SPEC = [
    # Baselines
    {
        "id": "baseline_character",
        "name": "Baseline Character Tokenizer (V=4,924)",
        "family": "Baseline",
        "type": "baseline_char",
        "ckpt_path": "out_baseline_korean_pos/best_ckpt.pt",
    },
    {
        "id": "baseline_char_bpe",
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "family": "Baseline",
        "type": "baseline_bpe",
        "ckpt_path": "out_baseline_char_bpe/best_ckpt.pt",
    },
    {
        "id": "baseline_pure_byte",
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "family": "Baseline",
        "type": "baseline_byte",
        "ckpt_path": "out_baseline_pure_byte/best_ckpt.pt",
    },
    # Hangul Factorizer + Char Stream (mc_25)
    {
        "id": "mc_full_pos_weighted_byte",
        "name": "25-Lane Factorizer + Char Stream (Full POS, Weighted)",
        "family": "25-Lane Char Stream",
        "type": "mc_25_full",
        "ckpt_path": "out_mc_full_pos_weighted_byte/best_ckpt.pt",
    },
    {
        "id": "mc_full_pos_unweighted_byte",
        "name": "25-Lane Factorizer + Char Stream (Full POS, Unweighted)",
        "family": "25-Lane Char Stream",
        "type": "mc_25_full",
        "ckpt_path": "out_mc_full_pos_unweighted_byte/best_ckpt.pt",
    },
    {
        "id": "mc_coarse_pos_weighted_byte",
        "name": "25-Lane Factorizer + Char Stream (Coarse POS, Weighted)",
        "family": "25-Lane Char Stream",
        "type": "mc_25_coarse",
        "ckpt_path": "out_mc_coarse_pos_weighted_byte/best_ckpt.pt",
    },
    {
        "id": "mc_coarse_pos_unweighted_byte",
        "name": "25-Lane Factorizer + Char Stream (Coarse POS, Unweighted)",
        "family": "25-Lane Char Stream",
        "type": "mc_25_coarse",
        "ckpt_path": "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt",
    },
    # No-Char Hangul Factorizer + Byte-Fallback BPE (mc_24, newly weighted)
    {
        "id": "mc_nochar_full_pos_new_weighted",
        "name": "24-Lane No-Char Factorizer (Full POS, Newly Weighted)",
        "family": "24-Lane No-Char",
        "type": "mc_24_full",
        "ckpt_path": "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt",
    },
    {
        "id": "mc_nochar_full_pos_unweighted",
        "name": "24-Lane No-Char Factorizer (Full POS, Unweighted)",
        "family": "24-Lane No-Char",
        "type": "mc_24_full",
        "ckpt_path": "out_mc_nochar_full_pos_unweighted/best_ckpt.pt",
    },
    {
        "id": "mc_nochar_coarse_pos_new_weighted",
        "name": "24-Lane No-Char Factorizer (Coarse POS, Newly Weighted)",
        "family": "24-Lane No-Char",
        "type": "mc_24_coarse",
        "ckpt_path": "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt",
    },
    {
        "id": "mc_nochar_coarse_pos_unweighted",
        "name": "24-Lane No-Char Factorizer (Coarse POS, Unweighted)",
        "family": "24-Lane No-Char",
        "type": "mc_24_coarse",
        "ckpt_path": "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt",
    },
    # Output-Only Target Lane Hangul Factorizer + Byte-Fallback BPE (charhead, newly weighted H11)
    {
        "id": "nochar_full_pos_charhead_h11",
        "name": "Output-Only Target Lane Factorizer (Full POS, Newly Weighted H11)",
        "family": "Output-Only Target Lane",
        "type": "mc_24_full",
        "ckpt_path": "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt",
    },
    {
        "id": "nochar_full_pos_charhead_unweighted",
        "name": "Output-Only Target Lane Factorizer (Full POS, Unweighted)",
        "family": "Output-Only Target Lane",
        "type": "mc_24_full",
        "ckpt_path": "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt",
    },
    {
        "id": "nochar_coarse_pos_charhead_h11",
        "name": "Output-Only Target Lane Factorizer (Coarse POS, Newly Weighted H11)",
        "family": "Output-Only Target Lane",
        "type": "mc_24_coarse",
        "ckpt_path": "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt",
    },
    {
        "id": "nochar_coarse_pos_charhead_unweighted",
        "name": "Output-Only Target Lane Factorizer (Coarse POS, Unweighted)",
        "family": "Output-Only Target Lane",
        "type": "mc_24_coarse",
        "ckpt_path": "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt",
    },
]


def main():
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    out_path = os.path.join(REPO_ROOT, "results_seq2seq_all_15_best_checkpoints_detailed.json")
    results: Dict[str, Any] = {
        "samples_metadata": [
            {"idx": idx, "en": s["en"], "ko": s["ko"], "canon_ko": canon_refs[idx]}
            for idx, s in enumerate(gen_samples)
        ],
        "models": {},
    }

    for spec in MODELS_SPEC:
        mid = spec["id"]
        ckpt_path = os.path.join(REPO_ROOT, spec["ckpt_path"])
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Missing checkpoint: {ckpt_path}")
            continue

        print(f"\n================================================================================")
        print(f" EVALUATING [{mid}]: {spec['name']}")
        print(f" Checkpoint: {spec['ckpt_path']}")
        print("================================================================================")

        model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
        val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
        iter_num = int(ckpt_dict.get("iter_num", 0))

        evaluator = TranslationEvaluator(spec, ckpt_dict=ckpt_dict, device=DEVICE)

        # 1. Corpus-level teacher-forced metrics (100 test samples)
        t_eval = evaluator.evaluate_teacher_forced(model, tf_samples)

        # 2. Autoregressive completions on 20 samples
        hyps, gen_lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)
        canon_hyps = [canonicalize(h) for h in hyps]
        filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
        filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]

        corpus_bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
        corpus_chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

        # 3. Per-sample metrics for each of the 20 samples
        per_sample_records: List[Dict[str, Any]] = []
        for idx, s in enumerate(gen_samples):
            single_tf = evaluator.evaluate_teacher_forced(model, [s])
            s_bpj = single_tf["bpj"]
            s_loss = single_tf["target_loss"]
            s_ppl = single_tf["target_ppl"]

            h_str = filt_hyps[idx]
            r_str = filt_refs[idx]
            s_bleu = sacrebleu.sentence_bleu(h_str, [r_str], smooth_method="exp", use_effective_order=True).score
            s_chrf = sacrebleu.sentence_chrf(h_str, [r_str], char_order=18).score

            per_sample_records.append(
                {
                    "sample_idx": idx,
                    "en": s["en"],
                    "ko_ref": s["ko"],
                    "response": hyps[idx],
                    "bpj": float(s_bpj),
                    "target_loss": float(s_loss),
                    "target_ppl": float(s_ppl),
                    "bleu": float(s_bleu),
                    "chrf": float(s_chrf),
                }
            )

        results["models"][mid] = {
            "id": mid,
            "name": spec["name"],
            "family": spec["family"],
            "ckpt_path": spec["ckpt_path"],
            "iter_num": iter_num,
            "train_val_loss": val_loss,
            "corpus_metrics": {
                "bpj": float(t_eval["bpj"]),
                "target_loss": float(t_eval["target_loss"]),
                "target_ppl": float(t_eval["target_ppl"]),
                "bleu": float(corpus_bleu),
                "chrf": float(corpus_chrf),
                "latency_ms": float(gen_lat_ms),
            },
            "per_sample": per_sample_records,
        }

        print(
            f"  => BPJ: {t_eval['bpj']:.4f} | PPL: {t_eval['target_ppl']:.2f} | "
            f"BLEU: {corpus_bleu:.2f} | chrF: {corpus_chrf:.2f} | Latency: {gen_lat_ms:.1f}ms"
        )

        # Save incrementally
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved full 15-model detailed seq2seq benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
