#!/usr/bin/env python3
"""Evaluate all Muon Char-Head checkpoints WITHOUT lm_head_24 across all downstream tasks."""

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
import time
import torch
import sacrebleu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_eval_without_lmhead24 import (
    load_model_from_ckpt,
    eval_seq2seq_teacher_forced_24lanes,
    eval_kobalt_without_charhead,
    generate_seq2seq_without_charhead,
    canonicalize,
    TEST_JSONL,
)
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "nochar_coarse_pos_charhead_h11": {
        "dir": "out_mc_nochar_coarse_pos_charhead_h11_muon",
        "pos_mode": "coarse",
        "milestones": [
            ("5_epochs", "5_epochs.pt"),
            ("10_epochs", "10_epochs.pt"),
            ("15_epochs", "15_epochs.pt"),
            ("20_epochs", "20_epochs.pt"),
            ("best_ckpt", "best_ckpt.pt"),
        ],
    },
    "nochar_full_pos_charhead_h11": {
        "dir": "out_mc_nochar_full_pos_charhead_h11_muon",
        "pos_mode": "full",
        "milestones": [
            ("5_epochs", "5_epochs.pt"),
            ("10_epochs", "10_epochs.pt"),
            ("15_epochs", "15_epochs.pt"),
            ("20_epochs", "20_epochs.pt"),
            ("best_ckpt", "best_ckpt.pt"),
        ],
    },
    "nochar_coarse_pos_charhead_unweighted": {
        "dir": "out_mc_nochar_coarse_pos_charhead_unweighted_muon",
        "pos_mode": "coarse",
        "milestones": [
            ("5_epochs", "5_epochs.pt"),
            ("10_epochs", "10_epochs.pt"),
            ("15_epochs", "15_epochs.pt"),
            ("20_epochs", "20_epochs.pt"),
            ("best_ckpt", "best_ckpt.pt"),
        ],
    },
    "nochar_full_pos_charhead_unweighted": {
        "dir": "out_mc_nochar_full_pos_charhead_unweighted_muon",
        "pos_mode": "full",
        "milestones": [
            ("5_epochs", "5_epochs.pt"),
            ("10_epochs", "10_epochs.pt"),
            ("15_epochs", "15_epochs.pt"),
            ("20_epochs", "20_epochs.pt"),
            ("best_ckpt", "best_ckpt.pt"),
        ],
    },
}

def main():
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")

    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    out_file = os.path.join(REPO_ROOT, "results_charhead_without_lmhead24_muon_all_metrics.json")
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for exp_key, exp_cfg in EXPERIMENTS.items():
        if exp_key not in results:
            results[exp_key] = {}
        tok = tok_coarse if exp_cfg["pos_mode"] == "coarse" else tok_full
        ckpt_dir = os.path.join(REPO_ROOT, exp_cfg["dir"])

        for ep_name, ckpt_name in exp_cfg["milestones"]:
            if ep_name in results[exp_key] and "seq2seq" in results[exp_key][ep_name]:
                print(f"Skipping already evaluated {exp_key} @ {ep_name}")
                continue

            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path} (not found)")
                continue

            print(f"\n==========================================================================")
            print(f" EVALUATING MUON WITHOUT LM_HEAD_24: {exp_key} @ {ep_name}")
            print(f" Checkpoint: {ckpt_path}")
            print(f"==========================================================================")

            model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
            iter_num = int(ckpt_dict.get("iter_num", 0))

            # 1. Seq2Seq Teacher-Forced (24 factor lanes)
            tf_metrics = eval_seq2seq_teacher_forced_24lanes(model, tok, tf_samples)

            # 2. Seq2Seq Autoregressive Generation without char head
            hyps, lat_ms = generate_seq2seq_without_charhead(model, tok, gen_samples)
            canon_hyps = [canonicalize(h) for h in hyps]
            filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
            filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]
            bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
            chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

            # 3. KoBALT & CUTE-Hangul Diagnostic Suite (pure Jamo + BPE)
            kobalt_scores = eval_kobalt_without_charhead(model, tok)

            entry = {
                "seq2seq": {
                    "iter_num": iter_num,
                    "bpj": tf_metrics["bpj"],
                    "target_loss": tf_metrics["target_loss"],
                    "target_ppl": tf_metrics["target_ppl"],
                    "bleu": float(bleu),
                    "chrf": float(chrf),
                    "latency_ms": float(lat_ms),
                },
                "kobalt_cute_hangul": kobalt_scores,
            }

            print(
                f"  -> {exp_key} @ {ep_name} | BPJ: {tf_metrics['bpj']:.3f} | "
                f"BLEU/chrF: {bleu:.2f}/{chrf:.2f} | Latency: {lat_ms:.1f}ms | "
                f"KoBALT Acc: {kobalt_scores['overall_accuracy']:.1f}%"
            )

            results[exp_key][ep_name] = entry
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAll Muon evaluations without lm_head_24 completed! Output saved to {out_file}")

if __name__ == "__main__":
    main()
