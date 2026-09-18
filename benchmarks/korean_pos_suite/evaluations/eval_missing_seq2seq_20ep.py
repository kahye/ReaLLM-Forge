#!/usr/bin/env python3
"""Evaluate 15_epoch, 20_epoch, and updated 20-epoch best_ckpt on Seq2Seq Translation
for the 3 Baselines and 4 Hangul Factorizer + Char Stream models in parallel on GPU.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import sacrebleu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_seq2seq_translation_benchmark import (
    TranslationEvaluator,
    canonicalize,
    load_model_from_ckpt,
    TEST_JSONL,
)

MODELS_TO_UPDATE = [
    {
        "id": "baseline_character",
        "name": "Baseline Character Tokenizer (V=4,924)",
        "type": "baseline_char",
        "dir": "out_baseline_korean_pos",
        "checkpoints": {
            "15_epoch": "out_baseline_korean_pos/15_epochs.pt",
            "20_epoch": "out_baseline_korean_pos/20_epochs.pt",
            "best": "out_baseline_korean_pos/best_ckpt.pt",
        },
    },
    {
        "id": "baseline_char_bpe",
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "type": "baseline_bpe",
        "dir": "out_baseline_char_bpe",
        "checkpoints": {
            "15_epoch": "out_baseline_char_bpe/15_epochs.pt",
            "20_epoch": "out_baseline_char_bpe/20_epochs.pt",
            "best": "out_baseline_char_bpe/best_ckpt.pt",
        },
    },
    {
        "id": "baseline_pure_byte",
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "type": "baseline_byte",
        "dir": "out_baseline_pure_byte",
        "checkpoints": {
            "15_epoch": "out_baseline_pure_byte/15_epochs.pt",
            "20_epoch": "out_baseline_pure_byte/20_epochs.pt",
            "best": "out_baseline_pure_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_full_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "type": "mc_25_full",
        "dir": "out_mc_full_pos_weighted_byte",
        "checkpoints": {
            "15_epoch": "out_mc_full_pos_weighted_byte/15_epochs.pt",
            "20_epoch": "out_mc_full_pos_weighted_byte/20_epochs.pt",
            "best": "out_mc_full_pos_weighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_full_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "type": "mc_25_full",
        "dir": "out_mc_full_pos_unweighted_byte",
        "checkpoints": {
            "15_epoch": "out_mc_full_pos_unweighted_byte/15_epochs.pt",
            "20_epoch": "out_mc_full_pos_unweighted_byte/20_epochs.pt",
            "best": "out_mc_full_pos_unweighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_coarse_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "type": "mc_25_coarse",
        "dir": "out_mc_coarse_pos_weighted_byte",
        "checkpoints": {
            "15_epoch": "out_mc_coarse_pos_weighted_byte/15_epochs.pt",
            "20_epoch": "out_mc_coarse_pos_weighted_byte/20_epochs.pt",
            "best": "out_mc_coarse_pos_weighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_coarse_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "type": "mc_25_coarse",
        "dir": "out_mc_coarse_pos_unweighted_byte",
        "checkpoints": {
            "15_epoch": "out_mc_coarse_pos_unweighted_byte/15_epochs.pt",
            "20_epoch": "out_mc_coarse_pos_unweighted_byte/20_epochs.pt",
            "best": "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt",
        },
    },
]


def eval_single_checkpoint(task):
    m_info, ckpt_key, ckpt_rel_path, tf_samples, gen_samples, canon_refs = task
    m_id = m_info["id"]
    ckpt_path = os.path.join(REPO_ROOT, ckpt_rel_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=device)
    val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
    iter_num = int(ckpt_dict.get("iter_num", 0))

    evaluator = TranslationEvaluator(m_info, ckpt_dict=ckpt_dict, device=device)
    t_eval = evaluator.evaluate_teacher_forced(model, tf_samples)
    hyps, gen_lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)
    canon_hyps = [canonicalize(h) for h in hyps]

    filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
    filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]

    bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
    chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

    res = {
        "path": ckpt_rel_path,
        "iter_num": iter_num,
        "train_val_loss": val_loss,
        "bpj": t_eval["bpj"],
        "target_loss": t_eval["target_loss"],
        "target_ppl": t_eval["target_ppl"],
        "bleu": float(bleu),
        "chrf": float(chrf),
        "latency_ms": float(gen_lat_ms),
        "demo_pred": hyps[0] if hyps else "",
    }
    print(f"DONE: {m_id} | {ckpt_key} | BPJ={res['bpj']:.4f} | BLEU={res['bleu']:.2f} | chrF={res['chrf']:.2f}")
    return m_id, ckpt_key, res


def main():
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    tf_samples = all_samples[:500]
    gen_samples = all_samples[:100]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    tasks = []
    for m_info in MODELS_TO_UPDATE:
        for ckpt_key, ckpt_rel_path in m_info["checkpoints"].items():
            tasks.append((m_info, ckpt_key, ckpt_rel_path, tf_samples, gen_samples, canon_refs))

    print(f"Running {len(tasks)} checkpoint evaluations across 4 parallel GPU workers...")
    json_path = os.path.join(REPO_ROOT, "results_seq2seq_translation_all_checkpoints.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = [executor.submit(eval_single_checkpoint, t) for t in tasks]
        for fut in as_completed(futures):
            m_id, ckpt_key, res = fut.result()
            if m_id not in all_results:
                all_results[m_id] = {"checkpoints": {}}
            all_results[m_id]["checkpoints"][ckpt_key] = res
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("All 21 missing checkpoints evaluated and saved to results_seq2seq_translation_all_checkpoints.json!")


if __name__ == "__main__":
    main()
