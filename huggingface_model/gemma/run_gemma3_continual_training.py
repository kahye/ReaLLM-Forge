#!/usr/bin/env python3
"""Unified Continual Pretraining Pipeline for Gemma 3 270M Backbone across 6 Tokenizer Architectures.

Supports:
1. Baseline Character Tokenizer (V=4,924)
2. Baseline Byte-Fallback BPE (V=4,924)
3. Baseline Pure Byte Tokenizer (V=256)
4. Hangul Factorizer + Character Stream & Byte Fallback (25 streams, coarse POS, weighted loss)
5. No-Char Hangul Factorizer + Byte-Fallback BPE (24 streams, coarse POS, winning C06 weights)
6. Output-Only Target Lane Hangul Factorizer + Byte-Fallback BPE (25 streams, coarse POS, winning H11 weights)

Features:
- Continual pretraining from official Gemma 3 270M weights (18 layers, d=640, GQA, RoPE, SDPA)
- Fused Head Projections for deployment (fusing linear heads into single matrix for kernel speedup)
- Saves 5, 10, 15, 20 epoch and best validation loss checkpoints
- Downstream evaluations:
  - KoBALT & CUTE-Hangul 5 diagnostic tasks
  - Seq2Seq English-to-Korean translation benchmark (BPJ, Target Loss, Target PPL, BLEU-4, chrF)
- Generates comprehensive markdown comparison report and JSON results.
"""
from __future__ import annotations
import os, os as _os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

import argparse
import copy
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import sacrebleu
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)

from gpt_conf import GPTConfig
from gemma3_continual_model import Gemma3ContinualModel
from sample import get_tokenizer_functions
from hangul_factorizer import HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
EVAL_INTERVAL = 1000

COMMON_LANES_MC = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
]

COMMON_LANES_NOCHAR = [
    "korean_pos_mc_nochar/script_bpe", "korean_pos_mc_nochar/choseong", "korean_pos_mc_nochar/jungseong", "korean_pos_mc_nochar/jongseong",
    "korean_pos_mc_nochar/jung_base1", "korean_pos_mc_nochar/jung_base2", "korean_pos_mc_nochar/jung_has_w", "korean_pos_mc_nochar/jung_has_y",
    "korean_pos_mc_nochar/jung_has_i", "korean_pos_mc_nochar/jong_base1", "korean_pos_mc_nochar/jong_base2", "korean_pos_mc_nochar/jong_base3",
    "korean_pos_mc_nochar/choseong_tense", "korean_pos_mc_nochar/choseong_aspirated", "korean_pos_mc_nochar/choseong_nasal_liquid",
    "korean_pos_mc_nochar/choseong_place", "korean_pos_mc_nochar/jung_height", "korean_pos_mc_nochar/jung_backness", "korean_pos_mc_nochar/jung_round",
    "korean_pos_mc_nochar/jong_complex", "korean_pos_mc_nochar/has_batchim", "korean_pos_mc_nochar/syllable_index_mod", "korean_pos_mc_nochar/codepoint_mod",
]

C06_WEIGHTS = [1.0, 1.0, 1.0, 1.0] + [0.05] * 19 + [0.5]
H11_WEIGHTS = [1.0, 0.5, 0.5, 0.5] + [0.05] * 19 + [0.5, 0.5]

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_character": {
        "name": "Baseline Character Tokenizer (V=4,924)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_baseline_char"),
        "dataset": "korean_pos_mc/char",
        "training_mode": "single",
        "multicontext": False,
        "type": "single",
        "eval_type": "baseline_char",
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "baseline_char_bpe": {
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_baseline_char_bpe"),
        "dataset": "korean_pos_mc/char_bpe",
        "training_mode": "single",
        "multicontext": False,
        "type": "single",
        "eval_type": "baseline_bpe",
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "baseline_pure_byte": {
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_baseline_pure_byte"),
        "dataset": "korean_pos_mc/byte",
        "training_mode": "single",
        "multicontext": False,
        "type": "single",
        "eval_type": "baseline_byte",
        "iters_5ep": 28105,
        "iters_10ep": 56210,
        "iters_15ep": 84315,
        "iters_20ep": 112434,
    },
    "coarse_pos_weighted_byte": {
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_mc_coarse_pos_weighted_byte"),
        "dataset": "korean_pos_mc/char_byte",
        "training_mode": "multicontext",
        "multicontext": True,
        "type": "mc_25",
        "eval_type": "mc_25",
        "lanes": [*COMMON_LANES_MC, "korean_pos_mc/pos", "korean_pos_mc/char_byte"],
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "iters_5ep": 18165,
        "iters_10ep": 36330,
        "iters_15ep": 54495,
        "iters_20ep": 72660,
    },
    "nochar_coarse_pos_new_weighted": {
        "name": "No-Char Hangul Factorizer + Byte-Fallback BPE (C06 Winning Weights)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_mc_nochar_coarse_pos_new_weighted"),
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "multicontext": True,
        "type": "mc_24",
        "eval_type": "mc_24",
        "lanes": [*COMMON_LANES_NOCHAR, "korean_pos_mc_nochar/pos"],
        "loss_args": ["--multicontext_loss_weights", *[str(w) for w in C06_WEIGHTS]],
        "iters_5ep": 11310,
        "iters_10ep": 22620,
        "iters_15ep": 33930,
        "iters_20ep": 45240,
    },
    "nochar_coarse_pos_charhead_h11": {
        "name": "Output-Only Char Target Lane Hangul Factorizer + Byte-Fallback BPE (H11 Winning Weights)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_mc_nochar_coarse_pos_charhead_h11"),
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "multicontext": True,
        "type": "mc_25_charhead",
        "eval_type": "mc_24",
        "lanes": [*COMMON_LANES_NOCHAR, "korean_pos_mc_nochar/pos", "korean_pos_mc_nochar/char"],
        "output_only_lanes": [24],
        "loss_args": ["--output_only_lanes", "24", "--multicontext_loss_weights", *[str(w) for w in H11_WEIGHTS]],
        "iters_5ep": 11310,
        "iters_10ep": 22620,
        "iters_15ep": 33930,
        "iters_20ep": 45240,
    },
    "nochar_full_pos_charhead_h11": {
        "name": "Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)",
        "out_dir": os.path.join(REPO_ROOT, "out_gemma3_mc_nochar_full_pos_charhead_h11"),
        "dataset": "korean_pos_mc_nochar/script_bpe",
        "training_mode": "multicontext",
        "multicontext": True,
        "type": "mc_25_charhead_full",
        "eval_type": "mc_24",
        "pos_mode": "full",
        "lanes": [*COMMON_LANES_NOCHAR, "korean_pos_mc_nochar/pos_full", "korean_pos_mc_nochar/char"],
        "output_only_lanes": [24],
        "loss_args": ["--output_only_lanes", "24", "--multicontext_loss_weights", *[str(w) for w in H11_WEIGHTS]],
        "iters_5ep": 11310,
        "iters_10ep": 22620,
        "iters_15ep": 33930,
        "iters_20ep": 45240,
    },
}

STAGES = [
    ("5_epochs", "iters_5ep", "5_epochs.pt"),
    ("10_epochs", "iters_10ep", "10_epochs.pt"),
    ("15_epochs", "iters_15ep", "15_epochs.pt"),
    ("20_epochs", "iters_20ep", "20_epochs.pt"),
]


def fuse_and_export_checkpoint(ckpt_path: str, fused_path: str) -> None:
    """Load checkpoint, apply fuse_lm_heads() to combine multicontext heads into W_fused, and save."""
    if not os.path.exists(ckpt_path):
        return
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = Gemma3ContinualModel.from_checkpoint_dict(ckpt)
        model.fuse_lm_heads()
        fused_ckpt = copy.deepcopy(ckpt)
        fused_ckpt["model"] = model.state_dict()
        fused_ckpt["model_args"]["fused_lm_head"] = True
        torch.save(fused_ckpt, fused_path)
        print(f"Exported Fused Deployment Checkpoint: {fused_path}")
    except Exception as e:
        print(f"Warning fusing checkpoint {ckpt_path}: {e}")


def train_model_stages(model_key: str, cfg: Dict[str, Any], selected_stages: Optional[List[str]] = None) -> bool:
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    prev_ckpt = os.path.join(out_dir, "ckpt.pt")
    trained_any = False

    print("\n" + "=" * 80)
    print(f" TRAINING: {model_key.upper()} ({cfg['name']})")
    print(f" Output Directory: {out_dir}")
    print(f" Target 20 Epochs: {cfg['iters_20ep']:,} iterations")
    print("=" * 80)

    for stage_name, target_key, ckpt_filename in STAGES:
        if selected_stages and stage_name not in selected_stages:
            continue

        if stage_name == "20_epochs":
            ckpt_15_path = os.path.join(out_dir, "15_epochs.pt")
            if os.path.exists(ckpt_15_path):
                is_decreasing = False
                iters_10ep = cfg.get("iters_10ep", 0)
                best_iter = 0
                best_loss = None
                try:
                    d15 = torch.load(ckpt_15_path, map_location="cpu", weights_only=False)
                    best_iter = d15.get("best_iter", 0)
                    best_loss = d15.get("best_val_loss")
                    if hasattr(best_loss, "item"):
                        best_loss = best_loss.item()
                    if best_iter >= iters_10ep:
                        is_decreasing = True
                except Exception as e:
                    print(f"[{model_key}] Error inspecting 15ep checkpoint: {e}")
                if not is_decreasing:
                    print(f"\n[{model_key}] 15-epoch milestone indicates validation loss did not decrease (best_iter={best_iter:,} < 10ep milestone {iters_10ep:,}, loss={best_loss}). Skipping 20_epochs.\n")
                    break

        target_iters = cfg[target_key]
        stage_ckpt = os.path.join(out_dir, ckpt_filename)

        if os.path.exists(stage_ckpt):
            try:
                d = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                if d.get("iter_num", 0) >= target_iters:
                    print(f"[{model_key}] Stage {stage_name} ({ckpt_filename}) already at {d.get('iter_num')} >= {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        if os.path.exists(prev_ckpt):
            try:
                d = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
                curr = d.get("iter_num", 0)
                if curr >= target_iters:
                    if not os.path.exists(stage_ckpt):
                        shutil.copyfile(prev_ckpt, stage_ckpt)
                    print(f"[{model_key}] Stage {stage_name} already reached in ckpt.pt ({curr} >= {target_iters}). Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{model_key}] Running {stage_name} up to iter {target_iters:,}...")

        cmd = [
            sys.executable,
            "train.py",
            "--model_backbone", "gemma3_270m",
            "--dataset", cfg["dataset"],
            "--training_mode", cfg["training_mode"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", str(cfg["iters_20ep"]),
            "--eval_interval", str(EVAL_INTERVAL),
            "--eval_iters", str(EVAL_ITERS),
            "--always_save_checkpoint",
            "--dropout", "0.1",
            "--dtype", "bfloat16",
            "--device", DEVICE,
            "--out_dir", out_dir,
        ]

        if cfg["multicontext"]:
            cmd.extend(["--multicontext", "--multicontext_datasets", *cfg["lanes"]])
            if "loss_args" in cfg:
                cmd.extend(cfg["loss_args"])

        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        trained_any = True
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {model_key} at stage {stage_name} (code {res.returncode})")

        if os.path.exists(prev_ckpt):
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"[{model_key}] Saved milestone checkpoint: {stage_ckpt}")

        print(f"[{model_key}] Stage {stage_name} completed in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")

        # Check early termination rule at 15-epoch milestone:
        if stage_name == "15_epochs":
            is_decreasing = False
            best_iter = 0
            best_loss = None
            iters_10ep = cfg.get("iters_10ep", 0)
            ckpt_15_path = os.path.join(out_dir, "15_epochs.pt")

            if os.path.exists(ckpt_15_path):
                try:
                    d15 = torch.load(ckpt_15_path, map_location="cpu", weights_only=False)
                    best_iter = d15.get("best_iter", 0)
                    best_loss = d15.get("best_val_loss")
                    if hasattr(best_loss, "item"):
                        best_loss = best_loss.item()
                    # If best_iter >= iters_10ep, val loss reached a new low during 10ep -> 15ep
                    if best_iter >= iters_10ep:
                        is_decreasing = True
                except Exception as e:
                    print(f"[{model_key}] Note checking 15ep val loss: {e}")

            if not is_decreasing:
                print("\n" + "=" * 80)
                print(f"[{model_key}] *** VALIDATION LOSS IS NOT DECREASING AT 15-EPOCH MILESTONE ***")
                print(f"[{model_key}] Best validation loss was at iter {best_iter:,} < 10ep milestone ({iters_10ep:,}) (loss={best_loss}).")
                print(f"[{model_key}] Terminating training early at 15 epochs as requested. Moving to evaluation & next model.")
                print("=" * 80 + "\n")
                break
            else:
                print(f"\n[{model_key}] Validation loss is still decreasing at 15 epochs (new best at iter {best_iter:,} >= {iters_10ep:,}, loss={best_loss}). Continuing to 20 epochs.\n")

    # Ensure best_ckpt.pt and deployment fused checkpoints exist
    best_ckpt = os.path.join(out_dir, "best_ckpt.pt")
    if not os.path.exists(best_ckpt) and os.path.exists(prev_ckpt):
        shutil.copyfile(prev_ckpt, best_ckpt)

    fused_best = os.path.join(out_dir, "deployment_fused_best_ckpt.pt")
    if os.path.exists(best_ckpt):
        fuse_and_export_checkpoint(best_ckpt, fused_best)

    ckpt_20 = os.path.join(out_dir, "20_epochs.pt")
    fused_20 = os.path.join(out_dir, "deployment_fused_20ep.pt")
    if os.path.exists(ckpt_20):
        fuse_and_export_checkpoint(ckpt_20, fused_20)

    ckpt_15 = os.path.join(out_dir, "15_epochs.pt")
    fused_15 = os.path.join(out_dir, "deployment_fused_15ep.pt")
    if os.path.exists(ckpt_15):
        fuse_and_export_checkpoint(ckpt_15, fused_15)

    return trained_any


def evaluate_all_checkpoints(model_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run KoBALT/CUTE-Hangul and Seq2Seq translation benchmarks on all available checkpoints."""
    from benchmarks.run_kobalt_cute_hangul_eval import (
        score_candidate_single,
        score_candidate_mc24,
        score_candidate_mc25,
        TASK_DATASETS,
    )
    from run_seq2seq_translation_benchmark import (
        load_model_from_ckpt,
        TranslationEvaluator,
        TEST_JSONL,
        canonicalize,
    )

    out_dir = cfg["out_dir"]
    checkpoints = ["5_epochs.pt", "10_epochs.pt", "15_epochs.pt", "20_epochs.pt", "best_ckpt.pt"]

    eval_results: Dict[str, Any] = {}

    trans_samples = []
    if os.path.exists(TEST_JSONL):
        with open(TEST_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trans_samples.append(json.loads(line))

    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(out_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue

        print(f"\n--- [EVAL] {model_key} @ {ckpt_name} ---")
        ckpt_res: Dict[str, Any] = {}

        try:
            model, ckpt_dict = load_model_from_ckpt(ckpt_path, DEVICE)
            model.eval()

            # 1. KoBALT / CUTE-Hangul Tasks
            kobalt_scores = {}
            m_type = cfg["type"]

            if m_type == "single":
                meta_path = os.path.join(out_dir, "meta.pkl")
                if not os.path.exists(meta_path):
                    meta_path = os.path.join(REPO_ROOT, "data", cfg["dataset"], "meta.pkl")
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                encode_fn, _ = get_tokenizer_functions(meta)

                for task_name, items in TASK_DATASETS.items():
                    correct = 0
                    for item in items:
                        scores = [score_candidate_single(model, encode_fn, item["ctx"], c, DEVICE) for c in item["candidates"]]
                        if scores.index(max(scores)) == item["answer"]:
                            correct += 1
                    kobalt_scores[task_name] = round(100.0 * correct / len(items), 2)

            elif m_type in ("mc_24", "mc_25_charhead", "mc_25_charhead_full"):
                pos_mode = cfg.get("pos_mode", "coarse")
                spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
                tok_24 = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode=pos_mode)
                for task_name, items in TASK_DATASETS.items():
                    correct = 0
                    for item in items:
                        scores = [score_candidate_mc24(model, tok_24, item["ctx"], c, DEVICE) for c in item["candidates"]]
                        if scores.index(max(scores)) == item["answer"]:
                            correct += 1
                    kobalt_scores[task_name] = round(100.0 * correct / len(items), 2)

            elif m_type == "mc_25":
                tok_25 = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")
                stois_25 = []
                for lp in cfg["lanes"]:
                    mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
                    with open(mp, "rb") as f:
                        stois_25.append(pickle.load(f)["stoi"])
                for task_name, items in TASK_DATASETS.items():
                    correct = 0
                    for item in items:
                        scores = [score_candidate_mc25(model, tok_25, stois_25, item["ctx"], c, DEVICE) for c in item["candidates"]]
                        if scores.index(max(scores)) == item["answer"]:
                            correct += 1
                    kobalt_scores[task_name] = round(100.0 * correct / len(items), 2)

            kobalt_avg = round(sum(kobalt_scores.values()) / max(1, len(kobalt_scores)), 2)
            kobalt_scores["average"] = kobalt_avg
            ckpt_res["kobalt_cute"] = kobalt_scores
            print(f"KoBALT/CUTE Avg: {kobalt_avg:.2f}% | Detail: {kobalt_scores}")

            # 2. Seq2Seq Translation Benchmark
            if trans_samples:
                evaluator = TranslationEvaluator({"type": cfg.get("eval_type", m_type), "dir": out_dir}, ckpt_dict, device=DEVICE)
                tf_metrics = evaluator.evaluate_teacher_forced(model, trans_samples)
                completions, lat_ms = evaluator.generate_completions(model, trans_samples[:100])

                refs = [canonicalize(s["ko"]) for s in trans_samples[:100]]
                hyps = [canonicalize(h) for h in completions]
                bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
                chrf = sacrebleu.corpus_chrf(hyps, [refs], char_order=18).score

                ckpt_res["seq2seq"] = {
                    "bpj": round(tf_metrics["bpj"], 3),
                    "target_loss": round(tf_metrics["target_loss"], 3),
                    "target_ppl": round(tf_metrics["target_ppl"], 2),
                    "bleu": round(bleu, 2),
                    "chrf": round(chrf, 2),
                    "latency_ms": round(lat_ms, 1),
                }
                print(f"Seq2Seq: BPJ={tf_metrics['bpj']:.3f}, Loss={tf_metrics['target_loss']:.3f}, BLEU={bleu:.2f}, chrF={chrf:.2f}")

        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()

        eval_results[ckpt_name] = ckpt_res

    return eval_results


def build_summary_report(all_results: Dict[str, Any], report_md_path: str, report_json_path: str) -> None:
    """Build and save structured Markdown and JSON comparison report."""
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON: {report_json_path}")

    md = []
    md.append("# Gemma 3 270M Continual Pretraining: Tokenizer Architecture Benchmark Report\n")
    md.append("**Date**: September 2026  ")
    md.append("**Backbone**: Gemma 3 270M (`google/gemma-3-270m`, 18 layers, $d_{\\text{model}}=640$, GQA, dual-frequency RoPE)  ")
    md.append("**Training Corpus**: 59M tokens Korean corpus (20 epochs)  ")
    md.append("**Deployment Optimization**: Fused Head Projections (`fuse_lm_heads()` combining multi-stream heads into contiguous matrix $W_{\\text{fused}}$)\n")
    md.append("---\n")

    md.append("## 1. Downstream Translation & Diagnostic Performance Summary (Best Checkpoints)\n")
    md.append("| Model Architecture | Tokenizer Type | Vocab Size | BPJ ↓ | Target Loss ↓ | BLEU-4 ↑ | chrF ↑ | KoBALT Avg ↑ | Latency (ms) |\n")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for m_key, cfg in MODEL_CONFIGS.items():
        m_res = all_results.get(m_key, {})
        best_res = m_res.get("best_ckpt.pt", m_res.get("20_epochs.pt", {}))
        seq = best_res.get("seq2seq", {})
        kob = best_res.get("kobalt_cute", {})

        if "byte" in m_key and "coarse" not in m_key:
            v_str = "256"
        elif "baseline_char" in m_key:
            v_str = "4,924"
        elif "mc_24" in cfg["type"]:
            v_str = "24 Lanes"
        elif "mc_25" in cfg["type"]:
            v_str = "25 Lanes"
        else:
            v_str = "4,924"
        bpj = f"{seq.get('bpj', '-'):.3f}" if isinstance(seq.get("bpj"), (int, float)) else "-"
        tloss = f"{seq.get('target_loss', '-'):.3f}" if isinstance(seq.get("target_loss"), (int, float)) else "-"
        bleu = f"{seq.get('bleu', '-'):.2f}" if isinstance(seq.get("bleu"), (int, float)) else "-"
        chrf = f"{seq.get('chrf', '-'):.2f}" if isinstance(seq.get("chrf"), (int, float)) else "-"
        k_avg = f"{kob.get('average', '-'):.2f}%" if isinstance(kob.get("average"), (int, float)) else "-"
        lat = f"{seq.get('latency_ms', '-'):.1f}" if isinstance(seq.get("latency_ms"), (int, float)) else "-"

        md.append(f"| **{cfg['name']}** | `{cfg['type']}` | {v_str} | {bpj} | {tloss} | {bleu} | {chrf} | {k_avg} | {lat} |")

    md.append("\n---\n")
    md.append("## 2. Detailed KoBALT & CUTE-Hangul Sub-Syllable Diagnostic Breakdown\n")
    md.append("| Model | Choseong Quiz | Allomorphic Particles | Phonological Assimilation | Vowel Harmony | Jamo Surgery | Overall Average |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for m_key, cfg in MODEL_CONFIGS.items():
        m_res = all_results.get(m_key, {})
        best_res = m_res.get("best_ckpt.pt", m_res.get("20_epochs.pt", {}))
        kob = best_res.get("kobalt_cute", {})

        q1 = f"{kob.get('choseong_quiz', '-'):.1f}%" if isinstance(kob.get("choseong_quiz"), (int, float)) else "-"
        q2 = f"{kob.get('allomorphic_particles', '-'):.1f}%" if isinstance(kob.get("allomorphic_particles"), (int, float)) else "-"
        q3 = f"{kob.get('phonological_assimilation', '-'):.1f}%" if isinstance(kob.get("phonological_assimilation"), (int, float)) else "-"
        val_q4 = kob.get("vowel_harmony_ideophones", kob.get("vowel_harmony"))
        val_q5 = kob.get("cute_jamo_manipulation", kob.get("jamo_surgery"))
        q4 = f"{val_q4:.1f}%" if isinstance(val_q4, (int, float)) else "-"
        q5 = f"{val_q5:.1f}%" if isinstance(val_q5, (int, float)) else "-"
        avg = f"{kob.get('average', '-'):.2f}%" if isinstance(kob.get("average"), (int, float)) else "-"

        md.append(f"| **{cfg['name']}** | {q1} | {q2} | {q3} | {q4} | {q5} | **{avg}** |")

    md.append("\n---\n")
    md.append("## 3. Epoch Progression (5ep, 10ep, 15ep, 20ep)\n")
    for m_key, cfg in MODEL_CONFIGS.items():
        m_res = all_results.get(m_key, {})
        md.append(f"### {cfg['name']}")
        md.append("| Checkpoint | BPJ ↓ | Target Loss ↓ | BLEU-4 ↑ | chrF ↑ | KoBALT Avg ↑ |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for ckpt in ["5_epochs.pt", "10_epochs.pt", "15_epochs.pt", "20_epochs.pt", "best_ckpt.pt"]:
            c_res = m_res.get(ckpt, {})
            seq = c_res.get("seq2seq", {})
            kob = c_res.get("kobalt_cute", {})
            bpj = f"{seq.get('bpj', '-'):.3f}" if isinstance(seq.get("bpj"), (int, float)) else "-"
            tloss = f"{seq.get('target_loss', '-'):.3f}" if isinstance(seq.get("target_loss"), (int, float)) else "-"
            bleu = f"{seq.get('bleu', '-'):.2f}" if isinstance(seq.get("bleu"), (int, float)) else "-"
            chrf = f"{seq.get('chrf', '-'):.2f}" if isinstance(seq.get("chrf"), (int, float)) else "-"
            k_avg = f"{kob.get('average', '-'):.2f}%" if isinstance(kob.get("average"), (int, float)) else "-"
            md.append(f"| `{ckpt}` | {bpj} | {tloss} | {bleu} | {chrf} | {k_avg} |")
        md.append("\n")

    content = "\n".join(md)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved Markdown Report: {report_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Gemma 3 270M Continual Pretraining & Benchmark Pipeline")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), default=None,
                        help="Specific model(s) to run (default: all 6)")
    parser.add_argument("--stages", nargs="+", choices=["5_epochs", "10_epochs", "15_epochs", "20_epochs"], default=None,
                        help="Specific training stages to run (default: all)")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluations without training")
    parser.add_argument("--report_dir", type=str, default="/google/data/rw/users/ka/kahye/gemma_tokens",
                        help="Directory to save final benchmark report and results")
    args = parser.parse_args()

    models_to_run = args.models or list(MODEL_CONFIGS.keys())
    os.makedirs(args.report_dir, exist_ok=True)

    report_json_path = os.path.join(args.report_dir, "gemma3_270m_all_6_models_benchmark_results.json")
    report_md_path = os.path.join(args.report_dir, "gemma3_270m_all_6_models_benchmark_report.md")

    all_results: Dict[str, Any] = {}
    if os.path.exists(report_json_path):
        try:
            with open(report_json_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            pass

    for m_key in models_to_run:
        cfg = MODEL_CONFIGS[m_key]

        trained_new = False
        if not args.eval_only:
            trained_new = train_model_stages(m_key, cfg, selected_stages=args.stages)

        m_res = all_results.get(m_key, {})
        has_eval = "best_ckpt.pt" in m_res and "seq2seq" in m_res.get("best_ckpt.pt", {})

        if not trained_new and has_eval and not args.eval_only:
            print(f"[{m_key}] No new training required and evaluation already exists in results. Skipping redundant evaluation.")
            m_eval = m_res
        else:
            print(f"\nEvaluating checkpoints for {m_key}...")
            m_eval = evaluate_all_checkpoints(m_key, cfg)
            all_results[m_key] = m_eval

        build_summary_report(all_results, report_md_path, report_json_path)

    print("\n" + "=" * 80)
    print(" ALL REQUESTED GEMMA 3 270M RUNS & BENCHMARKS COMPLETED SUCCESSFULLY!")
    print(f" Report: {report_md_path}")
    print(f" Results: {report_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
