#!/usr/bin/env python3
"""Train and evaluate the Unweighted Output-Only Target Lane Hangul Factorizer (`w=1.0` across all 25 lanes,
`output_only_lanes=[24]`) for both Coarse POS and Full POS across 20 epochs.
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
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

import sacrebleu
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from run_charhead_h11_20ep_training_and_eval import (
    COMMON_LANES,
    CHECKPOINT_MILESTONES,
    prepare_warmstart_25lane_checkpoint,
)
from run_kobalt_cute_hangul_eval import (
    TASK_DATASETS,
    load_gpt_model,
    score_candidate_mc24,
)
from run_seq2seq_translation_benchmark import (
    TranslationEvaluator,
    canonicalize,
    load_model_from_ckpt,
)
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
UNWEIGHTED_WEIGHTS = [1.0] * 25

UNWEIGHTED_EXPERIMENTS = {
    "nochar_coarse_pos_charhead_unweighted": {
        "display_name": "Output-Only Char Head + Coarse POS (Unweighted: w=1.0 across all 25 lanes)",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_unweighted",
        "src_24lane_dir": "./out_mc_nochar_coarse_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "pos_mode": "coarse",
        "type": "mc_24_coarse",
    },
    "nochar_full_pos_charhead_unweighted": {
        "display_name": "Output-Only Char Head + Full POS (Unweighted: w=1.0 across all 25 lanes)",
        "out_dir": "./out_mc_nochar_full_pos_charhead_unweighted",
        "src_24lane_dir": "./out_mc_nochar_full_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "pos_mode": "full",
        "type": "mc_24_full",
    },
}


def train_20ep_unweighted():
    for exp_id, cfg in UNWEIGHTED_EXPERIMENTS.items():
        out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
        os.makedirs(out_dir, exist_ok=True)
        src_dir = os.path.join(REPO_ROOT, cfg["src_24lane_dir"])

        lanes = COMMON_LANES + [cfg["pos_lane"], "korean_pos_mc_nochar/char"]
        loss_weights_str = " ".join(str(w) for w in UNWEIGHTED_WEIGHTS)

        best_loss = float("inf")
        best_ckpt_path = os.path.join(out_dir, "best_ckpt.pt")

        for tag, fname in CHECKPOINT_MILESTONES:
            dst_ckpt = os.path.join(out_dir, fname)
            if os.path.exists(dst_ckpt):
                print(f"[Skip Train] {exp_id} @ {fname} already exists.")
                ckpt_data = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
                val_loss = float(ckpt_data.get("best_val_loss", 999.0))
                if val_loss < best_loss:
                    best_loss = val_loss
                    shutil.copy2(dst_ckpt, best_ckpt_path)
                continue

            src_ckpt = os.path.join(src_dir, fname)
            if not os.path.exists(src_ckpt):
                src_ckpt = os.path.join(src_dir, "best_ckpt.pt")

            warmstart_ckpt = os.path.join(out_dir, "ckpt.pt")
            base_iter = prepare_warmstart_25lane_checkpoint(src_ckpt, warmstart_ckpt)
            target_iters = base_iter + 150

            cmd = [
                sys.executable,
                os.path.join(REPO_ROOT, "train.py"),
                "--out_dir", out_dir,
                "--dataset", "korean_pos_mc_nochar/script_bpe",
                "--training_mode", "multicontext",
                "--init_from", "resume",
                "--multicontext",
                "--multicontext_datasets", *lanes,
                "--output_only_datasets", "korean_pos_mc_nochar/char",
                "--multicontext_loss_weights", *loss_weights_str.split(),
                "--batch_size", "64",
                "--block_size", "256",
                "--max_iters", str(target_iters),
                "--eval_interval", "150",
                "--eval_iters", "40",
                "--learning_rate", "3e-4",
                "--always_save_checkpoint",
                "--no-compile",
            ]
            print(f"\n>>> Training {exp_id} milestone {fname} (iters {base_iter} -> {target_iters})...")
            subprocess.run(cmd, check=True)

            latest_ckpt = os.path.join(out_dir, "ckpt.pt")
            shutil.copy2(latest_ckpt, dst_ckpt)
            ckpt_data = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
            val_loss = float(ckpt_data.get("best_val_loss", 999.0))
            print(f"    Saved {dst_ckpt} | val_loss = {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                shutil.copy2(dst_ckpt, best_ckpt_path)

        print(f"==> Best checkpoint for {exp_id}: {best_ckpt_path} (best_val_loss={best_loss:.4f})")


def evaluate_unweighted_best_checkpoints():
    out_json_path = os.path.join(REPO_ROOT, "results_charhead_h11_20ep_all_metrics.json")
    full_results: Dict[str, Any] = {}
    if os.path.exists(out_json_path):
        with open(out_json_path, "r", encoding="utf-8") as f:
            full_results = json.load(f)

    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")

    test_path = os.path.join(REPO_ROOT, "data/korean_seq2seq_bench/test.jsonl")
    tf_samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tf_samples.append(json.loads(line))
    tf_samples = tf_samples[:100]
    gen_samples = tf_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    for exp_key, cfg in UNWEIGHTED_EXPERIMENTS.items():
        if exp_key in full_results and "best" in full_results[exp_key]:
            print(f"[Skip Eval] {exp_key} @ best already evaluated.")
            continue

        full_results[exp_key] = {}
        out_dir = cfg["out_dir"]
        tok_24 = tok_24_coarse if cfg["pos_mode"] == "coarse" else tok_24_full
        ckpt_path = os.path.join(REPO_ROOT, out_dir, "best_ckpt.pt")
        ep_name = "best"
        print(f"\n==========================================================================")
        print(f" EVALUATING: {exp_key} @ {ep_name} ({ckpt_path})")
        print("==========================================================================")
        entry: Dict[str, Any] = {}

        # 1. Four Capabilities
        t4 = f"/tmp/temp_unw_4cap_{exp_key}_{ep_name}.json"
        subprocess.run(
            [
                sys.executable,
                "benchmarks/run_four_capability_evals.py",
                "--ckpts", f"{ep_name}:{ckpt_path}",
                "--device", DEVICE,
                "--max_examples", "100",
                "--output_json", t4,
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(t4):
            with open(t4) as f:
                entry["four_capabilities"] = json.load(f).get(ep_name, {})
            os.remove(t4)

        # 2. Ko-HellaSwag
        ths = f"/tmp/temp_unw_hs_{exp_key}_{ep_name}.json"
        subprocess.run(
            [
                sys.executable,
                "benchmarks/run_ko_hellaswag.py",
                "--ckpt_path", ckpt_path,
                "--device", DEVICE,
                "--eval_all_norms",
                "--max_examples", "100",
                "--output_json", ths,
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(ths):
            with open(ths) as f:
                entry["ko_hellaswag"] = json.load(f).get("accuracies", {})
            os.remove(ths)

        # 3. Seq2Seq Translation Benchmark
        model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
        val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
        iter_num = int(ckpt_dict.get("iter_num", 0))
        evaluator = TranslationEvaluator({"id": exp_key, "type": cfg["type"]}, ckpt_dict=ckpt_dict, device=DEVICE)
        t_eval = evaluator.evaluate_teacher_forced(model, tf_samples)
        hyps, gen_lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)
        canon_hyps = [canonicalize(h) for h in hyps]
        filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
        filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]
        bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
        chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score
        entry["seq2seq"] = {
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

        # 4. KoBALT & CUTE-Hangul Diagnostic Evaluation Suite
        kobalt_scores = {}
        total_correct = 0
        total_items = 0
        for task_name, items in TASK_DATASETS.items():
            correct = 0
            for item in items:
                ctx = item["ctx"]
                cands = item["candidates"]
                ans = item["answer"]
                cand_scores = [score_candidate_mc24(model, tok_24, ctx, c, DEVICE) for c in cands]
                pred = int(max(range(len(cand_scores)), key=lambda i: cand_scores[i]))
                if pred == ans:
                    correct += 1
            acc = correct / max(1, len(items))
            kobalt_scores[task_name] = {
                "accuracy": round(acc * 100.0, 2),
                "correct": correct,
                "total": len(items),
            }
            total_correct += correct
            total_items += len(items)
        kobalt_scores["overall_accuracy"] = round((total_correct / max(1, total_items)) * 100.0, 2)
        entry["kobalt_cute_hangul"] = kobalt_scores

        print(
            f"  -> {exp_key} @ {ep_name} | Seq2Seq BLEU/chrF: {bleu:.2f}/{chrf:.2f} | KoBALT Acc: {kobalt_scores['overall_accuracy']:.1f}%"
        )

        full_results[exp_key][ep_name] = entry
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

    print("Saved unweighted charhead best metrics to:", out_json_path)


def main():
    train_20ep_unweighted()
    evaluate_unweighted_best_checkpoints()


if __name__ == "__main__":
    main()
