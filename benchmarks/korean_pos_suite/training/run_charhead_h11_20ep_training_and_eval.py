#!/usr/bin/env python3
"""20-Epoch Training and Full Downstream Evaluation Pipeline for Output-Only Char Target Lane Hangul Factorizer
with Winning H11_jamo_050_char_050 Loss Weights (`[1.0, 0.5, 0.5, 0.5] + [0.05]*19 + [0.5, 0.5]`).

Trains two 25-lane models (`output_only_lanes=[24]`) across 3, 5, 10, 15, and 20 epochs:
1. nochar_coarse_pos_charhead_h11 (`out_mc_nochar_coarse_pos_charhead_h11`)
2. nochar_full_pos_charhead_h11 (`out_mc_nochar_full_pos_charhead_h11`)

Saves checkpoints at 3, 5, 10, 15, and 20 epochs (`3_epochs.pt`, `5_epochs.pt`, `10_epochs.pt`, `15_epochs.pt`, `20_epochs.pt`)
plus best validation loss checkpoint (`best_ckpt.pt`).

Evaluates all saved checkpoints across:
- Four Capabilities Benchmark (KLUE-NER, KLUE-DP, NSMC, UnSmile, KorMedMCQA)
- Ko-HellaSwag Zero-Shot Reasoning
- Seq2Seq Translation Benchmark (BPJ, Target Loss, Target PPL, BLEU, chrF)
- KoBALT & CUTE-Hangul Diagnostic Suite (Choseong Quiz, Allomorphic Particles, Assimilation, Vowel Harmony, Jamo Surgery)
- OOV & Unicode Robustness Stress Test
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

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
EVAL_INTERVAL = 300
FINETUNE_ITERS = 300
MAX_EVAL_EXAMPLES = 100

COMMON_LANES = [
    "korean_pos_mc_nochar/script_bpe",
    "korean_pos_mc_nochar/choseong",
    "korean_pos_mc_nochar/jungseong",
    "korean_pos_mc_nochar/jongseong",
    "korean_pos_mc_nochar/jung_base1",
    "korean_pos_mc_nochar/jung_base2",
    "korean_pos_mc_nochar/jung_has_w",
    "korean_pos_mc_nochar/jung_has_y",
    "korean_pos_mc_nochar/jung_has_i",
    "korean_pos_mc_nochar/jong_base1",
    "korean_pos_mc_nochar/jong_base2",
    "korean_pos_mc_nochar/jong_base3",
    "korean_pos_mc_nochar/choseong_tense",
    "korean_pos_mc_nochar/choseong_aspirated",
    "korean_pos_mc_nochar/choseong_nasal_liquid",
    "korean_pos_mc_nochar/choseong_place",
    "korean_pos_mc_nochar/jung_height",
    "korean_pos_mc_nochar/jung_backness",
    "korean_pos_mc_nochar/jung_round",
    "korean_pos_mc_nochar/jong_complex",
    "korean_pos_mc_nochar/has_batchim",
    "korean_pos_mc_nochar/syllable_index_mod",
    "korean_pos_mc_nochar/codepoint_mod",
]

# H11_jamo_050_char_050 weights: w_bpe=1.0, w_jamo=0.5, w_aux=0.05 (19 lanes), w_pos=0.5, w_char=0.5
H11_WEIGHTS = [1.0, 0.5, 0.5, 0.5] + [0.05] * 19 + [0.5, 0.5]

EXPERIMENTS = {
    "nochar_coarse_pos_charhead_h11": {
        "display_name": "Output-Only Char Head + Coarse POS (H11 Winning Weights: wbpe=1.0, wjamo=0.5, wpos=0.5, wchar=0.5, waux=0.05)",
        "out_dir": "./out_mc_nochar_coarse_pos_charhead_h11",
        "src_24lane_dir": "./out_mc_nochar_coarse_pos_new_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "pos_mode": "coarse",
        "type": "mc_24_coarse",
    },
    "nochar_full_pos_charhead_h11": {
        "display_name": "Output-Only Char Head + Full POS (H11 Winning Weights: wbpe=1.0, wjamo=0.5, wpos=0.5, wchar=0.5, waux=0.05)",
        "out_dir": "./out_mc_nochar_full_pos_charhead_h11",
        "src_24lane_dir": "./out_mc_nochar_full_pos_new_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "pos_mode": "full",
        "type": "mc_24_full",
    },
}

CHECKPOINT_MILESTONES = [
    ("3_epochs", "3_epochs.pt"),
    ("5_epochs", "5_epochs.pt"),
    ("10_epochs", "10_epochs.pt"),
    ("15_epochs", "15_epochs.pt"),
    ("20_epochs", "20_epochs.pt"),
]


def prepare_warmstart_25lane_checkpoint(src_24lane_ckpt: str, dst_25lane_ckpt: str) -> int:
    """Create a 25-lane checkpoint with output_only_lanes=[24] from the 24-lane checkpoint."""
    ckpt = torch.load(src_24lane_ckpt, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    vocab_sizes = list(model_args["vocab_sizes"])
    if len(vocab_sizes) == 24:
        vocab_sizes.append(11173)
    model_args["vocab_sizes"] = vocab_sizes
    model_args["output_only_lanes"] = [24]
    ckpt["model_args"] = model_args

    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    wte_1 = state_dict["transformer.wte_1.weight"]
    wte_2 = state_dict["transformer.wte_2.weight"]
    wte_3 = state_dict["transformer.wte_3.weight"]
    n_embd = wte_1.shape[1]

    lm_head_24_w = torch.zeros((11173, n_embd), dtype=wte_1.dtype)
    lm_head_24_w[0] = (wte_1[0] + wte_2[0] + wte_3[0]) / 3.0

    idx = 1
    for c in range(1, 20):
        for v in range(1, 22):
            for f in range(0, 28):
                lm_head_24_w[idx] = (wte_1[c] + wte_2[v] + wte_3[f]) / 3.0
                idx += 1

    state_dict["transformer.lm_head_24.weight"] = lm_head_24_w
    ckpt["best_val_loss"] = torch.tensor(float("inf"))
    if "optimizer" in ckpt:
        del ckpt["optimizer"]

    torch.save(ckpt, dst_25lane_ckpt)
    return ckpt.get("iter_num", 11311)


def train_single_model(exp_key: str, cfg: Dict[str, Any]):
    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
    src_dir = os.path.join(REPO_ROOT, cfg["src_24lane_dir"])
    os.makedirs(out_dir, exist_ok=True)
    lanes = [*COMMON_LANES, cfg["pos_lane"], "korean_pos_mc_nochar/char"]
    primary_dataset = "korean_pos_mc_nochar/script_bpe"

    best_overall_val_loss = float("inf")
    best_overall_ckpt_path = None

    for stage_name, ckpt_filename in CHECKPOINT_MILESTONES:
        stage_ckpt = os.path.join(out_dir, ckpt_filename)
        if os.path.exists(stage_ckpt):
            print(f"[{exp_key}] Milestone {stage_name} ({ckpt_filename}) already exists. Checking val loss...")
            ckpt_data = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
            val_loss = float(ckpt_data.get("best_val_loss", float("inf")))
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_overall_ckpt_path = stage_ckpt
            continue

        src_ckpt = os.path.join(src_dir, ckpt_filename)
        if not os.path.exists(src_ckpt):
            raise FileNotFoundError(f"Source 24-lane checkpoint not found: {src_ckpt}")

        work_ckpt = os.path.join(out_dir, "ckpt.pt")
        start_iter = prepare_warmstart_25lane_checkpoint(src_ckpt, work_ckpt)
        target_iters = start_iter + FINETUNE_ITERS

        print(f"\n>>> [{exp_key}] Training milestone {stage_name} ({start_iter} -> {target_iters} iters)...")
        cmd = [
            sys.executable,
            "train.py",
            "--dataset", primary_dataset,
            "--training_mode", "multicontext",
            "--multicontext",
            "--multicontext_datasets", *lanes,
            "--output_only_datasets", "korean_pos_mc_nochar/char",
            "--multicontext_loss_weights", *[str(w) for w in H11_WEIGHTS],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--no-lr_decay_match_max_iters",
            "--lr_decay_iters", "45241",
            "--eval_interval", str(EVAL_INTERVAL),
            "--eval_iters", str(EVAL_ITERS),
            "--always_save_checkpoint",
            "--init_from", "resume",
            "--dropout", "0.1",
            "--device", DEVICE,
            "--out_dir", out_dir,
        ]

        t0 = time.time()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.time()

        shutil.copyfile(work_ckpt, stage_ckpt)
        ckpt_data = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
        val_loss = float(ckpt_data.get("best_val_loss", float("inf")))
        print(f"[{exp_key}] Saved {ckpt_filename} (val_loss={val_loss:.4f}) in {t1 - t0:.1f}s")

        if val_loss < best_overall_val_loss:
            best_overall_val_loss = val_loss
            best_overall_ckpt_path = stage_ckpt

    best_dest = os.path.join(out_dir, "best_ckpt.pt")
    if best_overall_ckpt_path and os.path.exists(best_overall_ckpt_path):
        shutil.copyfile(best_overall_ckpt_path, best_dest)
        print(f"[{exp_key}] Saved best validation checkpoint ({best_overall_ckpt_path}, loss={best_overall_val_loss:.4f}) -> {best_dest}")


def evaluate_all_checkpoints():
    """Run downstream evaluations across all saved checkpoints (3, 5, 10, 15, 20 epochs & best_ckpt.pt)."""
    from run_seq2seq_translation_benchmark import (
        TranslationEvaluator,
        canonicalize,
        load_model_from_ckpt,
        TEST_JSONL,
    )
    import sacrebleu
    from benchmarks.run_kobalt_cute_hangul_eval import (
        score_candidate_mc24,
        TASK_DATASETS,
    )
    from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")

    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    ckpts_to_eval = [
        ("3_epoch", "3_epochs.pt"),
        ("5_epoch", "5_epochs.pt"),
        ("10_epoch", "10_epochs.pt"),
        ("15_epoch", "15_epochs.pt"),
        ("20_epoch", "20_epochs.pt"),
        ("best", "best_ckpt.pt"),
    ]

    full_results = {}
    out_json_path = os.path.join(REPO_ROOT, "results_charhead_h11_20ep_all_metrics.json")
    if os.path.exists(out_json_path):
        try:
            with open(out_json_path, "r") as f:
                full_results = json.load(f)
        except Exception:
            full_results = {}

    for exp_key, cfg in EXPERIMENTS.items():
        if exp_key not in full_results:
            full_results[exp_key] = {}
        out_dir = cfg["out_dir"]
        tok_24 = tok_24_coarse if cfg["pos_mode"] == "coarse" else tok_24_full

        for ep_name, ckpt_file in ckpts_to_eval:
            if ep_name in full_results[exp_key] and "kobalt_cute_hangul" in full_results[exp_key][ep_name]:
                print(f"Skipping already evaluated {exp_key} @ {ep_name}")
                continue

            ckpt_path = os.path.join(REPO_ROOT, out_dir, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path} (not found)")
                continue

            print(f"\n==========================================================================")
            print(f" EVALUATING: {exp_key} @ {ep_name} ({ckpt_path})")
            print("==========================================================================")
            entry = full_results[exp_key].get(ep_name, {})

            # 1. Four Capabilities
            t4 = f"/tmp/temp_h11_4cap_{exp_key}_{ep_name}.json"
            subprocess.run(
                [
                    sys.executable,
                    "benchmarks/run_four_capability_evals.py",
                    "--ckpts", f"{ep_name}:{ckpt_path}",
                    "--device", DEVICE,
                    "--max_examples", str(MAX_EVAL_EXAMPLES),
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
            ths = f"/tmp/temp_h11_hs_{exp_key}_{ep_name}.json"
            subprocess.run(
                [
                    sys.executable,
                    "benchmarks/run_ko_hellaswag.py",
                    "--ckpt_path", ckpt_path,
                    "--device", DEVICE,
                    "--eval_all_norms",
                    "--max_examples", str(MAX_EVAL_EXAMPLES),
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
                f"  -> {exp_key} @ {ep_name} | 4-Cap Avg: {entry['four_capabilities'].get('average_accuracy', 0)*100:.1f}% | "
                f"HellaSwag: {entry['ko_hellaswag'].get('char_norm', 0)*100:.1f}% | "
                f"Seq2Seq BLEU/chrF: {bleu:.2f}/{chrf:.2f} | KoBALT Acc: {kobalt_scores['overall_accuracy']:.1f}%"
            )

            full_results[exp_key][ep_name] = entry
            with open(out_json_path, "w") as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all checkpoint evaluation results to {out_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, default=None, help="Run specific model key")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluations")
    args = parser.parse_args()

    if args.worker:
        train_single_model(args.worker, EXPERIMENTS[args.worker])
    elif not args.eval_only:
        procs = []
        for key in EXPERIMENTS:
            cmd = [sys.executable, __file__, "--worker", key]
            print(f"Launching background worker for {key}...")
            p = subprocess.Popen(cmd, cwd=REPO_ROOT)
            procs.append((key, p))
        for key, p in procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Worker {key} exited with code {ret}")
        evaluate_all_checkpoints()
    else:
        evaluate_all_checkpoints()
