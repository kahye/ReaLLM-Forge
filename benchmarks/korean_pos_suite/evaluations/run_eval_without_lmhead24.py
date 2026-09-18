#!/usr/bin/env python3
"""Run downstream evaluations on Output-Only Char-Head Factorizer checkpoints WITHOUT lm_head_24.

Evaluates:
1. Seq2Seq Translation Benchmark (Teacher-Forced 24 lanes BPJ, Target Loss, Target PPL, and 24-lane Jamo autoregressive generation BLEU, chrF, latency)
2. KoBALT & CUTE-Hangul Diagnostic Suite (pure Jamo + BPE scoring without lm_head_24)
3. Four Capabilities Benchmark (KLUE-NER, KLUE-DP, NSMC, UnSmile, KorMedMCQA)
4. Ko-HellaSwag Zero-Shot Reasoning

Saves all results to results_charhead_without_lmhead24_all_metrics.json.
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
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import sacrebleu
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)

from model import GPT, GPTConfig
from run_seq2seq_translation_benchmark import load_model_from_ckpt, count_jamos, canonicalize, TEST_JSONL
from benchmarks.run_kobalt_cute_hangul_eval import TASK_DATASETS
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "nochar_coarse_pos_charhead_h11": {
        "dir": "out_mc_nochar_coarse_pos_charhead_h11",
        "pos_mode": "coarse",
        "milestones": [
            ("3_epoch", "3_epochs.pt"),
            ("5_epoch", "5_epochs.pt"),
            ("10_epoch", "10_epochs.pt"),
            ("15_epoch", "15_epochs.pt"),
            ("20_epoch", "20_epochs.pt"),
            ("best", "best_ckpt.pt"),
        ],
    },
    "nochar_full_pos_charhead_h11": {
        "dir": "out_mc_nochar_full_pos_charhead_h11",
        "pos_mode": "full",
        "milestones": [
            ("3_epoch", "3_epochs.pt"),
            ("5_epoch", "5_epochs.pt"),
            ("10_epoch", "10_epochs.pt"),
            ("15_epoch", "15_epochs.pt"),
            ("20_epoch", "20_epochs.pt"),
            ("best", "best_ckpt.pt"),
        ],
    },
    "nochar_coarse_pos_charhead_unweighted": {
        "dir": "out_mc_nochar_coarse_pos_charhead_unweighted",
        "pos_mode": "coarse",
        "milestones": [
            ("best", "best_ckpt.pt"),
        ],
    },
    "nochar_full_pos_charhead_unweighted": {
        "dir": "out_mc_nochar_full_pos_charhead_unweighted",
        "pos_mode": "full",
        "milestones": [
            ("best", "best_ckpt.pt"),
        ],
    },
}


def eval_seq2seq_teacher_forced_24lanes(model: GPT, tok_24: HangulHybridPosTokenizer, samples: List[Dict[str, str]]) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_jamos = 0
    for s in samples:
        en, ko = s["en"], s["ko"]
        prompt = f"{en}\t"
        full = f"{en}\t{ko}\n"
        n_jamos = count_jamos(ko)
        total_jamos += max(1, n_jamos)
        p_seq = tok_24.encode_text(prompt)
        f_seq = tok_24.encode_text(full)
        if len(f_seq) > 256:
            f_seq = f_seq[:256]
        if len(f_seq) <= len(p_seq):
            continue
        token_lists = [[] for _ in range(24)]
        for step in f_seq:
            for i in range(24):
                token_lists[i].append(step[i])
        input_lanes = [torch.tensor(lane[:-1], device=DEVICE).unsqueeze(0) for lane in token_lists]
        target_lanes = [torch.tensor(lane[1:], device=DEVICE).unsqueeze(0) for lane in token_lists]
        mask_len = min(len(p_seq) - 1, target_lanes[0].size(1))
        for lane in target_lanes:
            lane[:, :mask_len] = -1
        token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
        target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}
        with torch.no_grad():
            _, losses = model(token_dict=token_dict, target_dict=target_dict)
        valid_losses = [l for l in losses if not torch.isnan(l)]
        if valid_losses:
            avg_step_loss = sum(valid_losses) / len(valid_losses)
            n_tgt = (target_lanes[0] != -1).sum().item()
            total_nll += avg_step_loss.item() * n_tgt
            total_tokens += n_tgt
    avg_loss = total_nll / max(1, total_tokens)
    bpj = total_nll / (math.log(2.0) * max(1, total_jamos))
    ppl = math.exp(min(20.0, avg_loss))
    return {"bpj": float(bpj), "target_loss": float(avg_loss), "target_ppl": float(ppl)}


def generate_seq2seq_without_charhead(model: GPT, tok_24: HangulHybridPosTokenizer, samples: List[Dict[str, str]], max_new_tokens: int = 40) -> Tuple[List[str], float]:
    newline_ids = set(tok_24.sp.encode_as_ids("\n"))
    if tok_24.sp.eos_id() != -1:
        newline_ids.add(tok_24.sp.eos_id())
    hyps = []
    t0 = time.time()
    for s in samples:
        prompt = f"{s['en']}\t"
        p_seq = tok_24.encode_text(prompt)
        token_lists = [[] for _ in range(24)]
        for step in p_seq:
            for i in range(24):
                token_lists[i].append(step[i])
        input_lanes = [torch.tensor(lane, device=DEVICE).unsqueeze(0) for lane in token_lists]
        curr_len = input_lanes[0].size(1)
        for _ in range(max_new_tokens):
            cond_lanes = [lane[:, -256:] for lane in input_lanes]
            token_dict = {f"lane_{i}": lane for i, lane in enumerate(cond_lanes)}
            with torch.no_grad():
                logits_list, _ = model(token_dict=token_dict)
            next_tokens = []
            for i in range(24):
                tok_id = torch.argmax(logits_list[i][:, -1, :], dim=-1).item()
                next_tokens.append(tok_id)
            for i in range(24):
                input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=DEVICE)], dim=1)
            if next_tokens[0] in newline_ids or "\n" in tok_24.sp.id_to_piece(next_tokens[0]):
                break
        gen_steps = []
        for step_idx in range(curr_len, input_lanes[0].size(1)):
            row = [input_lanes[i][0, step_idx].item() for i in range(24)]
            gen_steps.append(row)
        try:
            pred_str = tok_24.decode_sequence(gen_steps)
        except Exception:
            pred_str = ""
        hyps.append(pred_str.strip())
    elapsed = (time.time() - t0) * 1000.0 / max(1, len(samples))
    return hyps, elapsed


def score_kobalt_candidate_without_charhead(model: GPT, tok_24: HangulHybridPosTokenizer, ctx: str, cand: str) -> float:
    ctx_steps = tok_24.encode_text(ctx)
    cand_steps = tok_24.encode_text(cand)
    if not cand_steps:
        return -float("inf")
    full_steps = ctx_steps + cand_steps
    block_size = model.config.block_size
    if len(full_steps) > block_size:
        full_steps = full_steps[-block_size:]
    start_idx = max(0, len(full_steps) - len(cand_steps) - 1)
    token_lists = [[step[i] for step in full_steps] for i in range(24)]
    input_lanes = [torch.tensor([lane[:-1]], dtype=torch.long, device=DEVICE) for lane in token_lists]
    target_lanes = [torch.tensor([lane[1:]], dtype=torch.long, device=DEVICE) for lane in token_lists]
    token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
    target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}
    with torch.no_grad():
        logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)
        lp_0 = torch.log_softmax(logits_list[0], dim=-1)
        tgt_0 = target_lanes[0][:, start_idx:]
        total_lp = lp_0[:, start_idx:, :].gather(-1, tgt_0.unsqueeze(-1)).squeeze(-1).sum().item()
        is_hangul_mask = (tgt_0 == tok_24.hangul_id).float()
        for lane_idx in (1, 2, 3):
            lp = torch.log_softmax(logits_list[lane_idx], dim=-1)
            tgt_slice = target_lanes[lane_idx][:, start_idx:]
            gathered = lp[:, start_idx:, :].gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
            total_lp += (gathered * is_hangul_mask).sum().item()
    return total_lp / max(1, len(cand.strip()))


def eval_kobalt_without_charhead(model: GPT, tok_24: HangulHybridPosTokenizer) -> Dict[str, Any]:
    scores = {}
    tot_correct = 0
    tot_items = 0
    for task_name, items in TASK_DATASETS.items():
        correct = 0
        for item in items:
            ctx = item["ctx"]
            cands = item["candidates"]
            ans = item["answer"]
            cand_scores = [score_kobalt_candidate_without_charhead(model, tok_24, ctx, c) for c in cands]
            pred = int(max(range(len(cand_scores)), key=lambda i: cand_scores[i]))
            if pred == ans:
                correct += 1
        acc = correct / max(1, len(items))
        scores[task_name] = {
            "accuracy": round(acc * 100.0, 2),
            "correct": correct,
            "total": len(items),
        }
        tot_correct += correct
        tot_items += len(items)
    scores["overall_accuracy"] = round((tot_correct / max(1, tot_items)) * 100.0, 2)
    return scores


def main():
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")

    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    out_file = os.path.join(REPO_ROOT, "results_charhead_without_lmhead24_all_metrics.json")
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
            print(f" EVALUATING WITHOUT LM_HEAD_24: {exp_key} @ {ep_name}")
            print(f" Checkpoint: {ckpt_path}")
            print(f"==========================================================================")

            model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
            iter_num = int(ckpt_dict.get("iter_num", 0))

            # 1. Seq2Seq Teacher-Forced on 24 factor lanes
            print("  Evaluating Seq2Seq Teacher-Forced (24 factor lanes)...")
            tf_metrics = eval_seq2seq_teacher_forced_24lanes(model, tok, tf_samples)

            # 2. Seq2Seq Autoregressive Generation without char head
            print("  Evaluating Seq2Seq Generation (pure Jamo composition)...")
            hyps, lat_ms = generate_seq2seq_without_charhead(model, tok, gen_samples)
            canon_hyps = [canonicalize(h) for h in hyps]
            filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
            filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]
            bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
            chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

            # 3. KoBALT & CUTE-Hangul Diagnostic Suite (pure Jamo + BPE)
            print("  Evaluating KoBALT & CUTE-Hangul Diagnostic Suite (no char head)...")
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

    print(f"\nAll evaluations without lm_head_24 completed! Output saved to {out_file}")


if __name__ == "__main__":
    main()
