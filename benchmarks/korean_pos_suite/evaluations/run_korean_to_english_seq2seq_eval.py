#!/usr/bin/env python3
"""Comprehensive Korean-to-English Seq2Seq Translation Benchmark Evaluator for All 15 Models (Best Checkpoints).

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

Task: Korean -> English Translation
Input Prompt: "{ko}\t"
Target Reference: "{en}\n"

Computes:
1. Teacher-Forced Target Metrics (100 test samples):
   - BPJ / BPC: Bits Per Target Character / Jamo (Total NLL / (ln 2 * Total Characters))
   - Target Loss (average across active lanes)
   - Target Perplexity (PPL)
   - Text Lane Target Loss (Lane 0 for BPE / Lane 24 for byte stream)
   - Text Lane Perplexity
2. Autoregressive Completion Metrics (20 test samples, max_new_tokens=40):
   - English Word BLEU-4 (SacreBLEU standard)
   - English chrF (SacreBLEU standard char_order=6)
   - Generation Latency (ms/sentence)
   - Sentence-level BLEU-4 and chrF per sample
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
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import sacrebleu
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)

from run_seq2seq_translation_benchmark import (
    TEST_JSONL,
    TranslationEvaluator,
    count_jamos,
    load_model_from_ckpt,
)
from hangul_factorizer import (
    S_BASE,
    V_COUNT,
    T_COUNT,
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
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


def evaluate_ko_to_en_teacher_forced(evaluator: TranslationEvaluator, model, samples: List[Dict[str, str]]) -> Dict[str, float]:
    """Evaluates teacher-forced Target Loss, Target PPL, BPC, and Lane 0 Text Loss on Korean-to-English."""
    total_nll = 0.0
    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    
    lane0_nll = 0.0
    lane0_tokens = 0

    m_type = evaluator.model_type

    for s in samples:
        ko, en = s["ko"], s["en"]
        prompt = f"{ko}\t"
        full = f"{ko}\t{en}\n"
        n_chars = max(1, len(en.strip()))
        n_bytes = max(1, len(en.strip().encode("utf-8")))
        total_chars += n_chars
        total_bytes += n_bytes

        if m_type in ("baseline_char", "baseline_bpe", "baseline_byte"):
            p_ids = evaluator.encode_fn(prompt)
            f_ids = evaluator.encode_fn(full)
            if len(f_ids) > 256:
                f_ids = f_ids[:256]
            if len(f_ids) <= len(p_ids):
                continue
            inp = torch.tensor([f_ids[:-1]], device=evaluator.device, dtype=torch.long)
            tgt = torch.tensor([f_ids[1:]], device=evaluator.device, dtype=torch.long)
            mask_len = min(len(p_ids) - 1, tgt.size(1))
            tgt[:, :mask_len] = -1
            vocab_size = model.config.vocab_size
            inp = torch.clamp(inp, 0, vocab_size - 1)
            valid_mask = tgt != -1
            tgt[valid_mask] = torch.clamp(tgt[valid_mask], 0, vocab_size - 1)

            with torch.no_grad():
                _, loss = model(inp, targets=tgt)
            n_tgt = (tgt != -1).sum().item()
            if n_tgt > 0 and not torch.isnan(loss):
                step_nll = loss.item() * n_tgt
                total_nll += step_nll
                total_tokens += n_tgt
                lane0_nll += step_nll
                lane0_tokens += n_tgt

        elif m_type.startswith("mc_25"):
            p_seq = evaluator.tok_25.encode_text(prompt)
            f_seq = evaluator.tok_25.encode_text(full)
            if len(f_seq) > 256:
                f_seq = f_seq[:256]
            if len(f_seq) <= len(p_seq):
                continue
            n_lanes = len(evaluator.stois_25)
            token_lists = [[] for _ in range(n_lanes)]
            for item in f_seq:
                ch = item["char"]
                idx = item["indices"]
                if n_lanes == 4:
                    byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                    token_lists[0].append(byte_id)
                    for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                        t_char = evaluator.tok_25.token_for(factor_idx, idx[factor_idx])
                        token_lists[lane_i].append(evaluator.stois_25[lane_i].get(t_char, 0))
                else:
                    for i in range(min(24, n_lanes - 1)):
                        t_char = evaluator.tok_25.token_for(i, idx[i])
                        token_lists[i].append(evaluator.stois_25[i].get(t_char, 0))
                    if n_lanes > 24:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[24].append(byte_id)

            input_lanes = [torch.tensor(lane[:-1], device=evaluator.device).unsqueeze(0) for lane in token_lists]
            target_lanes = [torch.tensor(lane[1:], device=evaluator.device).unsqueeze(0) for lane in token_lists]
            mask_len = min(len(p_seq) - 1, target_lanes[0].size(1))
            for lane in target_lanes:
                lane[:, :mask_len] = -1

            token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
            target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

            with torch.no_grad():
                logits_list, losses = model(token_dict=token_dict, target_dict=target_dict)
            valid_losses = [l for l in losses if not torch.isnan(l)]
            if valid_losses:
                avg_step_loss = sum(valid_losses) / len(valid_losses)
                n_tgt = (target_lanes[0] != -1).sum().item()
                total_nll += avg_step_loss.item() * n_tgt
                total_tokens += n_tgt
                
                # Text lane for mc_25 is byte lane (lane 24 or lane 0)
                byte_lane_idx = 0 if n_lanes == 4 else 24
                if byte_lane_idx < len(losses) and not torch.isnan(losses[byte_lane_idx]):
                    lane0_nll += losses[byte_lane_idx].item() * n_tgt
                    lane0_tokens += n_tgt

        elif m_type.startswith("mc_24"):
            p_seq = evaluator.tok_24.encode_text(prompt)
            f_seq = evaluator.tok_24.encode_text(full)
            if len(f_seq) > 256:
                f_seq = f_seq[:256]
            if len(f_seq) <= len(p_seq):
                continue
            n_lanes = len(evaluator.stois_24)
            token_lists = [[] for _ in range(n_lanes)]
            for step in f_seq:
                for i in range(min(24, len(step))):
                    token_lists[i].append(step[i])
                if n_lanes >= 25:
                    if step[0] == evaluator.tok_24.hangul_id:
                        char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                    else:
                        char_id = 0
                    token_lists[24].append(char_id)

            input_lanes = [torch.tensor(lane[:-1], device=evaluator.device).unsqueeze(0) for lane in token_lists]
            target_lanes = [torch.tensor(lane[1:], device=evaluator.device).unsqueeze(0) for lane in token_lists]
            mask_len = min(len(p_seq) - 1, target_lanes[0].size(1))
            for lane in target_lanes:
                lane[:, :mask_len] = -1

            token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
            target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

            with torch.no_grad():
                logits_list, losses = model(token_dict=token_dict, target_dict=target_dict)
            valid_losses = [l for l in losses if not torch.isnan(l)]
            if valid_losses:
                avg_step_loss = sum(valid_losses) / len(valid_losses)
                n_tgt = (target_lanes[0] != -1).sum().item()
                total_nll += avg_step_loss.item() * n_tgt
                total_tokens += n_tgt
                
                # Lane 0 is the BPE text prediction lane
                if not torch.isnan(losses[0]):
                    lane0_nll += losses[0].item() * n_tgt
                    lane0_tokens += n_tgt

    avg_loss = total_nll / max(1, total_tokens)
    bpc = total_nll / (math.log(2.0) * max(1, total_chars))
    bpb = total_nll / (math.log(2.0) * max(1, total_bytes))
    ppl = math.exp(min(20.0, avg_loss))

    l0_loss = lane0_nll / max(1, lane0_tokens)
    l0_ppl = math.exp(min(20.0, l0_loss))
    l0_bpc = lane0_nll / (math.log(2.0) * max(1, total_chars))

    return {
        "target_loss": float(avg_loss),
        "target_ppl": float(ppl),
        "bpc": float(bpc),
        "bpb": float(bpb),
        "bpj": float(bpc),  # Aligned key for direct cross-task comparison
        "lane0_target_loss": float(l0_loss),
        "lane0_target_ppl": float(l0_ppl),
        "lane0_bpc": float(l0_bpc),
        "total_nll": float(total_nll),
        "total_chars": int(total_chars),
        "total_bytes": int(total_bytes),
        "total_tokens": int(total_tokens),
    }


def generate_ko_to_en_completions(evaluator: TranslationEvaluator, model, samples: List[Dict[str, str]], max_new_tokens: int = 40) -> Tuple[List[str], float]:
    """Generates autoregressive English completions from Korean prompts with latency measurement."""
    hypotheses = []
    t0 = time.time()
    m_type = evaluator.model_type

    for s in samples:
        ko = s["ko"]
        prompt = f"{ko}\t"

        if m_type in ("baseline_char", "baseline_bpe", "baseline_byte"):
            p_ids = evaluator.encode_fn(prompt)
            inp = torch.tensor([p_ids], device=evaluator.device, dtype=torch.long)
            curr_len = inp.size(1)
            for _ in range(max_new_tokens):
                cond = inp[:, -256:]
                with torch.no_grad():
                    logits, _ = model(cond)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                inp = torch.cat([inp, next_tok], dim=1)
                if m_type == "baseline_byte" and next_tok.item() == ord("\n"):
                    break
                elif m_type != "baseline_byte":
                    dec_c = evaluator.decode_fn([next_tok.item()])
                    if "\n" in dec_c:
                        break
            gen_ids = inp[0, curr_len:].tolist()
            if m_type == "baseline_byte":
                pred_str = bytes([b for b in gen_ids if 0 <= b < 256]).decode("utf-8", errors="ignore").strip()
            else:
                pred_str = evaluator.decode_fn(gen_ids).strip()
            hypotheses.append(pred_str)

        elif m_type.startswith("mc_25"):
            p_seq = evaluator.tok_25.encode_text(prompt)
            n_lanes = len(evaluator.stois_25)
            token_lists = [[] for _ in range(n_lanes)]
            for item in p_seq:
                ch = item["char"]
                idx = item["indices"]
                if n_lanes == 4:
                    byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                    token_lists[0].append(byte_id)
                    for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                        t_char = evaluator.tok_25.token_for(factor_idx, idx[factor_idx])
                        token_lists[lane_i].append(evaluator.stois_25[lane_i].get(t_char, 0))
                else:
                    for i in range(min(24, n_lanes - 1)):
                        t_char = evaluator.tok_25.token_for(i, idx[i])
                        token_lists[i].append(evaluator.stois_25[i].get(t_char, 0))
                    if n_lanes > 24:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[24].append(byte_id)

            input_lanes = [torch.tensor(lane, device=evaluator.device).unsqueeze(0) for lane in token_lists]
            curr_len = input_lanes[0].size(1)
            char_lane_idx = 0 if n_lanes == 4 else 24
            for _ in range(max_new_tokens):
                cond_lanes = [lane[:, -256:] for lane in input_lanes]
                token_dict = {f"lane_{i}": lane for i, lane in enumerate(cond_lanes)}
                with torch.no_grad():
                    logits_list, _ = model(token_dict=token_dict)
                next_tokens = []
                for i, logits in enumerate(logits_list):
                    tok_id = torch.argmax(logits[:, -1, :], dim=-1).item()
                    next_tokens.append(tok_id)
                for i in range(n_lanes):
                    input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=evaluator.device)], dim=1)
                if next_tokens[char_lane_idx] == ord("\n"):
                    break

            gen_steps = []
            for step_idx in range(curr_len, input_lanes[0].size(1)):
                row = [input_lanes[i][0, step_idx].item() for i in range(n_lanes)]
                gen_steps.append(row)

            decoded_chars = []
            for row in gen_steps:
                t1 = evaluator.itos_25[1].get(row[1], "")
                t2 = evaluator.itos_25[2].get(row[2], "")
                t3 = evaluator.itos_25[3].get(row[3], "")
                if len(t1) > 0 and len(t2) > 0 and len(t3) > 0:
                    l = (ord(t1) - 0xE100) - 1
                    v = (ord(t2) - 0xE200) - 1
                    t = ord(t3) - 0xE300
                    if 0 <= l < len(CHOSEONG) and 0 <= v < len(JUNGSEONG) and 0 <= t < len(JONGSEONG):
                        decoded_chars.append(chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t))
                    else:
                        b_val = row[char_lane_idx]
                        decoded_chars.append(chr(b_val) if 0 <= b_val < 256 else "")
                else:
                    b_val = row[char_lane_idx]
                    decoded_chars.append(chr(b_val) if 0 <= b_val < 256 else "")
            hypotheses.append("".join(decoded_chars).strip())

        elif m_type.startswith("mc_24"):
            p_seq = evaluator.tok_24.encode_text(prompt)
            n_lanes = len(evaluator.stois_24)
            token_lists = [[] for _ in range(n_lanes)]
            for step in p_seq:
                for i in range(min(24, len(step))):
                    token_lists[i].append(step[i])
                if n_lanes >= 25:
                    if step[0] == evaluator.tok_24.hangul_id:
                        char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                    else:
                        char_id = 0
                    token_lists[24].append(char_id)

            input_lanes = [torch.tensor(lane, device=evaluator.device).unsqueeze(0) for lane in token_lists]
            curr_len = input_lanes[0].size(1)
            for _ in range(max_new_tokens):
                cond_lanes = [lane[:, -256:] for lane in input_lanes]
                token_dict = {f"lane_{i}": lane for i, lane in enumerate(cond_lanes)}
                with torch.no_grad():
                    logits_list, _ = model(token_dict=token_dict)
                next_tokens = []
                for i, logits in enumerate(logits_list):
                    tok_id = torch.argmax(logits[:, -1, :], dim=-1).item()
                    next_tokens.append(tok_id)
                if n_lanes >= 25 and next_tokens[0] == evaluator.tok_24.hangul_id:
                    char_logits = logits_list[24][0, -1, 1:11173]
                    best_syll_id = torch.argmax(char_logits).item() + 1
                    s_idx = best_syll_id - 1
                    syl_char = chr(0xAC00 + s_idx)
                    factored_step = evaluator.tok_24.encode_text(syl_char)[0]
                    for lane_idx in range(min(23, len(factored_step))):
                        next_tokens[lane_idx] = factored_step[lane_idx]
                    next_tokens[24] = best_syll_id
                for i in range(n_lanes):
                    input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=evaluator.device)], dim=1)
                if next_tokens[0] in evaluator.newline_ids_24 or "\n" in evaluator.tok_24.sp.id_to_piece(next_tokens[0]):
                    break

            gen_steps = []
            for step_idx in range(curr_len, input_lanes[0].size(1)):
                row = [input_lanes[i][0, step_idx].item() for i in range(min(24, n_lanes))]
                gen_steps.append(row)
            try:
                pred_str = evaluator.tok_24.decode_sequence(gen_steps)
            except Exception:
                pred_str = ""
            hypotheses.append(pred_str.strip())

    elapsed_ms = (time.time() - t0) * 1000.0 / max(1, len(samples))
    return hypotheses, elapsed_ms


def main():
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    en_refs = [s["en"].strip() for s in gen_samples]

    out_path = os.path.join(REPO_ROOT, "results_seq2seq_ko_to_en_all_15_best_checkpoints.json")
    results: Dict[str, Any] = {
        "benchmark_task": "korean_to_english_seq2seq_translation",
        "samples_metadata": [
            {"idx": idx, "ko": s["ko"], "en": s["en"]}
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
        print(f" EVALUATING KO -> EN [{mid}]: {spec['name']}")
        print(f" Checkpoint: {spec['ckpt_path']}")
        print("================================================================================")

        model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
        val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
        iter_num = int(ckpt_dict.get("iter_num", 0))

        evaluator = TranslationEvaluator(spec, ckpt_dict=ckpt_dict, device=DEVICE)

        # 1. Corpus-level teacher-forced metrics (100 test samples)
        t_eval = evaluate_ko_to_en_teacher_forced(evaluator, model, tf_samples)

        # 2. Autoregressive completions on 20 samples
        hyps, gen_lat_ms = generate_ko_to_en_completions(evaluator, model, gen_samples, max_new_tokens=40)
        filt_hyps = [h if h.strip() else "<empty>" for h in hyps]
        filt_refs = [r if r.strip() else "<empty>" for r in en_refs]

        corpus_bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
        corpus_chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs]).score

        # 3. Per-sample metrics for each of the 20 samples
        per_sample_records: List[Dict[str, Any]] = []
        for idx, s in enumerate(gen_samples):
            single_tf = evaluate_ko_to_en_teacher_forced(evaluator, model, [s])
            s_bpc = single_tf["bpc"]
            s_loss = single_tf["target_loss"]
            s_ppl = single_tf["target_ppl"]
            s_l0_loss = single_tf["lane0_target_loss"]

            h_str = filt_hyps[idx]
            r_str = filt_refs[idx]
            s_bleu = sacrebleu.sentence_bleu(h_str, [r_str], smooth_method="exp", use_effective_order=True).score
            s_chrf = sacrebleu.sentence_chrf(h_str, [r_str]).score

            per_sample_records.append(
                {
                    "sample_idx": idx,
                    "ko": s["ko"],
                    "en_ref": s["en"],
                    "response": hyps[idx],
                    "bpc": float(s_bpc),
                    "target_loss": float(s_loss),
                    "target_ppl": float(s_ppl),
                    "lane0_target_loss": float(s_l0_loss),
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
                "bpc": float(t_eval["bpc"]),
                "bpb": float(t_eval["bpb"]),
                "target_loss": float(t_eval["target_loss"]),
                "target_ppl": float(t_eval["target_ppl"]),
                "lane0_target_loss": float(t_eval["lane0_target_loss"]),
                "lane0_target_ppl": float(t_eval["lane0_target_ppl"]),
                "lane0_bpc": float(t_eval["lane0_bpc"]),
                "bleu": float(corpus_bleu),
                "chrf": float(corpus_chrf),
                "latency_ms": float(gen_lat_ms),
            },
            "per_sample": per_sample_records,
        }

        print(
            f"  => BPC: {t_eval['bpc']:.4f} | Target Loss: {t_eval['target_loss']:.4f} | Target PPL: {t_eval['target_ppl']:.2f} | "
            f"Lane 0 Loss: {t_eval['lane0_target_loss']:.4f} | BLEU: {corpus_bleu:.2f} | chrF: {corpus_chrf:.2f} | Latency: {gen_lat_ms:.1f}ms"
        )

        # Save incrementally
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved full 15-model detailed Korean-to-English seq2seq benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
