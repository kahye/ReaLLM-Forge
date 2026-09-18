#!/usr/bin/env python3
"""Automated Seq2Seq Translation Benchmark across All 11 Models and Checkpoints.

Evaluates 4 checkpoints per model:
- 3 epochs
- 5 epochs
- 10 epochs
- Best validation checkpoint

Metrics:
1. BPJ (Bits-Per-Jamo): Total Target NLL / (ln 2 * Total Target Jamos) [EACL 2023 standard]
2. Target Loss: Average Cross-Entropy loss per target step
3. Target Perplexity (PPL): exp(Target NLL / Target Tokens)
4. BLEU-4: Word 4-gram BLEU on canonicalized jamo [EACL 2023 standard]
5. chrF: Character n-gram F-score (char_order=18) on canonicalized jamo [EACL 2023 standard]
6. Generation Latency: Average inference time per sentence (ms)
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
from dataclasses import fields
import json
import math
import os
import pickle
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from jamo import h2j, j2hcj

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from hangul_factorizer import (
    HangulPosFactorizedTokenizer,
    S_BASE,
    V_COUNT,
    T_COUNT,
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
)
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_JSONL = os.path.join(REPO_ROOT, "data", "korean_seq2seq_bench", "test.jsonl")


def canonicalize(text: str) -> str:
    """Canonicalize text according to EACL 2023 Appendix A & Section 4."""
    if not isinstance(text, str):
        return ""
    decomposed = j2hcj(h2j(text))
    stripped = re.sub(r"[^\w\s]", "", decomposed)
    return re.sub(r"\s+", " ", stripped).strip()


def count_jamos(text: str) -> int:
    canon = canonicalize(text)
    return len(canon.replace(" ", ""))


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[GPT, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    filtered_args = {k: v for k, v in model_args.items() if k in allowed}
    filtered_args["dropout"] = 0.0
    conf = GPTConfig(**filtered_args)
    model = GPT(conf)
    state_dict = checkpoint["model"]
    prefix = "_orig_mod."
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            clean_sd[k[len(prefix):]] = v
        else:
            clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    model.eval().to(device)
    return model, checkpoint


class TranslationEvaluator:
    def __init__(self, model_info: Dict[str, Any], ckpt_dict: Dict[str, Any], device: str = DEVICE):
        self.info = model_info
        self.device = device
        self.model_type = model_info["type"]
        self.ckpt_dict = ckpt_dict
        self._init_tokenizer()

    def _init_tokenizer(self):
        m_type = self.model_type
        if m_type == "baseline_char":
            with open(os.path.join(REPO_ROOT, "out_baseline_korean_pos", "meta.pkl"), "rb") as f:
                self.meta = pickle.load(f)
            self.encode_fn, self.decode_fn = get_tokenizer_functions(self.meta)
        elif m_type == "baseline_bpe":
            with open(os.path.join(REPO_ROOT, "out_baseline_char_bpe", "meta.pkl"), "rb") as f:
                self.meta = pickle.load(f)
            self.encode_fn, self.decode_fn = get_tokenizer_functions(self.meta)
        elif m_type == "baseline_byte":
            with open(os.path.join(REPO_ROOT, "out_baseline_pure_byte", "meta.pkl"), "rb") as f:
                self.meta = pickle.load(f)
            self.encode_fn, self.decode_fn = get_tokenizer_functions(self.meta)
        elif m_type.startswith("mc_25"):
            self.lane_names_25 = self.ckpt_dict.get("config", {}).get("multicontext_datasets", [])
            pos_mode = "full" if any("pos_full" in lp for lp in self.lane_names_25) else "coarse"
            self.tok_25 = HangulPosFactorizedTokenizer(use_pos=True, pos_mode=pos_mode)
            self.stois_25 = []
            self.itos_25 = []
            for lp in self.lane_names_25:
                mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                    self.stois_25.append(m["stoi"])
                    self.itos_25.append(m["itos"])
        elif m_type.startswith("mc_24"):
            self.lane_names_24 = self.ckpt_dict.get("config", {}).get("multicontext_datasets", [])
            pos_mode = "full" if any("pos_full" in lp for lp in self.lane_names_24) else "coarse"
            spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
            self.tok_24 = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode=pos_mode)
            self.newline_ids_24 = set(self.tok_24.sp.encode_as_ids("\n"))
            if self.tok_24.sp.eos_id() != -1:
                self.newline_ids_24.add(self.tok_24.sp.eos_id())
            self.stois_24 = []
            self.itos_24 = []
            for lp in self.lane_names_24:
                mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                    self.stois_24.append(m["stoi"])
                    self.itos_24.append(m["itos"])

    def evaluate_teacher_forced(self, model: GPT, samples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluates BPJ, target loss, and target perplexity on test samples."""
        total_nll = 0.0
        total_tokens = 0
        total_jamos = 0

        m_type = self.model_type

        for s in samples:
            en, ko = s["en"], s["ko"]
            prompt = f"{en}\t"
            full = f"{en}\t{ko}\n"
            n_jamos = count_jamos(ko)
            total_jamos += max(1, n_jamos)

            if m_type in ("baseline_char", "baseline_bpe", "baseline_byte"):
                p_ids = self.encode_fn(prompt)
                f_ids = self.encode_fn(full)
                if len(f_ids) > 256:
                    f_ids = f_ids[:256]
                if len(f_ids) <= len(p_ids):
                    continue
                inp = torch.tensor([f_ids[:-1]], device=self.device, dtype=torch.long)
                tgt = torch.tensor([f_ids[1:]], device=self.device, dtype=torch.long)
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
                    total_nll += loss.item() * n_tgt
                    total_tokens += n_tgt

            elif m_type.startswith("mc_25"):
                p_seq = self.tok_25.encode_text(prompt)
                f_seq = self.tok_25.encode_text(full)
                if len(f_seq) > 256:
                    f_seq = f_seq[:256]
                if len(f_seq) <= len(p_seq):
                    continue
                n_lanes = len(self.stois_25)
                token_lists = [[] for _ in range(n_lanes)]
                for item in f_seq:
                    ch = item["char"]
                    idx = item["indices"]
                    if n_lanes == 4:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[0].append(byte_id)
                        for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                            t_char = self.tok_25.token_for(factor_idx, idx[factor_idx])
                            token_lists[lane_i].append(self.stois_25[lane_i].get(t_char, 0))
                    else:
                        for i in range(min(24, n_lanes - 1)):
                            t_char = self.tok_25.token_for(i, idx[i])
                            token_lists[i].append(self.stois_25[i].get(t_char, 0))
                        if n_lanes > 24:
                            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                            token_lists[24].append(byte_id)

                input_lanes = [torch.tensor(lane[:-1], device=self.device).unsqueeze(0) for lane in token_lists]
                target_lanes = [torch.tensor(lane[1:], device=self.device).unsqueeze(0) for lane in token_lists]
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

            elif m_type.startswith("mc_24"):
                p_seq = self.tok_24.encode_text(prompt)
                f_seq = self.tok_24.encode_text(full)
                if len(f_seq) > 256:
                    f_seq = f_seq[:256]
                if len(f_seq) <= len(p_seq):
                    continue
                n_lanes = len(self.stois_24)
                token_lists = [[] for _ in range(n_lanes)]
                for step in f_seq:
                    for i in range(min(24, len(step))):
                        token_lists[i].append(step[i])
                    if n_lanes >= 25:
                        if step[0] == self.tok_24.hangul_id:
                            char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                        else:
                            char_id = 0
                        token_lists[24].append(char_id)

                input_lanes = [torch.tensor(lane[:-1], device=self.device).unsqueeze(0) for lane in token_lists]
                target_lanes = [torch.tensor(lane[1:], device=self.device).unsqueeze(0) for lane in token_lists]
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

        avg_loss = total_nll / max(1, total_tokens)
        bpj = total_nll / (math.log(2.0) * max(1, total_jamos))
        ppl = math.exp(min(20.0, avg_loss))

        return {
            "target_loss": float(avg_loss),
            "target_ppl": float(ppl),
            "bpj": float(bpj),
            "total_nll": float(total_nll),
            "total_jamos": int(total_jamos),
        }

    def generate_completions(self, model: GPT, samples: List[Dict[str, str]], max_new_tokens: int = 40) -> Tuple[List[str], float]:
        """Generates greedy completions for a list of samples."""
        hypotheses = []
        t0 = time.time()
        m_type = self.model_type

        for s in samples:
            en = s["en"]
            prompt = f"{en}\t"

            if m_type in ("baseline_char", "baseline_bpe", "baseline_byte"):
                p_ids = self.encode_fn(prompt)
                inp = torch.tensor([p_ids], device=self.device, dtype=torch.long)
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
                        dec_c = self.decode_fn([next_tok.item()])
                        if "\n" in dec_c:
                            break
                gen_ids = inp[0, curr_len:].tolist()
                if m_type == "baseline_byte":
                    pred_str = bytes([b for b in gen_ids if 0 <= b < 256]).decode("utf-8", errors="ignore").strip()
                else:
                    pred_str = self.decode_fn(gen_ids).strip()
                hypotheses.append(pred_str)

            elif m_type.startswith("mc_25"):
                p_seq = self.tok_25.encode_text(prompt)
                n_lanes = len(self.stois_25)
                token_lists = [[] for _ in range(n_lanes)]
                for item in p_seq:
                    ch = item["char"]
                    idx = item["indices"]
                    if n_lanes == 4:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        token_lists[0].append(byte_id)
                        for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                            t_char = self.tok_25.token_for(factor_idx, idx[factor_idx])
                            token_lists[lane_i].append(self.stois_25[lane_i].get(t_char, 0))
                    else:
                        for i in range(min(24, n_lanes - 1)):
                            t_char = self.tok_25.token_for(i, idx[i])
                            token_lists[i].append(self.stois_25[i].get(t_char, 0))
                        if n_lanes > 24:
                            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                            token_lists[24].append(byte_id)

                input_lanes = [torch.tensor(lane, device=self.device).unsqueeze(0) for lane in token_lists]
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
                    for i in range(n_lanes):
                        input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=self.device)], dim=1)
                    char_lane_idx = 0 if n_lanes == 4 else 24
                    if next_tokens[char_lane_idx] == ord("\n"):
                        break

                gen_steps = []
                for step_idx in range(curr_len, input_lanes[0].size(1)):
                    row = [input_lanes[i][0, step_idx].item() for i in range(n_lanes)]
                    gen_steps.append(row)

                decoded_chars = []
                char_lane_idx = 0 if n_lanes == 4 else 24
                for row in gen_steps:
                    t1 = self.itos_25[1].get(row[1], "")
                    t2 = self.itos_25[2].get(row[2], "")
                    t3 = self.itos_25[3].get(row[3], "")
                    if len(t1) > 0 and len(t2) > 0 and len(t3) > 0:
                        l = (ord(t1) - 0xE100) - 1
                        v = (ord(t2) - 0xE200) - 1
                        t = ord(t3) - 0xE300
                        if 0 <= l < len(CHOSEONG) and 0 <= v < len(JUNGSEONG) and 0 <= t < len(JONGSEONG):
                            decoded_chars.append(chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t))
                        else:
                            decoded_chars.append(chr(row[char_lane_idx]))
                    else:
                        decoded_chars.append(chr(row[char_lane_idx]))
                hypotheses.append("".join(decoded_chars).strip())

            elif m_type.startswith("mc_24"):
                p_seq = self.tok_24.encode_text(prompt)
                n_lanes = len(self.stois_24)
                token_lists = [[] for _ in range(n_lanes)]
                for step in p_seq:
                    for i in range(min(24, len(step))):
                        token_lists[i].append(step[i])
                    if n_lanes >= 25:
                        if step[0] == self.tok_24.hangul_id:
                            char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                        else:
                            char_id = 0
                        token_lists[24].append(char_id)

                input_lanes = [torch.tensor(lane, device=self.device).unsqueeze(0) for lane in token_lists]
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
                    if n_lanes >= 25 and next_tokens[0] == self.tok_24.hangul_id:
                        # Use output-only char head (logits_list[24]) over valid Hangul syllables (1..11172)
                        char_logits = logits_list[24][0, -1, 1:11173]
                        best_syll_id = torch.argmax(char_logits).item() + 1
                        s_idx = best_syll_id - 1
                        syl_char = chr(0xAC00 + s_idx)
                        factored_step = self.tok_24.encode_text(syl_char)[0]
                        for lane_idx in range(min(23, len(factored_step))):
                            next_tokens[lane_idx] = factored_step[lane_idx]
                        next_tokens[24] = best_syll_id
                    for i in range(n_lanes):
                        input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=self.device)], dim=1)
                    if next_tokens[0] in self.newline_ids_24 or "\n" in self.tok_24.sp.id_to_piece(next_tokens[0]):
                        break

                gen_steps = []
                for step_idx in range(curr_len, input_lanes[0].size(1)):
                    row = [input_lanes[i][0, step_idx].item() for i in range(min(24, n_lanes))]
                    gen_steps.append(row)
                try:
                    pred_str = self.tok_24.decode_sequence(gen_steps)
                except Exception:
                    pred_str = ""
                hypotheses.append(pred_str.strip())

        elapsed_ms = (time.time() - t0) * 1000.0 / max(1, len(samples))
        return hypotheses, elapsed_ms


MODELS_CONFIG = [
    {
        "id": "baseline_character",
        "name": "Baseline Character Tokenizer (V=4,924)",
        "type": "baseline_char",
        "dir": "out_baseline_korean_pos",
        "checkpoints": {
            "3_epoch": "out_baseline_korean_pos/11572.pt",
            "5_epoch": "out_baseline_korean_pos/18165.pt",
            "10_epoch": "out_baseline_korean_pos/36330.pt",
            "best": "out_baseline_korean_pos/best_ckpt.pt",
        },
    },
    {
        "id": "baseline_char_bpe",
        "name": "Baseline Byte-Fallback BPE (V=4,924)",
        "type": "baseline_bpe",
        "dir": "out_baseline_char_bpe",
        "checkpoints": {
            "3_epoch": "out_baseline_char_bpe/10899.pt",
            "5_epoch": "out_baseline_char_bpe/18165.pt",
            "10_epoch": "out_baseline_char_bpe/36330.pt",
            "best": "out_baseline_char_bpe/best_ckpt.pt",
        },
    },
    {
        "id": "baseline_pure_byte",
        "name": "Baseline Pure Byte Tokenizer (V=256)",
        "type": "baseline_byte",
        "dir": "out_baseline_pure_byte",
        "checkpoints": {
            "3_epoch": "out_baseline_pure_byte/10899.pt",
            "5_epoch": "out_baseline_pure_byte/18165.pt",
            "10_epoch": "out_baseline_pure_byte/36330.pt",
            "best": "out_baseline_pure_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_full_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Weighted)",
        "type": "mc_25_full",
        "dir": "out_mc_full_pos_weighted_byte",
        "checkpoints": {
            "3_epoch": "out_mc_full_pos_weighted_byte/10899.pt",
            "5_epoch": "out_mc_full_pos_weighted_byte/18165.pt",
            "10_epoch": "out_mc_full_pos_weighted_byte/36330.pt",
            "best": "out_mc_full_pos_weighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_full_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Full POS, Unweighted)",
        "type": "mc_25_full",
        "dir": "out_mc_full_pos_unweighted_byte",
        "checkpoints": {
            "3_epoch": "out_mc_full_pos_unweighted_byte/10899.pt",
            "5_epoch": "out_mc_full_pos_unweighted_byte/18165.pt",
            "10_epoch": "out_mc_full_pos_unweighted_byte/36330.pt",
            "best": "out_mc_full_pos_unweighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_coarse_pos_weighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Weighted)",
        "type": "mc_25_coarse",
        "dir": "out_mc_coarse_pos_weighted_byte",
        "checkpoints": {
            "3_epoch": "out_mc_coarse_pos_weighted_byte/10899.pt",
            "5_epoch": "out_mc_coarse_pos_weighted_byte/18165.pt",
            "10_epoch": "out_mc_coarse_pos_weighted_byte/36330.pt",
            "best": "out_mc_coarse_pos_weighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_coarse_pos_unweighted_byte",
        "name": "Hangul Factorizer + Char Stream (Coarse POS, Unweighted)",
        "type": "mc_25_coarse",
        "dir": "out_mc_coarse_pos_unweighted_byte",
        "checkpoints": {
            "3_epoch": "out_mc_coarse_pos_unweighted_byte/10899.pt",
            "5_epoch": "out_mc_coarse_pos_unweighted_byte/18165.pt",
            "10_epoch": "out_mc_coarse_pos_unweighted_byte/36330.pt",
            "best": "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt",
        },
    },
    {
        "id": "mc_nochar_full_pos_weighted",
        "name": "Hangul Factorizer No-Char (Full POS, Weighted)",
        "type": "mc_24_full",
        "dir": "out_mc_nochar_full_pos_weighted",
        "checkpoints": {
            "3_epoch": "out_mc_nochar_full_pos_weighted/3_epochs.pt",
            "5_epoch": "out_mc_nochar_full_pos_weighted/5_epochs.pt",
            "10_epoch": "out_mc_nochar_full_pos_weighted/10_epochs.pt",
            "15_epoch": "out_mc_nochar_full_pos_weighted/15_epochs.pt",
            "20_epoch": "out_mc_nochar_full_pos_weighted/20_epochs.pt",
            "best": "out_mc_nochar_full_pos_weighted/best_ckpt.pt",
        },
    },
    {
        "id": "mc_nochar_full_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Full POS, Unweighted)",
        "type": "mc_24_full",
        "dir": "out_mc_nochar_full_pos_unweighted",
        "checkpoints": {
            "3_epoch": "out_mc_nochar_full_pos_unweighted/3_epochs.pt",
            "5_epoch": "out_mc_nochar_full_pos_unweighted/5_epochs.pt",
            "10_epoch": "out_mc_nochar_full_pos_unweighted/10_epochs.pt",
            "15_epoch": "out_mc_nochar_full_pos_unweighted/15_epochs.pt",
            "20_epoch": "out_mc_nochar_full_pos_unweighted/20_epochs.pt",
            "best": "out_mc_nochar_full_pos_unweighted/best_ckpt.pt",
        },
    },
    {
        "id": "mc_nochar_coarse_pos_weighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Weighted)",
        "type": "mc_24_coarse",
        "dir": "out_mc_nochar_coarse_pos_weighted",
        "checkpoints": {
            "3_epoch": "out_mc_nochar_coarse_pos_weighted/3_epochs.pt",
            "5_epoch": "out_mc_nochar_coarse_pos_weighted/5_epochs.pt",
            "10_epoch": "out_mc_nochar_coarse_pos_weighted/10_epochs.pt",
            "15_epoch": "out_mc_nochar_coarse_pos_weighted/15_epochs.pt",
            "20_epoch": "out_mc_nochar_coarse_pos_weighted/20_epochs.pt",
            "best": "out_mc_nochar_coarse_pos_weighted/best_ckpt.pt",
        },
    },
    {
        "id": "mc_nochar_coarse_pos_unweighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, Unweighted)",
        "type": "mc_24_coarse",
        "dir": "out_mc_nochar_coarse_pos_unweighted",
        "checkpoints": {
            "3_epoch": "out_mc_nochar_coarse_pos_unweighted/3_epochs.pt",
            "5_epoch": "out_mc_nochar_coarse_pos_unweighted/5_epochs.pt",
            "10_epoch": "out_mc_nochar_coarse_pos_unweighted/10_epochs.pt",
            "15_epoch": "out_mc_nochar_coarse_pos_unweighted/15_epochs.pt",
            "20_epoch": "out_mc_nochar_coarse_pos_unweighted/20_epochs.pt",
            "best": "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt",
        },
    },
    {
        "id": "three_hot_ind_char_byte",
        "name": "Three-Hot Ind Char-Byte (Song 2018)",
        "type": "mc_25_three_hot",
        "dir": "out_three_hot_ind_char_byte",
        "checkpoints": {
            "best": "out_three_hot_ind_char_byte/best_ckpt.pt",
        },
    },
    {
        "id": "three_hot_cond_char_byte",
        "name": "Three-Hot Cond Char-Byte (Cognetta 2023)",
        "type": "mc_25_three_hot",
        "dir": "out_three_hot_cond_char_byte",
        "checkpoints": {
            "best": "out_three_hot_cond_char_byte/best_ckpt.pt",
        },
    },
    {
        "id": "three_hot_ind_nochar",
        "name": "Three-Hot Ind No-Char BPE (Song 2018)",
        "type": "mc_24_three_hot",
        "dir": "out_three_hot_ind_nochar",
        "checkpoints": {
            "best": "out_three_hot_ind_nochar/best_ckpt.pt",
        },
    },
    {
        "id": "three_hot_cond_nochar",
        "name": "Three-Hot Cond No-Char BPE (Cognetta 2023)",
        "type": "mc_24_three_hot",
        "dir": "out_three_hot_cond_nochar",
        "checkpoints": {
            "best": "out_three_hot_cond_nochar/best_ckpt.pt",
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run Seq2Seq Translation Benchmark")
    parser.add_argument("--teacher_forced_samples", type=int, default=500, help="Number of samples for teacher-forced BPJ/PPL")
    parser.add_argument("--gen_samples", type=int, default=100, help="Number of samples for BLEU/chrF generation")
    parser.add_argument("--output_json", type=str, default="results_seq2seq_translation_benchmark.json", help="Output JSON path")
    parser.add_argument("--models", nargs="+", default=None, help="Filter specific model IDs to evaluate")
    args = parser.parse_args()

    print(f"Loading test samples from {TEST_JSONL}...")
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    tf_samples = all_samples[:args.teacher_forced_samples]
    gen_samples = all_samples[:args.gen_samples]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    print(f"Loaded {len(all_samples)} total test pairs.")
    print(f"Using {len(tf_samples)} samples for teacher-forced evaluation, {len(gen_samples)} samples for generation.")

    models_to_run = [m for m in MODELS_CONFIG if not args.models or m["id"] in args.models]
    results: Dict[str, Any] = {}
    total_ckpts = sum(len(m["checkpoints"]) for m in models_to_run)
    done_ckpts = 0

    print(f"Beginning evaluation across {len(models_to_run)} models ({total_ckpts} checkpoints)...")

    for model_info in models_to_run:
        m_id = model_info["id"]
        m_name = model_info["name"]
        print(f"\n==================================================================")
        print(f"Evaluating Model: {m_name} ({m_id})")
        print(f"==================================================================")

        results[m_id] = {
            "name": m_name,
            "type": model_info["type"],
            "dir": model_info["dir"],
            "checkpoints": {},
        }

        for ckpt_key, ckpt_rel_path in model_info["checkpoints"].items():
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel_path)
            done_ckpts += 1
            print(f"\n[{done_ckpts}/{total_ckpts}] Checkpoint: {ckpt_key} -> {ckpt_rel_path}")

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint {ckpt_path} not found! Skipping.")
                continue

            try:
                model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
                val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
                iter_num = int(ckpt_dict.get("iter_num", 0))

                evaluator = TranslationEvaluator(model_info, ckpt_dict=ckpt_dict, device=DEVICE)

                # 1. Teacher-Forced Evaluation
                t_eval = evaluator.evaluate_teacher_forced(model, tf_samples)
                print(f"  [Teacher-Forced] BPJ: {t_eval['bpj']:.4f} | Target Loss: {t_eval['target_loss']:.4f} | Target PPL: {t_eval['target_ppl']:.2f}")

                # 2. Autoregressive Generation
                hyps, gen_lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)
                canon_hyps = [canonicalize(h) for h in hyps]

                # Compute BLEU and chrF
                filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
                filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]

                bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
                chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

                print(f"  [Generation] BLEU: {bleu:.2f} | chrF: {chrf:.2f} | Latency: {gen_lat_ms:.1f} ms/sample")

                # Sample demonstration
                demo_en = gen_samples[0]["en"]
                demo_ref = gen_samples[0]["ko"]
                demo_pred = hyps[0]
                print(f"  Sample EN: {demo_en[:50]}...")
                print(f"  Sample KO: {demo_ref[:50]}...")
                print(f"  Sample Pred: {demo_pred[:50]}...")

                results[m_id]["checkpoints"][ckpt_key] = {
                    "path": ckpt_rel_path,
                    "iter_num": iter_num,
                    "train_val_loss": val_loss,
                    "bpj": t_eval["bpj"],
                    "target_loss": t_eval["target_loss"],
                    "target_ppl": t_eval["target_ppl"],
                    "bleu": float(bleu),
                    "chrf": float(chrf),
                    "latency_ms": float(gen_lat_ms),
                    "demo_pred": demo_pred,
                }

                # Free GPU memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error evaluating checkpoint {ckpt_path}: {e}")
                import traceback
                traceback.print_exc()

        # Save checkpoint results incrementally
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete! Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
