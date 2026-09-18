#!/usr/bin/env python3
"""Run Tail BPC, Held-Out Syllables, and OOV Resilience evaluations across:
1. Muon Best Validation Loss Checkpoints
2. AdamW Best Validation Loss Checkpoints
3. AdamW Best Downstream Checkpoints
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
import argparse
from collections import Counter
from dataclasses import fields
import json
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
if KOREAN_DIR not in sys.path:
    sys.path.insert(0, KOREAN_DIR)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from hangul_factorizer import S_BASE, S_COUNT, N_COUNT, T_COUNT, CHOSEONG, JUNGSEONG, JONGSEONG, HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MUON_VAL_LOSS_MODELS = [
    ("Baseline Character Tokenizer (V=4,924)", "out_baseline_korean_pos_muon/best_ckpt.pt", "best_ckpt (70000)", "single_char"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "out_baseline_char_bpe_muon/best_ckpt.pt", "best_ckpt (63000)", "single_bpe"),
    ("Baseline Pure Byte Tokenizer (V=256)", "out_baseline_pure_byte_muon/best_ckpt.pt", "best_ckpt (111000)", "single_byte"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "out_three_hot_ind_char_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "three_hot_ind"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "out_three_hot_cond_char_byte_muon/best_ckpt.pt", "best_ckpt (65000)", "three_hot_cond"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "out_mc_full_pos_weighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_full_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "out_mc_full_pos_unweighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_full_pos_unweighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "out_mc_coarse_pos_weighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_coarse_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "out_mc_coarse_pos_unweighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_coarse_pos_unweighted"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "out_mc_nochar_full_pos_new_weighted_muon/best_ckpt.pt", "best_ckpt (44000)", "nochar_full_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_unweighted_muon/best_ckpt.pt", "best_ckpt (41000)", "nochar_full_pos_unweighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "out_mc_nochar_coarse_pos_new_weighted_muon/best_ckpt.pt", "best_ckpt (35000)", "nochar_coarse_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_unweighted_muon/best_ckpt.pt", "best_ckpt (35000)", "nochar_coarse_pos_unweighted"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_full_h11_with"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_full_h11_without"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24, Joint Factor-Product)", "out_mc_nochar_full_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_full_h11_joint"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_full_unweighted_with"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_full_unweighted_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_coarse_h11_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_coarse_h11_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_coarse_unweighted_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "charhead_coarse_unweighted_without"),
]

ADAMW_VAL_LOSS_MODELS = [
    ("Baseline Character Tokenizer (V=4,924)", "out_baseline_korean_pos/best_ckpt.pt", "best_ckpt (62000)", "single_char"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "out_baseline_char_bpe/best_ckpt.pt", "best_ckpt (71000)", "single_bpe"),
    ("Baseline Pure Byte Tokenizer (V=256)", "out_baseline_pure_byte/best_ckpt.pt", "best_ckpt (106000)", "single_byte"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "out_three_hot_ind_char_byte/best_ckpt.pt", "best_ckpt (72000)", "three_hot_ind"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "out_three_hot_cond_char_byte/best_ckpt.pt", "best_ckpt (65000)", "three_hot_cond"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "out_mc_full_pos_weighted_byte/best_ckpt.pt", "best_ckpt (72000)", "mc_full_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "out_mc_full_pos_unweighted_byte/best_ckpt.pt", "best_ckpt (65000)", "mc_full_pos_unweighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "out_mc_coarse_pos_weighted_byte/best_ckpt.pt", "best_ckpt (67000)", "mc_coarse_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt", "best_ckpt (57000)", "mc_coarse_pos_unweighted"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt", "best_ckpt (45000)", "nochar_full_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_unweighted/best_ckpt.pt", "best_ckpt (43000)", "nochar_full_pos_unweighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "best_ckpt (45000)", "nochar_coarse_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt", "best_ckpt (45000)", "nochar_coarse_pos_unweighted"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34232)", "charhead_full_h11_with"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34232)", "charhead_full_h11_without"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24, Joint Factor-Product)", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34232)", "charhead_full_h11_joint"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_full_unweighted_with"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_full_unweighted_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34232)", "charhead_coarse_h11_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34232)", "charhead_coarse_h11_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_coarse_unweighted_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_coarse_unweighted_without"),
]

ADAMW_DOWNSTREAM_MODELS = [
    ("Baseline Character Tokenizer (V=4,924)", "out_baseline_korean_pos/18165.pt", "5_epoch (18165)", "single_char"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "out_baseline_char_bpe/best_ckpt.pt", "best_ckpt (71000)", "single_bpe"),
    ("Baseline Pure Byte Tokenizer (V=256)", "out_baseline_pure_byte/best_ckpt.pt", "best_ckpt (106000)", "single_byte"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "out_three_hot_ind_char_byte/best_ckpt.pt", "best_ckpt (72000)", "three_hot_ind"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "out_three_hot_cond_char_byte/best_ckpt.pt", "best_ckpt (65000)", "three_hot_cond"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "out_mc_full_pos_weighted_byte/best_ckpt.pt", "best_ckpt (72000)", "mc_full_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "out_mc_full_pos_unweighted_byte/best_ckpt.pt", "best_ckpt (65000)", "mc_full_pos_unweighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "out_mc_coarse_pos_weighted_byte/15_epochs.pt", "15_epochs (54496)", "mc_coarse_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "out_mc_coarse_pos_unweighted_byte/20_epochs.pt", "20_epochs (72661)", "mc_coarse_pos_unweighted"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "out_mc_nochar_full_pos_new_weighted/15_epochs.pt", "15_epochs (33931)", "nochar_full_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_unweighted/15_epochs.pt", "15_epochs (33931)", "nochar_full_pos_unweighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "best_ckpt (45000)", "nochar_coarse_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_unweighted/15_epochs.pt", "15_epochs (33931)", "nochar_coarse_pos_unweighted"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_h11/20_epochs.pt", "20_epochs (45542)", "charhead_full_h11_with"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_h11/20_epochs.pt", "20_epochs (45542)", "charhead_full_h11_without"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11) (WITHOUT lm_head_24, Joint Factor-Product)", "out_mc_nochar_full_pos_charhead_h11/20_epochs.pt", "20_epochs (45542)", "charhead_full_h11_joint"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_full_unweighted_with"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_full_unweighted_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11/3_epochs.pt", "3_epoch (7088)", "charhead_coarse_h11_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_h11/3_epochs.pt", "3_epoch (7088)", "charhead_coarse_h11_without"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITH lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_coarse_unweighted_with"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted) (WITHOUT lm_head_24)", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34082)", "charhead_coarse_unweighted_without"),
]


def load_model(ckpt_path: str) -> Tuple[GPT, GPTConfig]:
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    m_args = {k: v for k, v in m_args.items() if k in allowed}
    m_args["dropout"] = 0.0
    conf = GPTConfig(**m_args)
    model = GPT(conf)
    sd = checkpoint["model"]
    prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    return model, conf


def get_corpus_syllable_distribution() -> Tuple[List[str], List[str], Dict[int, List[str]]]:
    corpus_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "input.txt")
    counts = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            for c in line:
                if 0xAC00 <= ord(c) <= 0xD7A3:
                    counts[c] += 1

    all_syllables = [chr(code) for code in range(0xAC00, 0xD7A3 + 1)]
    seen_syllables = [c for c in all_syllables if counts[c] > 0]
    unseen_syllables = [c for c in all_syllables if counts[c] == 0]
    seen_syllables.sort(key=lambda c: counts[c], reverse=True)

    dec_size = len(seen_syllables) // 10
    seen_deciles: Dict[int, List[str]] = {}
    for d in range(10):
        dec_num = d + 1
        start = d * dec_size
        end = (d + 1) * dec_size if d < 9 else len(seen_syllables)
        seen_deciles[dec_num] = seen_syllables[start:end]

    return seen_syllables, unseen_syllables, seen_deciles


def load_tokenizers():
    with open("out_baseline_korean_pos_muon/meta.pkl", "rb") as f:
        meta_char = pickle.load(f)
    with open("data/korean_pos_mc/char_bpe/meta.pkl", "rb") as f:
        meta_bpe = pickle.load(f)
    enc_bpe, dec_bpe = get_tokenizer_functions(meta_bpe)
    with open("out_baseline_pure_byte_muon/meta.pkl", "rb") as f:
        meta_byte = pickle.load(f)
    enc_byte, dec_byte = get_tokenizer_functions(meta_byte)

    stois_three_hot = []
    for lp in ["korean_pos_mc/char_byte", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong"]:
        with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
            stois_three_hot.append(pickle.load(f)["stoi"])

    from run_all_muon_20ep_suite import COMMON_CHAR_LANES
    stois_mc_full = []
    for lp in COMMON_CHAR_LANES + ["korean_pos_mc/pos_full", "korean_pos_mc/char_byte"]:
        with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
            stois_mc_full.append(pickle.load(f)["stoi"])

    stois_mc_coarse = []
    for lp in COMMON_CHAR_LANES + ["korean_pos_mc/pos", "korean_pos_mc/char_byte"]:
        with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
            stois_mc_coarse.append(pickle.load(f)["stoi"])

    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_hybrid_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_hybrid_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_factorizer_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
    tok_factorizer_coarse = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")

    return {
        "meta_char": meta_char,
        "meta_bpe": meta_bpe,
        "enc_bpe": enc_bpe,
        "dec_bpe": dec_bpe,
        "meta_byte": meta_byte,
        "enc_byte": enc_byte,
        "dec_byte": dec_byte,
        "stois_three_hot": stois_three_hot,
        "stois_mc_full": stois_mc_full,
        "stois_mc_coarse": stois_mc_coarse,
        "tok_hybrid_full": tok_hybrid_full,
        "tok_hybrid_coarse": tok_hybrid_coarse,
        "tok_factorizer_full": tok_factorizer_full,
        "tok_factorizer_coarse": tok_factorizer_coarse,
    }


def eval_syllable_bpc(model: GPT, conf: GPTConfig, m_type: str, tok: dict, s: str) -> float:
    code = ord(s) - S_BASE
    L = code // N_COUNT
    V = (code % N_COUNT) // T_COUNT
    T = code % T_COUNT
    prefix = "단어: "

    if m_type == "single_char":
        stoi = tok["meta_char"]["stoi"]
        pref_tokens = [stoi.get(c, 0) for c in prefix]
        inp = torch.tensor([pref_tokens], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(inp)
        lp = F.log_softmax(logits[0, -1, :], dim=-1)
        tok_id = stoi.get(s, 0)
        nll = -lp[tok_id].item()
        if tok_id == 0:
            # Lossless Shannon coding penalty for uniform prior over 8,901 OOV syllables mapped to <unk>
            nll += math.log(8901)
        return nll / math.log(2)

    elif m_type == "single_bpe":
        enc_bpe = tok["enc_bpe"]
        pref_ids = enc_bpe(prefix)
        full_ids = enc_bpe(prefix + s)
        s_ids = full_ids[len(pref_ids):]
        if not s_ids:
            return 15.0
        nll = 0.0
        curr = pref_ids.copy()
        for target_id in s_ids:
            inp = torch.tensor([curr], device=DEVICE)
            with torch.no_grad():
                logits, _ = model(inp)
            lp = F.log_softmax(logits[0, -1, :], dim=-1)
            nll += -lp[target_id].item()
            curr.append(target_id)
        return nll / math.log(2)

    elif m_type == "single_byte":
        enc_byte = tok["enc_byte"]
        pref_b = enc_byte(prefix)
        s_b = list(s.encode("utf-8"))
        nll = 0.0
        curr = pref_b.copy()
        for b in s_b:
            inp = torch.tensor([curr], device=DEVICE)
            with torch.no_grad():
                logits, _ = model(inp)
            lp = F.log_softmax(logits[0, -1, :], dim=-1)
            nll += -lp[b].item()
            curr.append(b)
        return nll / math.log(2)

    elif m_type.startswith("three_hot"):
        tok_f = tok["tok_factorizer_full"]
        stois = tok["stois_three_hot"]
        prompt = prefix
        seq = tok_f.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        t_lists = [[] for _ in range(4)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
            t_lists[0].append(byte_id)
            for li, fi in enumerate([1, 2, 3], start=1):
                tc = tok_f.token_for(fi, idx[fi])
                t_lists[li].append(stois[li].get(tc, 0))
        inp = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        tc_l = tok_f.token_for(1, L)
        tc_v = tok_f.token_for(2, V)
        tc_t = tok_f.token_for(3, T)
        tid_l = min(stois[1].get(tc_l, 0), vocab_sizes[1] - 1)
        tid_v = min(stois[2].get(tc_v, 0), vocab_sizes[2] - 1)
        tid_t = min(stois[3].get(tc_t, 0), vocab_sizes[3] - 1)

        is_cond = getattr(conf, "three_hot_conditional", False) or m_type == "three_hot_cond"
        if is_cond:
            tgt_lists = [l.copy() for l in t_lists]
            tgt_lists[1][-1] = tid_l
            tgt_lists[2][-1] = tid_v
            tgt_lists[3][-1] = tid_t
            tg = {f"lane_{i}": torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(tgt_lists)}
            with torch.no_grad():
                logits_list, _ = model(token_dict=td, target_dict=tg)
        else:
            with torch.no_grad():
                logits_list, _ = model(token_dict=td)

        lp_l = F.log_softmax(logits_list[1][0, -1, :], dim=-1)[tid_l].item()
        lp_v = F.log_softmax(logits_list[2][0, -1, :], dim=-1)[tid_v].item()
        lp_t = F.log_softmax(logits_list[3][0, -1, :], dim=-1)[tid_t].item()
        nll = -(lp_l + lp_v + lp_t)
        return nll / math.log(2)

    elif m_type.startswith("mc_"):
        is_full = "full" in m_type
        stois = tok["stois_mc_full"] if is_full else tok["stois_mc_coarse"]
        tok_f = tok["tok_factorizer_full"] if is_full else tok["tok_factorizer_coarse"]
        prompt = prefix
        seq = tok_f.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        n_l = len(vocab_sizes)
        t_lists = [[] for _ in range(n_l)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            for i in range(min(24, n_l - 1)):
                tc = tok_f.token_for(i, idx[i])
                t_lists[i].append(stois[i].get(tc, 0))
            if n_l > 24:
                byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                t_lists[24].append(byte_id)
        inp = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td)
        tc_l = tok_f.token_for(1, L)
        tc_v = tok_f.token_for(2, V)
        tc_t = tok_f.token_for(3, T)
        tid_l = min(stois[1].get(tc_l, 0), logits_list[1].size(-1) - 1)
        tid_v = min(stois[2].get(tc_v, 0), logits_list[2].size(-1) - 1)
        tid_t = min(stois[3].get(tc_t, 0), logits_list[3].size(-1) - 1)
        lp_l = F.log_softmax(logits_list[1][0, -1, :], dim=-1)[tid_l].item()
        lp_v = F.log_softmax(logits_list[2][0, -1, :], dim=-1)[tid_v].item()
        lp_t = F.log_softmax(logits_list[3][0, -1, :], dim=-1)[tid_t].item()
        nll = -(lp_l + lp_v + lp_t)
        return nll / math.log(2)

    elif m_type.startswith("nochar_") or m_type.startswith("charhead_"):
        is_full = "full" in m_type
        tok_h = tok["tok_hybrid_full"] if is_full else tok["tok_hybrid_coarse"]
        prompt = prefix
        steps = tok_h.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        n_lanes = len(vocab_sizes)
        inp = [torch.tensor([[step[i] for step in steps]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i in range(min(24, n_lanes))]
        if n_lanes >= 25:
            char_lane_inp = [((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1 if step[0] == tok_h.hangul_id else 0 for step in steps]
            inp.append(torch.tensor([char_lane_inp], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[24] - 1))

        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td)
        
        # Only use lane 24 if strictly WITH lm_head_24
        if "charhead" in m_type and m_type.endswith("_with") and n_lanes >= 25:
            target_char_id = min(code + 1, logits_list[24].size(-1) - 1)
            lp_char = F.log_softmax(logits_list[24][0, -1, :], dim=-1)
            nll = -lp_char[target_char_id].item()
            return nll / math.log(2)
        else:
            target_step = tok_h.encode_text(s)[0]
            target_l = min(target_step[1], logits_list[1].size(-1) - 1)
            target_v = min(target_step[2], logits_list[2].size(-1) - 1)
            target_t = min(target_step[3], logits_list[3].size(-1) - 1)
            lp_l = F.log_softmax(logits_list[1][0, -1, :], dim=-1)[target_l].item()
            lp_v = F.log_softmax(logits_list[2][0, -1, :], dim=-1)[target_v].item()
            lp_t = F.log_softmax(logits_list[3][0, -1, :], dim=-1)[target_t].item()
            nll = -(lp_l + lp_v + lp_t)
            return nll / math.log(2)

    return 20.0


def eval_syllable_exact_match(model: GPT, conf: GPTConfig, m_type: str, tok: dict, s: str) -> bool:
    code = ord(s) - S_BASE
    L = code // N_COUNT
    V = (code % N_COUNT) // T_COUNT
    T = code % T_COUNT
    prefix = "단어: "

    if m_type == "single_char":
        return False

    elif m_type == "single_byte":
        enc_byte = tok["enc_byte"]
        pref_b = enc_byte(prefix)
        s_b = list(s.encode("utf-8"))
        inp_seq = list(pref_b)
        generated_bytes = []
        for _ in range(len(s_b)):
            inp = torch.tensor([inp_seq], device=DEVICE)
            with torch.no_grad():
                logits, _ = model(inp)
            next_b = logits[0, -1, :].argmax().item()
            generated_bytes.append(next_b)
            inp_seq.append(next_b)
        return generated_bytes == s_b

    elif m_type == "single_bpe":
        enc_bpe = tok["enc_bpe"]
        dec_bpe = tok["dec_bpe"]
        pref_tokens = enc_bpe(prefix)
        inp = torch.tensor([pref_tokens], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(inp)
        next_tok = logits[0, -1, :].argmax().item()
        gen_str = dec_bpe([next_tok])
        return gen_str == s

    elif m_type.startswith("three_hot"):
        tok_f = tok["tok_factorizer_full"]
        stois = tok["stois_three_hot"]
        seq = tok_f.encode_text(prefix)
        vocab_sizes = conf.vocab_sizes
        t_lists = [[] for _ in range(4)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
            t_lists[0].append(byte_id)
            for li, fi in enumerate([1, 2, 3], start=1):
                tc = tok_f.token_for(fi, idx[fi])
                t_lists[li].append(stois[li].get(tc, 0))
        inp = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td)
        pred_l = logits_list[1][0, -1, :].argmax().item()
        pred_v = logits_list[2][0, -1, :].argmax().item()
        pred_t = logits_list[3][0, -1, :].argmax().item()
        tc_l = tok_f.token_for(1, L)
        tc_v = tok_f.token_for(2, V)
        tc_t = tok_f.token_for(3, T)
        tid_l = stois[1].get(tc_l, 0)
        tid_v = stois[2].get(tc_v, 0)
        tid_t = stois[3].get(tc_t, 0)
        return (pred_l == tid_l and pred_v == tid_v and pred_t == tid_t)

    elif m_type.startswith("mc_"):
        is_full = "full" in m_type
        stois = tok["stois_mc_full"] if is_full else tok["stois_mc_coarse"]
        tok_f = tok["tok_factorizer_full"] if is_full else tok["tok_factorizer_coarse"]
        seq = tok_f.encode_text(prefix)
        vocab_sizes = conf.vocab_sizes
        n_l = len(vocab_sizes)
        t_lists = [[] for _ in range(n_l)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            for i in range(min(24, n_l - 1)):
                tc = tok_f.token_for(i, idx[i])
                t_lists[i].append(stois[i].get(tc, 0))
            if n_l > 24:
                byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                t_lists[24].append(byte_id)
        inp = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td)
        pred_l = logits_list[1][0, -1, :].argmax().item()
        pred_v = logits_list[2][0, -1, :].argmax().item()
        pred_t = logits_list[3][0, -1, :].argmax().item()
        tc_l = tok_f.token_for(1, L)
        tc_v = tok_f.token_for(2, V)
        tc_t = tok_f.token_for(3, T)
        tid_l = stois[1].get(tc_l, 0)
        tid_v = stois[2].get(tc_v, 0)
        tid_t = stois[3].get(tc_t, 0)
        return (pred_l == tid_l and pred_v == tid_v and pred_t == tid_t)

    elif m_type.startswith("nochar_") or m_type.startswith("charhead_"):
        is_full = "full" in m_type
        tok_h = tok["tok_hybrid_full"] if is_full else tok["tok_hybrid_coarse"]
        steps = tok_h.encode_text(prefix)
        vocab_sizes = conf.vocab_sizes
        n_lanes = len(vocab_sizes)
        inp = [torch.tensor([[step[i] for step in steps]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i in range(min(24, n_lanes))]
        if n_lanes >= 25:
            char_lane = [((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1 if step[0] == tok_h.hangul_id else 0 for step in steps]
            inp.append(torch.tensor([char_lane], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[24] - 1))
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td)
        
        if "charhead" in m_type and m_type.endswith("_with") and n_lanes >= 25:
            pred_code = logits_list[24][0, -1, 1:11173].argmax().item()
            return pred_code == code
        elif "joint" in m_type:
            log_p_L = F.log_softmax(logits_list[1][0, -1, 1:20], dim=-1)
            log_p_V = F.log_softmax(logits_list[2][0, -1, 1:22], dim=-1)
            log_p_T = F.log_softmax(logits_list[3][0, -1, 0:28], dim=-1)
            joint_log_p = (log_p_L.view(19, 1, 1) + log_p_V.view(1, 21, 1) + log_p_T.view(1, 1, 28)).view(11172)
            best_syl_idx = torch.argmax(joint_log_p).item()
            return best_syl_idx == code
        else:
            pred_l = logits_list[1][0, -1, 1:20].argmax().item() + 1
            pred_v = logits_list[2][0, -1, 1:22].argmax().item() + 1
            pred_t = logits_list[3][0, -1, 0:28].argmax().item()
            target_step = tok_h.encode_text(s)[0]
            return (pred_l == target_step[1] and pred_v == target_step[2] and pred_t == target_step[3])

    return False


def eval_syllable_minimal_pair(model: GPT, conf: GPTConfig, m_type: str, tok: dict, s: str) -> bool:
    if m_type == "single_char" and tok["meta_char"]["stoi"].get(s, 0) == 0:
        return False

    code = ord(s) - S_BASE
    L = code // N_COUNT
    V = (code % N_COUNT) // T_COUNT
    T = code % T_COUNT

    distr_codes = [
        ((L + 1) % 19) * N_COUNT + V * T_COUNT + T,
        L * N_COUNT + ((V + 1) % 21) * T_COUNT + T,
        L * N_COUNT + V * T_COUNT + ((T + 1) % 28)
    ]
    candidates = [s] + [chr(S_BASE + dc) for dc in distr_codes]

    cand_nlls = []
    for cand_s in candidates:
        bpc = eval_syllable_bpc(model, conf, m_type, tok, cand_s)
        cand_nlls.append(bpc if bpc is not None else 999.0)

    return cand_nlls[0] < min(cand_nlls[1:])


def eval_prompt_oov_bpc(model: GPT, conf: GPTConfig, m_type: str, tok: dict, prompt: str, oov_chars: set) -> float:
    oov_positions = [i for i, c in enumerate(prompt) if c in oov_chars]
    if not oov_positions:
        return None

    if m_type == "single_char":
        stoi = tok["meta_char"]["stoi"]
        tokens = [stoi.get(c, 0) for c in prompt]
        inp = torch.tensor([tokens[:-1]], device=DEVICE)
        tgt = torch.tensor([tokens[1:]], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(inp, targets=tgt)
        lp = F.log_softmax(logits[0], dim=-1)
        char_bits = []
        for pos in range(1, len(prompt)):
            if prompt[pos] in oov_chars:
                b = -lp[pos - 1, tokens[pos]].item() / math.log(2)
                char_bits.append(b)
        return float(np.mean(char_bits)) if char_bits else None

    elif m_type == "single_byte":
        prompt_bytes = list(prompt.encode("utf-8"))
        inp = torch.tensor([prompt_bytes[:-1]], device=DEVICE)
        tgt = torch.tensor([prompt_bytes[1:]], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(inp, targets=tgt)
        lp = F.log_softmax(logits[0], dim=-1)
        curr_b = 0
        char_bits = []
        for pos, c in enumerate(prompt):
            c_b = list(c.encode("utf-8"))
            start = curr_b
            end = curr_b + len(c_b)
            curr_b = end
            if c in oov_chars and start > 0:
                c_bit_sum = 0.0
                for b_pos in range(start, end):
                    c_bit_sum += -lp[b_pos - 1, prompt_bytes[b_pos]].item() / math.log(2)
                char_bits.append(c_bit_sum)
        return float(np.mean(char_bits)) if char_bits else None

    elif m_type == "single_bpe":
        enc_bpe = tok["enc_bpe"]
        meta_bpe = tok["meta_bpe"]
        bpe_ids = enc_bpe(prompt)
        inp = torch.tensor([bpe_ids[:-1]], device=DEVICE)
        tgt = torch.tensor([bpe_ids[1:]], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(inp, targets=tgt)
        lp = F.log_softmax(logits[0], dim=-1)
        char_bits = []
        for pos in range(1, len(bpe_ids)):
            piece = meta_bpe["itos"].get(bpe_ids[pos], "")
            is_oov = isinstance(piece, bytes) or any(ch in oov_chars for ch in str(piece))
            if is_oov:
                b = -lp[pos - 1, bpe_ids[pos]].item() / math.log(2)
                char_bits.append(b)
        return float(np.mean(char_bits)) if char_bits else None

    elif m_type.startswith("three_hot"):
        tok_f = tok["tok_factorizer_full"]
        stois = tok["stois_three_hot"]
        seq = tok_f.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        t_lists = [[] for _ in range(4)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
            t_lists[0].append(byte_id)
            for li, fi in enumerate([1, 2, 3], start=1):
                tc = tok_f.token_for(fi, idx[fi])
                t_lists[li].append(stois[li].get(tc, 0))
        inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td, target_dict=tg)

        char_bits = []
        for pos in range(1, len(prompt)):
            c = prompt[pos]
            if c in oov_chars:
                if 0xAC00 <= ord(c) <= 0xD7A3:
                    tid_l = min(t_lists[1][pos], logits_list[1].size(-1) - 1)
                    tid_v = min(t_lists[2][pos], logits_list[2].size(-1) - 1)
                    tid_t = min(t_lists[3][pos], logits_list[3].size(-1) - 1)
                    lp_l = F.log_softmax(logits_list[1][0, pos - 1, :], dim=-1)[tid_l].item()
                    lp_v = F.log_softmax(logits_list[2][0, pos - 1, :], dim=-1)[tid_v].item()
                    lp_t = F.log_softmax(logits_list[3][0, pos - 1, :], dim=-1)[tid_t].item()
                    char_bits.append(-(lp_l + lp_v + lp_t) / math.log(2))
                else:
                    byte_lane = 0
                    target_byte = min(t_lists[byte_lane][pos], logits_list[byte_lane].size(-1) - 1)
                    lp = F.log_softmax(logits_list[byte_lane][0, pos - 1, :], dim=-1)[target_byte].item()
                    char_bits.append(-lp / math.log(2))
        return float(np.mean(char_bits)) if char_bits else None

    elif m_type.startswith("mc_"):
        is_full = "full" in m_type
        stois = tok["stois_mc_full"] if is_full else tok["stois_mc_coarse"]
        tok_f = tok["tok_factorizer_full"] if is_full else tok["tok_factorizer_coarse"]
        seq = tok_f.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        n_l = len(vocab_sizes)
        t_lists = [[] for _ in range(n_l)]
        for item in seq:
            ch = item["char"]
            idx = item["indices"]
            for i in range(min(24, n_l - 1)):
                tc = tok_f.token_for(i, idx[i])
                t_lists[i].append(stois[i].get(tc, 0))
            if n_l > 24:
                byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                t_lists[24].append(byte_id)
        inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td, target_dict=tg)

        char_bits = []
        for pos in range(1, len(prompt)):
            c = prompt[pos]
            if c in oov_chars:
                if 0xAC00 <= ord(c) <= 0xD7A3:
                    tid_l = min(t_lists[1][pos], logits_list[1].size(-1) - 1)
                    tid_v = min(t_lists[2][pos], logits_list[2].size(-1) - 1)
                    tid_t = min(t_lists[3][pos], logits_list[3].size(-1) - 1)
                    lp_l = F.log_softmax(logits_list[1][0, pos - 1, :], dim=-1)[tid_l].item()
                    lp_v = F.log_softmax(logits_list[2][0, pos - 1, :], dim=-1)[tid_v].item()
                    lp_t = F.log_softmax(logits_list[3][0, pos - 1, :], dim=-1)[tid_t].item()
                    char_bits.append(-(lp_l + lp_v + lp_t) / math.log(2))
                else:
                    byte_lane = 24 if n_l > 24 else 0
                    target_byte = min(t_lists[byte_lane][pos], logits_list[byte_lane].size(-1) - 1)
                    lp = F.log_softmax(logits_list[byte_lane][0, pos - 1, :], dim=-1)[target_byte].item()
                    char_bits.append(-lp / math.log(2))
        return float(np.mean(char_bits)) if char_bits else None

    elif m_type.startswith("nochar_") or m_type.startswith("charhead_"):
        is_full = "full" in m_type
        tok_h = tok["tok_hybrid_full"] if is_full else tok["tok_hybrid_coarse"]
        steps = tok_h.encode_text(prompt)
        vocab_sizes = conf.vocab_sizes
        n_lanes = len(vocab_sizes)
        inp = [torch.tensor([[s[i] for s in steps[:-1]]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i in range(min(24, n_lanes))]
        tgt = [torch.tensor([[s[i] for s in steps[1:]]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i in range(min(24, n_lanes))]
        if n_lanes >= 25:
            char_lane_inp = [((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1 if step[0] == tok_h.hangul_id else 0 for step in steps[:-1]]
            char_lane_tgt = [((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1 if step[0] == tok_h.hangul_id else 0 for step in steps[1:]]
            inp.append(torch.tensor([char_lane_inp], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[24] - 1))
            tgt.append(torch.tensor([char_lane_tgt], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[24] - 1))

        td = {f"lane_{i}": l for i, l in enumerate(inp)}
        tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
        with torch.no_grad():
            logits_list, _ = model(token_dict=td, target_dict=tg)

        char_bits = []
        for pos in range(1, len(steps)):
            step = steps[pos]
            if step[0] == tok_h.hangul_id:
                syl_idx = (step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]
                syl_char = chr(0xAC00 + syl_idx)
                if syl_char in oov_chars:
                    if "charhead" in m_type and m_type.endswith("_with") and n_lanes >= 25:
                        target_char_id = min(syl_idx + 1, logits_list[24].size(-1) - 1)
                        lp = F.log_softmax(logits_list[24][0, pos - 1, :], dim=-1)[target_char_id].item()
                        char_bits.append(-lp / math.log(2))
                    else:
                        target_l = min(step[1], logits_list[1].size(-1) - 1)
                        target_v = min(step[2], logits_list[2].size(-1) - 1)
                        target_t = min(step[3], logits_list[3].size(-1) - 1)
                        lp_l = F.log_softmax(logits_list[1][0, pos - 1, :], dim=-1)[target_l].item()
                        lp_v = F.log_softmax(logits_list[2][0, pos - 1, :], dim=-1)[target_v].item()
                        lp_t = F.log_softmax(logits_list[3][0, pos - 1, :], dim=-1)[target_t].item()
                        char_bits.append(-(lp_l + lp_v + lp_t) / math.log(2))
            else:
                piece = tok_h.sp.id_to_piece(step[0])
                if piece.startswith("<0x") or any(ch in oov_chars for ch in piece):
                    target_spm = min(step[0], logits_list[0].size(-1) - 1)
                    lp = F.log_softmax(logits_list[0][0, pos - 1, :], dim=-1)[target_spm].item()
                    char_bits.append(-lp / math.log(2))

        return float(np.mean(char_bits)) if char_bits else None


def evaluate_suite(suite_name: str, model_list: list, out_json_path: str, tokenizers: dict, seen_deciles: dict, unseen_syllables: list, oov_prompts: list, all_oov_chars: set, eval_cache: dict):
    print(f"\n================================================================================")
    print(f" SUITE: {suite_name} ({len(model_list)} models)")
    print(f"================================================================================")
    
    rng = np.random.RandomState(42)
    sample_seen: Dict[int, List[str]] = {}
    for d in [1, 3, 5, 7, 10]:
        sample_seen[d] = list(rng.choice(seen_deciles[d], size=min(30, len(seen_deciles[d])), replace=False))
    
    sample_unseen = list(rng.choice(unseen_syllables, size=100, replace=False))

    results_all: Dict[str, Any] = {}

    for idx, (model_name, ckpt_path, ckpt_label, m_type) in enumerate(model_list, 1):
        print(f"\n[{idx}/{len(model_list)}] Evaluating: {model_name}")
        print(f"  Checkpoint: {ckpt_path} ({ckpt_label})")
        
        cache_key = (ckpt_path, m_type)
        if cache_key in eval_cache:
            print(f"  -> Reusing cached evaluation from identical checkpoint & configuration!")
            res = dict(eval_cache[cache_key])
            res["checkpoint"] = f"{ckpt_label} ({ckpt_path})"
            results_all[model_name] = res
            continue

        t0 = time.time()
        model, conf = load_model(ckpt_path)

        # 1. Seen Decile BPC
        decile_bpcs: Dict[str, float] = {}
        for d in [1, 3, 5, 7, 10]:
            bpcs = []
            for s in sample_seen[d]:
                bpc = eval_syllable_bpc(model, conf, m_type, tokenizers, s)
                if bpc is not None and not math.isnan(bpc):
                    bpcs.append(bpc)
            decile_bpcs[f"decile_{d}"] = round(float(np.mean(bpcs)), 2) if bpcs else None

        # 2. Held-Out Syllables
        heldout_bpcs = []
        exact_matches = 0
        discr_matches = 0
        total_heldout = len(sample_unseen)

        for s in sample_unseen:
            bpc = eval_syllable_bpc(model, conf, m_type, tokenizers, s)
            if bpc is not None and not math.isnan(bpc):
                heldout_bpcs.append(bpc)
            if eval_syllable_exact_match(model, conf, m_type, tokenizers, s):
                exact_matches += 1
            if eval_syllable_minimal_pair(model, conf, m_type, tokenizers, s):
                discr_matches += 1

        heldout_res = {
            "bpc": round(float(np.mean(heldout_bpcs)), 2) if heldout_bpcs else None,
            "exact_top1_acc_pct": round(exact_matches / total_heldout * 100.0, 2),
            "minimal_pair_discr_acc_pct": round(discr_matches / total_heldout * 100.0, 2),
            "evaluated_heldout_count": total_heldout,
            "total_heldout_census": len(unseen_syllables),
        }

        # 3. Non-Hangul OOV Unicode Stress Prompts
        oov_prompt_bpcs = []
        for p in oov_prompts:
            bpc = eval_prompt_oov_bpc(model, conf, m_type, tokenizers, p, all_oov_chars)
            if bpc is not None and not math.isnan(bpc):
                oov_prompt_bpcs.append(bpc)

        oov_res = {
            "bpc": round(float(np.mean(oov_prompt_bpcs)), 2) if oov_prompt_bpcs else None,
            "prompts_evaluated": len(oov_prompt_bpcs),
        }

        elapsed = time.time() - t0
        print(f"  Seen D1: {decile_bpcs['decile_1']} | D10: {decile_bpcs['decile_10']} BPC")
        print(f"  Held-Out ({len(unseen_syllables):,} set): BPC={heldout_res['bpc']} | Exact={heldout_res['exact_top1_acc_pct']}% | 4-Way={heldout_res['minimal_pair_discr_acc_pct']}%")
        print(f"  OOV Stress BPC: {oov_res['bpc']} (eval took {elapsed:.1f}s)")

        model_entry = {
            "checkpoint": f"{ckpt_label} ({ckpt_path})",
            "model_type": m_type,
            "seen_deciles_bpc": decile_bpcs,
            "held_out_syllables": heldout_res,
            "non_hangul_oov_stress": oov_res,
            "eval_time_sec": round(elapsed, 1),
        }
        results_all[model_name] = model_entry
        eval_cache[cache_key] = model_entry

        del model
        torch.cuda.empty_cache()

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {out_json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", "muon_val", "adamw_val", "adamw_down"], default="all")
    args = parser.parse_args()

    seen_syllables, unseen_syllables, seen_deciles = get_corpus_syllable_distribution()
    oov_prompts_file = os.path.join(REPO_ROOT, "benchmarks", "prompts", "all_oov_prompts.txt")
    with open(oov_prompts_file, "r", encoding="utf-8") as f:
        oov_prompts = [line.strip() for line in f if line.strip()]

    all_oov_chars = set()
    for p in oov_prompts:
        for c in p:
            if ord(c) > 127:
                if 0xAC00 <= ord(c) <= 0xD7A3:
                    if c in unseen_syllables:
                        all_oov_chars.add(c)
                else:
                    all_oov_chars.add(c)

    tokenizers = load_tokenizers()
    eval_cache: Dict[Tuple[str, str], Any] = {}

    suites_to_run = []
    if args.suite in ["all", "muon_val"]:
        suites_to_run.append(("Muon Best Validation Loss Checkpoints", MUON_VAL_LOSS_MODELS, "results_oov_tail_bpc_muon_val_loss.json"))
    if args.suite in ["all", "adamw_val"]:
        suites_to_run.append(("AdamW Best Validation Loss Checkpoints", ADAMW_VAL_LOSS_MODELS, "results_oov_tail_bpc_adamw_val_loss.json"))
    if args.suite in ["all", "adamw_down"]:
        suites_to_run.append(("AdamW Best Downstream Checkpoints", ADAMW_DOWNSTREAM_MODELS, "results_oov_tail_bpc_adamw_downstream.json"))

    for suite_name, m_list, json_p in suites_to_run:
        full_json_p = os.path.join(REPO_ROOT, json_p)
        evaluate_suite(suite_name, m_list, full_json_p, tokenizers, seen_deciles, unseen_syllables, oov_prompts, all_oov_chars, eval_cache)

    print("\nAll requested evaluations complete!")


if __name__ == "__main__":
    main()
