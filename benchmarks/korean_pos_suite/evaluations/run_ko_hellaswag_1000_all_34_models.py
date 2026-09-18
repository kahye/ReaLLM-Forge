#!/usr/bin/env python3
"""Evaluate 1,000 Ko-HellaSwag validation examples across all 34 models (17 AdamW + 17 Muon best_ckpt)."""
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
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.run_ko_hellaswag import (
    ALL_NORM_MODES,
    DEFAULT_MULTICONTEXT_LANES,
    MulticontextHangulPosEncoder,
    _build_context,
    _load_checkpoint,
    compute_unigram_logprobs,
)
from sample import get_tokenizer_functions

ADAMW_MODELS = [
    ("baseline_character", "out_baseline_korean_pos/best_ckpt.pt", "single_char"),
    ("baseline_char_bpe", "out_baseline_char_bpe/best_ckpt.pt", "single_bpe"),
    ("baseline_pure_byte", "out_baseline_pure_byte/best_ckpt.pt", "single_byte"),
    ("three_hot_ind_char_byte", "out_three_hot_ind_char_byte/best_ckpt.pt", "mc_three_hot"),
    ("three_hot_cond_char_byte", "out_three_hot_cond_char_byte/best_ckpt.pt", "mc_three_hot"),
    ("mc_full_pos_weighted_byte", "out_mc_full_pos_weighted_byte/best_ckpt.pt", "mc_25_full"),
    ("mc_full_pos_unweighted_byte", "out_mc_full_pos_unweighted_byte/best_ckpt.pt", "mc_25_full"),
    ("mc_coarse_pos_weighted_byte", "out_mc_coarse_pos_weighted_byte/best_ckpt.pt", "mc_25_coarse"),
    ("mc_coarse_pos_unweighted_byte", "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt", "mc_25_coarse"),
    ("nochar_full_pos_new_weighted", "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt", "mc_24_full"),
    ("nochar_full_pos_unweighted", "out_mc_nochar_full_pos_unweighted/best_ckpt.pt", "mc_24_full"),
    ("nochar_coarse_pos_new_weighted", "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "mc_24_coarse"),
    ("nochar_coarse_pos_unweighted", "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt", "mc_24_coarse"),
    ("nochar_full_pos_charhead_h11", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_full_pos_charhead_unweighted", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_coarse_pos_charhead_h11", "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt", "mc_25_charhead_coarse"),
    ("nochar_coarse_pos_charhead_unweighted", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "mc_25_charhead_coarse"),
]

MUON_MODELS = [
    (m_id, ckpt.replace("/best_ckpt.pt", "_muon/best_ckpt.pt"), tok_key)
    for m_id, ckpt, tok_key in ADAMW_MODELS
]


def load_single_encoder(meta_rel_path: str):
    meta_path = os.path.join(REPO_ROOT, meta_rel_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    encode_fn, _ = get_tokenizer_functions(meta)
    return encode_fn


def build_tokenizer_map() -> Dict[str, Any]:
    """Instantiate each unique encoder once."""
    print("Initializing unique tokenizers...")
    tok_map = {}
    tok_map["single_char"] = ("single", load_single_encoder("data/korean_pos_mc/char/meta.pkl"))
    tok_map["single_bpe"] = ("single", load_single_encoder("data/korean_pos_mc/char_bpe/meta.pkl"))
    tok_map["single_byte"] = ("single", load_single_encoder("data/korean_pos_mc/byte/meta.pkl"))

    # Three-hot (4 lanes)
    tok_map["mc_three_hot"] = (
        "mc",
        MulticontextHangulPosEncoder([
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ]),
    )

    # 25-lane char stream (coarse & full)
    common_char = [
        "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
        "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
        "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
        "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
        "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness",
        "korean_pos_mc/jung_round", "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim",
        "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
    ]
    tok_map["mc_25_full"] = ("mc", MulticontextHangulPosEncoder([*common_char, "korean_pos_mc/pos_full", "korean_pos_mc/char_byte"]))
    tok_map["mc_25_coarse"] = ("mc", MulticontextHangulPosEncoder([*common_char, "korean_pos_mc/pos", "korean_pos_mc/char_byte"]))

    # 24-lane no-char & 25-lane charhead
    common_nochar = [
        "korean_pos_mc_nochar/script_bpe", "korean_pos_mc_nochar/choseong", "korean_pos_mc_nochar/jungseong",
        "korean_pos_mc_nochar/jongseong", "korean_pos_mc_nochar/jung_base1", "korean_pos_mc_nochar/jung_base2",
        "korean_pos_mc_nochar/jung_has_w", "korean_pos_mc_nochar/jung_has_y", "korean_pos_mc_nochar/jung_has_i",
        "korean_pos_mc_nochar/jong_base1", "korean_pos_mc_nochar/jong_base2", "korean_pos_mc_nochar/jong_base3",
        "korean_pos_mc_nochar/choseong_tense", "korean_pos_mc_nochar/choseong_aspirated",
        "korean_pos_mc_nochar/choseong_nasal_liquid", "korean_pos_mc_nochar/choseong_place",
        "korean_pos_mc_nochar/jung_height", "korean_pos_mc_nochar/jung_backness", "korean_pos_mc_nochar/jung_round",
        "korean_pos_mc_nochar/jong_complex", "korean_pos_mc_nochar/has_batchim",
        "korean_pos_mc_nochar/syllable_index_mod", "korean_pos_mc_nochar/codepoint_mod",
    ]
    tok_map["mc_24_full"] = ("mc", MulticontextHangulPosEncoder([*common_nochar, "korean_pos_mc_nochar/pos_full"]))
    tok_map["mc_24_coarse"] = ("mc", MulticontextHangulPosEncoder([*common_nochar, "korean_pos_mc_nochar/pos"]))
    tok_map["mc_25_charhead_full"] = ("mc", MulticontextHangulPosEncoder([*common_nochar, "korean_pos_mc_nochar/pos_full", "korean_pos_mc_nochar/char"]))
    tok_map["mc_25_charhead_coarse"] = ("mc", MulticontextHangulPosEncoder([*common_nochar, "korean_pos_mc_nochar/pos", "korean_pos_mc_nochar/char"]))

    return tok_map


def _worker_hellaswag_chunk(args):
    key, kind, chunk = args
    tok_map = build_tokenizer_map()
    _, enc = tok_map[key]
    out = []
    for ex in chunk:
        ctx_text = _build_context(ex)
        endings = ex.get("endings", [])
        label = int(ex["label"])
        if kind == "single":
            ctx_toks = enc(ctx_text)
            end_toks_list = [enc(e) for e in endings]
            uncond_ctx_toks = enc(" ")
            out.append((ctx_toks, endings, end_toks_list, uncond_ctx_toks, label))
        else:
            ctx_lanes = enc.encode(ctx_text)
            end_lanes_list = [enc.encode(e) for e in endings]
            uncond_lanes_list = [enc.encode(" " + e) for e in endings]
            out.append((ctx_lanes, endings, end_lanes_list, uncond_lanes_list, label))
    return out


def pre_tokenize_dataset(dataset, tok_map: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Pre-tokenize all examples once per unique tokenizer with multiprocessing and disk cache."""
    cache_file = f"/tmp/ko_hellaswag_pretok_{len(dataset)}.pt"
    if os.path.exists(cache_file):
        print(f"Loading pre-tokenized Ko-HellaSwag cache from {cache_file}...")
        return torch.load(cache_file, weights_only=False)

    import multiprocessing as mp
    cache = {}
    base_keys = ["single_char", "single_bpe", "single_byte", "mc_three_hot", "mc_25_full", "mc_25_coarse", "mc_25_charhead_full", "mc_25_charhead_coarse"]
    all_ex = list(dataset)
    n_workers = min(12, mp.cpu_count() or 4)
    chunk_size = max(1, (len(all_ex) + n_workers - 1) // n_workers)
    chunks = [all_ex[i : i + chunk_size] for i in range(0, len(all_ex), chunk_size)]

    for key in base_keys:
        t0 = time.time()
        kind, _ = tok_map[key]
        if len(all_ex) > 1500 and kind != "single":
            with mp.get_context("spawn").Pool(n_workers) as pool:
                res_chunks = pool.map(_worker_hellaswag_chunk, [(key, kind, c) for c in chunks])
            pre_list = [item for sub in res_chunks for item in sub]
        else:
            _, enc = tok_map[key]
            pre_list = []
            for ex in all_ex:
                ctx_text = _build_context(ex)
                endings = ex.get("endings", [])
                label = int(ex["label"])
                if kind == "single":
                    ctx_toks = enc(ctx_text)
                    end_toks_list = [enc(e) for e in endings]
                    uncond_ctx_toks = enc(" ")
                    pre_list.append((ctx_toks, endings, end_toks_list, uncond_ctx_toks, label))
                else:
                    ctx_lanes = enc.encode(ctx_text)
                    end_lanes_list = [enc.encode(e) for e in endings]
                    uncond_lanes_list = [enc.encode(" " + e) for e in endings]
                    pre_list.append((ctx_lanes, endings, end_lanes_list, uncond_lanes_list, label))
        cache[key] = pre_list
        print(f"  Pre-tokenized {key} ({len(dataset)} examples) in {time.time() - t0:.2f}s")

    for src_key, dst_key in [("mc_25_charhead_full", "mc_24_full"), ("mc_25_charhead_coarse", "mc_24_coarse")]:
        derived = []
        for ctx_lanes, endings, end_lanes_list, uncond_lanes_list, label in cache[src_key]:
            c24 = ctx_lanes[:24]
            e24 = [el[:24] for el in end_lanes_list]
            u24 = [ul[:24] for ul in uncond_lanes_list]
            derived.append((c24, endings, e24, u24, label))
        cache[dst_key] = derived
        print(f"  Derived {dst_key} from {src_key}")

    torch.save(cache, cache_file)
    return cache


_SINGLE_BATCH_CACHE = {}
_MC_BATCH_CACHE = {}


def evaluate_single_model_batched(
    model,
    pre_tok: List[Any],
    unigram_dict: Dict[str, float],
    default_unigram_logprob: float,
    block_size: int = 256,
    batch_size: int = 64,
) -> Dict[str, float]:
    vocab_size = getattr(model.config, "vocab_size", None)
    cache_key = (id(pre_tok), block_size, batch_size)
    if cache_key not in _SINGLE_BATCH_CACHE:
        items = []
        for ex_idx, (ctx_toks, endings, end_toks_list, uncond_ctx_toks, label) in enumerate(pre_tok):
            for opt_idx, (ending, end_toks) in enumerate(zip(endings, end_toks_list)):
                if len(end_toks) > block_size:
                    end_toks = end_toks[-block_size:]
                max_ctx_len = max(0, block_size - len(end_toks))
                ctx_trim = ctx_toks[-max_ctx_len:] if max_ctx_len > 0 else []
                full = ctx_trim + end_toks
                ending_start = max(len(ctx_trim) - 1, 0)
                items.append(("cond", ex_idx, opt_idx, full[:-1], full[1:], ending_start, ending))

                uncond_max_ctx_len = max(0, block_size - len(end_toks))
                uncond_ctx_trim = uncond_ctx_toks[-uncond_max_ctx_len:] if uncond_max_ctx_len > 0 else []
                uncond_full = uncond_ctx_trim + end_toks
                uncond_ending_start = max(len(uncond_ctx_trim) - 1, 0)
                items.append(("uncond", ex_idx, opt_idx, uncond_full[:-1], uncond_full[1:], uncond_ending_start, ending))

        batches = []
        for b_start in range(0, len(items), batch_size):
            batch = items[b_start : b_start + batch_size]
            max_len = max(len(x[3]) for x in batch)
            b_in_cpu = torch.tensor([x[3] + [0] * (max_len - len(x[3])) for x in batch], dtype=torch.long)
            b_tg_cpu = torch.tensor([x[4] + [0] * (max_len - len(x[4])) for x in batch], dtype=torch.long)
            meta = [(x[0], x[1], x[2], len(x[3]), x[5], x[6]) for x in batch]
            batches.append((b_in_cpu, b_tg_cpu, meta))
        _SINGLE_BATCH_CACHE[cache_key] = batches

    batches = _SINGLE_BATCH_CACHE[cache_key]
    results_map = {}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for b_in_cpu, b_tg_cpu, meta in batches:
            if vocab_size is not None:
                b_in = b_in_cpu.clamp(0, vocab_size - 1).to("cuda", non_blocking=True)
                b_tg = b_tg_cpu.clamp(0, vocab_size - 1).to("cuda", non_blocking=True)
            else:
                b_in = b_in_cpu.to("cuda", non_blocking=True)
                b_tg = b_tg_cpu.to("cuda", non_blocking=True)

            with ctx_autocast:
                logits, _ = model(b_in, b_tg)
            logprobs = torch.log_softmax(logits, dim=-1)
            lp_full = logprobs.gather(-1, b_tg.unsqueeze(-1)).squeeze(-1).cpu()
            for row_i, (kind, ex_idx, opt_idx, s_len, ending_start, ending) in enumerate(meta):
                results_map[(kind, ex_idx, opt_idx)] = (lp_full[row_i, ending_start:s_len], ending)

    uni_map = _get_unigram_cache(pre_tok, unigram_dict, default_unigram_logprob)
    return _compute_accs_fast(results_map, pre_tok, uni_map)


_UNIGRAM_CACHE = {}


def _get_unigram_cache(pre_tok: List[Any], unigram_dict: Dict[str, float], default_unigram_logprob: float):
    key = id(pre_tok)
    if key not in _UNIGRAM_CACHE:
        uni_map = {}
        for ex_idx, (_, endings, _, _, _) in enumerate(pre_tok):
            for opt_idx, ending in enumerate(endings):
                uni_map[(ex_idx, opt_idx)] = torch.tensor(
                    [unigram_dict.get(c, default_unigram_logprob) for c in ending],
                    dtype=torch.float32,
                )
        _UNIGRAM_CACHE[key] = uni_map
    return _UNIGRAM_CACHE[key]


def _compute_accs_fast(results_map: Dict[Tuple[str, int, int], Tuple[torch.Tensor, str]], pre_tok: List[Any], uni_map: Dict[Tuple[int, int], torch.Tensor]) -> Dict[str, float]:
    correct = {m: 0 for m in ALL_NORM_MODES}
    total = len(pre_tok)
    for ex_idx, (_, endings, _, _, label) in enumerate(pre_tok):
        n_opts = len(endings)
        s_len = [0.0] * n_opts
        s_none = [0.0] * n_opts
        s_plen = [0.0] * n_opts
        s_prior = [0.0] * n_opts
        s_ulen = [0.0] * n_opts
        s_uni = [0.0] * n_opts
        for opt_idx in range(n_opts):
            lp_cond, _ = results_map[("cond", ex_idx, opt_idx)]
            lp_uncond, _ = results_map[("uncond", ex_idx, opt_idx)]
            lp_uni_full = uni_map[(ex_idx, opt_idx)]
            c_len = lp_cond.size(-1)
            c_sum = lp_cond.sum().item()
            s_none[opt_idx] = c_sum
            s_len[opt_idx] = c_sum / max(1, c_len)

            min_len = min(c_len, lp_uncond.size(-1))
            diff_p_sum = (lp_cond[-min_len:].sum() - lp_uncond[-min_len:].sum()).item()
            s_prior[opt_idx] = diff_p_sum
            s_plen[opt_idx] = diff_p_sum / max(1, min_len)

            min_len_u = min(c_len, lp_uni_full.size(-1))
            diff_u_sum = (lp_cond[-min_len_u:].sum() - lp_uni_full[-min_len_u:].sum()).item()
            s_uni[opt_idx] = diff_u_sum
            s_ulen[opt_idx] = diff_u_sum / max(1, min_len_u)

        if max(range(n_opts), key=s_len.__getitem__) == label:
            correct["length"] += 1
        if max(range(n_opts), key=s_none.__getitem__) == label:
            correct["none"] += 1
        if max(range(n_opts), key=s_plen.__getitem__) == label:
            correct["prior_length"] += 1
        if max(range(n_opts), key=s_prior.__getitem__) == label:
            correct["prior"] += 1
        if max(range(n_opts), key=s_ulen.__getitem__) == label:
            correct["unigram_length"] += 1
        if max(range(n_opts), key=s_uni.__getitem__) == label:
            correct["unigram"] += 1

    return {m: correct[m] / total for m in ALL_NORM_MODES}


def evaluate_mc_model_batched(
    model,
    pre_tok: List[Any],
    tok_key: str,
    unigram_dict: Dict[str, float],
    default_unigram_logprob: float,
    block_size: int = 256,
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    is_hybrid = "nochar" in tok_key or "charhead" in tok_key
    has_charhead = "charhead" in tok_key
    n_lanes = len(pre_tok[0][0])

    cache_key = (id(pre_tok), block_size, batch_size)
    if cache_key not in _MC_BATCH_CACHE:
        items = []
        for ex_idx, (ctx_lanes, endings, end_lanes_list, uncond_lanes_list, label) in enumerate(pre_tok):
            for opt_idx, (ending, end_lanes, uncond_lanes) in enumerate(zip(endings, end_lanes_list, uncond_lanes_list)):
                num_end_tokens = len(end_lanes[0])
                if num_end_tokens > block_size:
                    end_lanes = [lane[-block_size:] for lane in end_lanes]
                    num_end_tokens = block_size
                max_ctx_len = max(0, block_size - num_end_tokens)
                ctx_trim = [lane[-max_ctx_len:] if max_ctx_len > 0 else [] for lane in ctx_lanes]
                ctx_trim_len = len(ctx_trim[0])
                full_lanes = [c + e for c, e in zip(ctx_trim, end_lanes)]
                ending_start = max(ctx_trim_len - 1, 0)
                items.append(("cond", ex_idx, opt_idx, [l[:-1] for l in full_lanes], [l[1:] for l in full_lanes], ending_start, ending))

                uncond_seq_len = len(uncond_lanes[0])
                if uncond_seq_len > block_size:
                    uncond_lanes = [lane[-block_size:] for lane in uncond_lanes]
                    uncond_seq_len = len(uncond_lanes[0])
                uncond_ctx_trim_len = max(0, uncond_seq_len - num_end_tokens)
                uncond_ending_start = max(uncond_ctx_trim_len - 1, 0)
                items.append(("uncond", ex_idx, opt_idx, [l[:-1] for l in uncond_lanes], [l[1:] for l in uncond_lanes], uncond_ending_start, ending))

        batches = []
        for b_start in range(0, len(items), batch_size):
            batch = items[b_start : b_start + batch_size]
            max_seq_len = max(len(x[3][0]) for x in batch)
            b_inputs_cpu = [
                torch.tensor([x[3][l_i] + [0] * (max_seq_len - len(x[3][0])) for x in batch], dtype=torch.long)
                for l_i in range(n_lanes)
            ]
            b_targets_cpu = [
                torch.tensor([x[4][l_i] + [0] * (max_seq_len - len(x[4][0])) for x in batch], dtype=torch.long)
                for l_i in range(n_lanes)
            ]
            meta = [(x[0], x[1], x[2], len(x[3][0]), x[5], x[6]) for x in batch]
            batches.append((b_inputs_cpu, b_targets_cpu, meta))
        _MC_BATCH_CACHE[cache_key] = batches

    batches = _MC_BATCH_CACHE[cache_key]
    scoring_modes = ["primary"]
    if is_hybrid:
        scoring_modes.append("jamo_lvt")
    if has_charhead:
        scoring_modes.append("charhead_24")

    results_map = {sm: {} for sm in scoring_modes}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        for b_inputs_cpu, b_targets_cpu, meta in batches:
            b_inputs = []
            b_targets = []
            for lane_i in range(n_lanes):
                if vocab_sizes is not None and lane_i < len(vocab_sizes):
                    b_inputs.append(b_inputs_cpu[lane_i].clamp(0, vocab_sizes[lane_i] - 1).to("cuda", non_blocking=True))
                    b_targets.append(b_targets_cpu[lane_i].clamp(0, vocab_sizes[lane_i] - 1).to("cuda", non_blocking=True))
                else:
                    b_inputs.append(b_inputs_cpu[lane_i].to("cuda", non_blocking=True))
                    b_targets.append(b_targets_cpu[lane_i].to("cuda", non_blocking=True))

            token_dict = {f"lane_{i}": b_inputs[i] for i in range(n_lanes)}
            target_dict = {f"lane_{i}": b_targets[i] for i in range(n_lanes)}
            with ctx_autocast:
                logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)

            head_idx = 0 if is_hybrid else -1
            lp_primary_all = torch.log_softmax(logits_list[head_idx], dim=-1)
            lp_prim_full = lp_primary_all.gather(-1, b_targets[head_idx].unsqueeze(-1)).squeeze(-1)

            if is_hybrid:
                is_hangul_full = (b_targets[head_idx] == 2)
                lp1_full = torch.log_softmax(logits_list[1], dim=-1).gather(-1, b_targets[1].unsqueeze(-1)).squeeze(-1)
                lp2_full = torch.log_softmax(logits_list[2], dim=-1).gather(-1, b_targets[2].unsqueeze(-1)).squeeze(-1)
                lp3_full = torch.log_softmax(logits_list[3], dim=-1).gather(-1, b_targets[3].unsqueeze(-1)).squeeze(-1)
                lp_lvt_full = torch.where(is_hangul_full, lp_prim_full + lp1_full + lp2_full + lp3_full, lp_prim_full).cpu()
            if has_charhead:
                lp24_full = torch.log_softmax(logits_list[24], dim=-1).gather(-1, b_targets[24].unsqueeze(-1)).squeeze(-1)
                lp_ch24_full = torch.where(is_hangul_full, lp_prim_full + lp24_full, lp_prim_full).cpu()
            lp_prim_full = lp_prim_full.cpu()

            for row_i, (kind, ex_idx, opt_idx, s_len, ending_start, ending) in enumerate(meta):
                results_map["primary"][(kind, ex_idx, opt_idx)] = (lp_prim_full[row_i, ending_start:s_len], ending)
                if is_hybrid:
                    results_map["jamo_lvt"][(kind, ex_idx, opt_idx)] = (lp_lvt_full[row_i, ending_start:s_len], ending)
                if has_charhead:
                    results_map["charhead_24"][(kind, ex_idx, opt_idx)] = (lp_ch24_full[row_i, ending_start:s_len], ending)

    uni_map = _get_unigram_cache(pre_tok, unigram_dict, default_unigram_logprob)
    return {sm: _compute_accs_fast(results_map[sm], pre_tok, uni_map) for sm in scoring_modes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="results_ko_hellaswag_1000_all_models.json")
    args = parser.parse_args()

    print(f"Loading Ko-HellaSwag validation split (max_examples={args.max_examples}, seed=1337)...")
    dataset = load_dataset("KETI-AIR/kor_hellaswag", split="validation")
    dataset = dataset.shuffle(seed=1337).select(range(min(args.max_examples, len(dataset))))

    print("Computing unigram logprobs...")
    unigram_dict, default_unigram_logprob = compute_unigram_logprobs(dataset)

    tok_map = build_tokenizer_map()
    pre_tok_cache = pre_tokenize_dataset(dataset, tok_map)

    out_path = os.path.join(REPO_ROOT, args.output_json)
    all_results = {"adamw": {}, "muon": {}, "num_examples": len(dataset)}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("num_examples") == len(dataset):
                all_results["adamw"] = saved.get("adamw", {})
                all_results["muon"] = saved.get("muon", {})
                print(f"Resuming: found {len(all_results['adamw'])} AdamW and {len(all_results['muon'])} Muon models in {args.output_json}")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    for opt_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== EVALUATING {opt_name.upper()} MODELS (N={len(dataset)}) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            if m_id in all_results[opt_name]:
                print(f"  [SKIP] {opt_name.upper()} {m_id} already evaluated.")
                continue
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            t0 = time.time()
            load_args = argparse.Namespace(
                ckpt_path=ckpt_path,
                out_dir=os.path.dirname(ckpt_path),
                config_path=None,
                init_from="resume",
                device="cuda",
                dtype="bfloat16",
                weights_only=False,
                block_size=256,
            )
            model, _ = _load_checkpoint(load_args)
            model.eval().to("cuda")

            pre_tok = pre_tok_cache[tok_key]
            if tok_key.startswith("single_"):
                accs = evaluate_single_model_batched(
                    model, pre_tok, unigram_dict, default_unigram_logprob, block_size=256, batch_size=args.batch_size
                )
                entry = {"primary": accs}
            else:
                entry = evaluate_mc_model_batched(
                    model, pre_tok, tok_key, unigram_dict, default_unigram_logprob, block_size=256, batch_size=args.batch_size
                )

            elapsed = time.time() - t0
            prim = entry["primary"]
            msg = f"  [{opt_name.upper()}] {m_id:38s} | none: {prim['none']*100:5.1f}% | len: {prim['length']*100:5.1f}% | prior: {prim['prior']*100:5.1f}% | unigram: {prim['unigram']*100:5.1f}%"
            if "jamo_lvt" in entry:
                lvt = entry["jamo_lvt"]
                msg += f" || LVT none: {lvt['none']*100:5.1f}% | len: {lvt['length']*100:5.1f}%"
            if "charhead_24" in entry:
                ch = entry["charhead_24"]
                msg += f" || CH24 none: {ch['none']*100:5.1f}% | len: {ch['length']*100:5.1f}%"
            msg += f" ({elapsed:.1f}s)"
            print(msg)

            all_results[opt_name][m_id] = entry
            del model
            torch.cuda.empty_cache()

            out_path = os.path.join(REPO_ROOT, args.output_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved complete 1,000-example evaluation results to {args.output_json}")


if __name__ == "__main__":
    main()
