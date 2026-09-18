#!/usr/bin/env python3
"""
Fast Batched Evaluation of Extreme Noise Resilience (NSMC 80% & UnSmile 80%, N=1,000 examples each)
across all 34 Best-Validation-Loss Checkpoints (17 AdamW + 17 Muon).
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
import json
import math
import os
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.run_ko_hellaswag import (
    MulticontextHangulPosEncoder,
    _load_checkpoint,
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


def corrupt_text(text: str, rng: random.Random, rate: float = 0.8) -> str:
    res = []
    for c in text:
        if rng.random() < rate:
            res.append(chr(ord(c) ^ 1))
        else:
            res.append(c)
    return "".join(res)


def load_single_encoder(meta_rel_path: str):
    meta_path = os.path.join(REPO_ROOT, meta_rel_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    encode_fn, _ = get_tokenizer_functions(meta)
    return encode_fn


def build_tokenizer_map() -> Dict[str, Any]:
    print("Initializing unique tokenizers...")
    tok_map = {}
    tok_map["single_char"] = ("single", load_single_encoder("data/korean_pos_mc/char/meta.pkl"))
    tok_map["single_bpe"] = ("single", load_single_encoder("data/korean_pos_mc/char_bpe/meta.pkl"))
    tok_map["single_byte"] = ("single", load_single_encoder("data/korean_pos_mc/byte/meta.pkl"))

    tok_map["mc_three_hot"] = (
        "mc",
        MulticontextHangulPosEncoder([
            "korean_pos_mc/char_byte",
            "korean_pos_mc/choseong",
            "korean_pos_mc/jungseong",
            "korean_pos_mc/jongseong",
        ]),
    )

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


def load_noisy_datasets_1000(max_examples: int = 1000, seed: int = 1337) -> Dict[str, List[Tuple[str, str]]]:
    rng = random.Random(seed)
    ds_nsmc = load_dataset("Blpeng/nsmc", split="test")
    ds_unsmile = load_dataset("smilegate-ai/kor_unsmile", split="valid")

    nsmc_pairs = []
    for row in ds_nsmc:
        if len(nsmc_pairs) >= max_examples:
            break
        doc = row.get("document", "")
        if doc and len(doc.strip()) >= 2:
            corr_doc = corrupt_text(doc, rng=rng, rate=0.8)
            nsmc_pairs.append((doc, corr_doc))

    unsmile_pairs = []
    for row in ds_unsmile:
        if len(unsmile_pairs) >= max_examples:
            break
        text = row.get("문장", "")
        if text and len(text.strip()) >= 2:
            corr_text = corrupt_text(text, rng=rng, rate=0.8)
            unsmile_pairs.append((text, corr_text))

    print(f"Loaded {len(nsmc_pairs)} NSMC pairs and {len(unsmile_pairs)} UnSmile pairs (80% noise).")
    return {"nsmc": nsmc_pairs, "unsmile": unsmile_pairs}


def _worker_noise_chunk(args):
    key, kind, chunk, block_size = args
    tok_map = build_tokenizer_map()
    _, enc = tok_map[key]
    out = []
    for clean_str, corr_str in chunk:
        if kind == "single":
            c_toks = enc(clean_str)[:block_size]
            n_toks = enc(corr_str)[:block_size]
            out.append((c_toks, n_toks))
        else:
            c_lanes = [l[:block_size] for l in enc.encode(clean_str)]
            n_lanes = [l[:block_size] for l in enc.encode(corr_str)]
            out.append((c_lanes, n_lanes))
    return out


def pre_tokenize_noise(datasets: Dict[str, List[Tuple[str, str]]], tok_map: Dict[str, Any], block_size: int = 256) -> Dict[str, Dict[str, List[Any]]]:
    total_pairs = sum(len(v) for v in datasets.values())
    import multiprocessing as mp
    cache = {}
    base_keys = [
        "single_char",
        "single_bpe",
        "single_byte",
        "mc_three_hot",
        "mc_25_full",
        "mc_25_coarse",
        "mc_25_charhead_full",
        "mc_25_charhead_coarse",
    ]
    n_workers = min(12, mp.cpu_count() or 4)

    for key in base_keys:
        t0 = time.time()
        kind, _ = tok_map[key]
        ds_dict = {}
        for ds_name in ["nsmc", "unsmile"]:
            pairs = datasets[ds_name]
            if len(pairs) > 1500 and kind != "single":
                chunk_size = max(1, (len(pairs) + n_workers - 1) // n_workers)
                chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
                with mp.get_context("spawn").Pool(n_workers) as pool:
                    res_chunks = pool.map(_worker_noise_chunk, [(key, kind, c, block_size) for c in chunks])
                pre_list = [item for sub in res_chunks for item in sub]
            else:
                _, enc = tok_map[key]
                pre_list = []
                for clean_str, corr_str in pairs:
                    if kind == "single":
                        c_toks = enc(clean_str)[:block_size]
                        n_toks = enc(corr_str)[:block_size]
                        pre_list.append((c_toks, n_toks))
                    else:
                        c_lanes = [l[:block_size] for l in enc.encode(clean_str)]
                        n_lanes = [l[:block_size] for l in enc.encode(corr_str)]
                        pre_list.append((c_lanes, n_lanes))
            ds_dict[ds_name] = pre_list
        cache[key] = ds_dict
        print(f"  Pre-tokenized {key} ({total_pairs} pairs = {total_pairs*2} sequences) in {time.time() - t0:.2f}s")

    for src_key, dst_key in [("mc_25_charhead_full", "mc_24_full"), ("mc_25_charhead_coarse", "mc_24_coarse")]:
        derived = {}
        for ds_name in ["nsmc", "unsmile"]:
            derived_list = []
            for c_lanes, n_lanes in cache[src_key][ds_name]:
                derived_list.append((c_lanes[:24], n_lanes[:24]))
            derived[ds_name] = derived_list
        cache[dst_key] = derived
        print(f"  Derived {dst_key} from {src_key}")

    return cache


_NOISE_SINGLE_BATCH_CACHE = {}
_NOISE_MC_BATCH_CACHE = {}


@torch.no_grad()
def evaluate_single_noise_batched(model, ds_dict: Dict[str, List[Any]], batch_size: int = 64) -> Dict[str, Any]:
    vocab_size = getattr(model.config, "vocab_size", None)
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    res = {}

    for ds_name in ["nsmc", "unsmile"]:
        pairs = ds_dict[ds_name]
        import numpy as np
        cache_key = (id(pairs), batch_size)
        if cache_key not in _NOISE_SINGLE_BATCH_CACHE:
            flat_seqs = []
            for idx, (c_toks, n_toks) in enumerate(pairs):
                if len(c_toks) >= 2 and len(n_toks) >= 2:
                    flat_seqs.append((idx, "clean", c_toks))
                    flat_seqs.append((idx, "corr", n_toks))
            batches = []
            for b_start in range(0, len(flat_seqs), batch_size):
                batch = flat_seqs[b_start : b_start + batch_size]
                max_full_len = max(len(x[2]) for x in batch)
                b_full_np = np.zeros((len(batch), max_full_len), dtype=np.int64)
                for row_i, x in enumerate(batch):
                    arr = np.array(x[2], dtype=np.int64)
                    b_full_np[row_i, :arr.shape[0]] = arr
                b_full_t = torch.from_numpy(b_full_np)
                b_in_cpu = b_full_t[:, :-1]
                b_tg_cpu = b_full_t[:, 1:]
                meta = [(x[0], x[1], len(x[2]) - 1) for x in batch]
                batches.append((b_in_cpu, b_tg_cpu, meta))
            _NOISE_SINGLE_BATCH_CACHE[cache_key] = batches

        batches = _NOISE_SINGLE_BATCH_CACHE[cache_key]
        scores_map = {}
        for b_in_cpu, b_tg_cpu, meta in batches:
            if vocab_size is not None:
                b_in = b_in_cpu.clamp(0, vocab_size - 1).to("cuda", non_blocking=True)
                b_tg = b_tg_cpu.clamp(0, vocab_size - 1).to("cuda", non_blocking=True)
            else:
                b_in = b_in_cpu.to("cuda", non_blocking=True)
                b_tg = b_tg_cpu.to("cuda", non_blocking=True)

            with ctx_autocast:
                logits, _ = model(b_in, b_tg)
            logprobs = F.log_softmax(logits, dim=-1)
            lp_full = logprobs.gather(-1, b_tg.unsqueeze(-1)).squeeze(-1).cpu()
            for row_i, (idx, kind, s_len) in enumerate(meta):
                scores_map[(idx, kind)] = lp_full[row_i, :s_len].mean().item()

        clean_lps = [scores_map[(idx, "clean")] for idx in range(len(pairs)) if (idx, "clean") in scores_map]
        corr_lps = [scores_map[(idx, "corr")] for idx in range(len(pairs)) if (idx, "corr") in scores_map]
        avg_clean = sum(clean_lps) / max(1, len(clean_lps))
        avg_corr = sum(corr_lps) / max(1, len(corr_lps))
        degr = avg_clean - avg_corr
        res[ds_name] = {
            "clean_logprob": round(avg_clean, 4),
            "corrupted_logprob": round(avg_corr, 4),
            "degradation": round(degr, 2),
        }

    return {"primary": res}


@torch.no_grad()
def evaluate_mc_noise_batched(model, ds_dict: Dict[str, List[Any]], tok_key: str, batch_size: int = 64) -> Dict[str, Any]:
    import numpy as np
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    has_charhead = "charhead" in tok_key
    n_lanes = len(ds_dict["nsmc"][0][0])

    scoring_modes = ["primary"]
    if has_charhead:
        scoring_modes.extend(["charhead_24", "jamo_lvt"])

    out = {sm: {} for sm in scoring_modes}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for ds_name in ["nsmc", "unsmile"]:
        pairs = ds_dict[ds_name]
        cache_key = (id(pairs), batch_size)
        if cache_key not in _NOISE_MC_BATCH_CACHE:
            flat_seqs = []
            for idx, (c_lanes, n_lanes_seq) in enumerate(pairs):
                if len(c_lanes[0]) >= 2 and len(n_lanes_seq[0]) >= 2:
                    flat_seqs.append((idx, "clean", c_lanes))
                    flat_seqs.append((idx, "corr", n_lanes_seq))
            batches = []
            for b_start in range(0, len(flat_seqs), batch_size):
                batch = flat_seqs[b_start : b_start + batch_size]
                max_full_len = max(len(x[2][0]) for x in batch)
                b_full_np = np.zeros((n_lanes, len(batch), max_full_len), dtype=np.int64)
                for row_i, x in enumerate(batch):
                    arr = np.array(x[2], dtype=np.int64)
                    b_full_np[:, row_i, :arr.shape[1]] = arr
                b_full_t = torch.from_numpy(b_full_np)
                b_inputs_cpu = [b_full_t[l_i, :, :-1] for l_i in range(n_lanes)]
                b_targets_cpu = [b_full_t[l_i, :, 1:] for l_i in range(n_lanes)]
                meta = [(x[0], x[1], len(x[2][0]) - 1) for x in batch]
                batches.append((b_inputs_cpu, b_targets_cpu, meta))
            _NOISE_MC_BATCH_CACHE[cache_key] = batches

        batches = _NOISE_MC_BATCH_CACHE[cache_key]
        scores_map = {sm: {} for sm in scoring_modes}
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

            lp_lanes_all = [F.log_softmax(logits_list[i], dim=-1) for i in range(n_lanes)]
            lp_stack = torch.stack(
                [lp_lanes_all[i].gather(-1, b_targets[i].unsqueeze(-1)).squeeze(-1) for i in range(n_lanes)],
                dim=0,
            )  # shape (n_lanes, B, T)

            if not has_charhead:
                seq_lp_prim_batch = lp_stack[:n_lanes].mean(dim=0).cpu()
                for row_i, (idx, kind, s_len) in enumerate(meta):
                    scores_map["primary"][(idx, kind)] = seq_lp_prim_batch[row_i, :s_len].mean().item()
            else:
                seq_lp_25_batch = lp_stack[:25].mean(dim=0).cpu()
                sum_24_greedy_batch = lp_stack[:24].sum(dim=0)
                seq_lp_24_greedy_batch = (sum_24_greedy_batch / 24.0).cpu()
                log_Z_valid_batch = (
                    torch.logsumexp(lp_lanes_all[1][:, :, 1:20], dim=-1)
                    + torch.logsumexp(lp_lanes_all[2][:, :, 1:22], dim=-1)
                    + torch.logsumexp(lp_lanes_all[3][:, :, 1:29], dim=-1)
                )
                is_hangul_batch = (b_targets[0] == 2)
                sum_24_joint_batch = torch.where(is_hangul_batch, sum_24_greedy_batch - log_Z_valid_batch, sum_24_greedy_batch)
                seq_lp_24_joint_batch = (sum_24_joint_batch / 24.0).cpu()

                for row_i, (idx, kind, s_len) in enumerate(meta):
                    scores_map["charhead_24"][(idx, kind)] = seq_lp_25_batch[row_i, :s_len].mean().item()
                    scores_map["primary"][(idx, kind)] = seq_lp_24_greedy_batch[row_i, :s_len].mean().item()
                    scores_map["jamo_lvt"][(idx, kind)] = seq_lp_24_joint_batch[row_i, :s_len].mean().item()

        for sm in scoring_modes:
            clean_lps = [scores_map[sm][(idx, "clean")] for idx in range(len(pairs)) if (idx, "clean") in scores_map[sm]]
            corr_lps = [scores_map[sm][(idx, "corr")] for idx in range(len(pairs)) if (idx, "corr") in scores_map[sm]]
            avg_clean = sum(clean_lps) / max(1, len(clean_lps))
            avg_corr = sum(corr_lps) / max(1, len(corr_lps))
            degr = avg_clean - avg_corr
            out[sm][ds_name] = {
                "clean_logprob": round(avg_clean, 4),
                "corrupted_logprob": round(avg_corr, 4),
                "degradation": round(degr, 2),
            }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="results_noise_resilience_1000_all_models.json")
    args = parser.parse_args()

    datasets = load_noisy_datasets_1000(max_examples=args.max_examples, seed=1337)
    tok_map = build_tokenizer_map()
    pre_cache = pre_tokenize_noise(datasets, tok_map, block_size=256)

    out_path = os.path.join(REPO_ROOT, args.output_json)
    all_results = {"adamw": {}, "muon": {}, "num_examples": args.max_examples}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("num_examples") == args.max_examples:
                all_results["adamw"] = saved.get("adamw", {})
                all_results["muon"] = saved.get("muon", {})
                print(f"Resuming: found {len(all_results['adamw'])} AdamW and {len(all_results['muon'])} Muon models in {args.output_json}")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== EVALUATING {suite_name.upper()} SUITE ON NSMC & UNSMILE 80% NOISE (N={args.max_examples}) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            if m_id in all_results[suite_name]:
                print(f"  [SKIP] {suite_name.upper()} {m_id} already evaluated.")
                continue
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"  [MISSING] {ckpt_path}")
                continue
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

            ds_dict = pre_cache[tok_key]
            if tok_key.startswith("single_"):
                res = evaluate_single_noise_batched(model, ds_dict, batch_size=args.batch_size)
            else:
                res = evaluate_mc_noise_batched(model, ds_dict, tok_key, batch_size=args.batch_size)

            all_results[suite_name][m_id] = res
            dt = time.time() - t0
            p = res["primary"]
            msg = (
                f"  [{suite_name.upper()}] {m_id:38s} | NSMC: Clean={p['nsmc']['clean_logprob']:7.4f} Deg={p['nsmc']['degradation']:4.2f} "
                f"| UnSmile: Clean={p['unsmile']['clean_logprob']:7.4f} Deg={p['unsmile']['degradation']:4.2f}"
            )
            if "charhead_24" in res:
                c = res["charhead_24"]
                j = res["jamo_lvt"]
                msg += f" || WITH_24 NSMC Deg={c['nsmc']['degradation']:4.2f} | JointLVT NSMC Deg={j['nsmc']['degradation']:4.2f}"
            msg += f" [{dt:.1f}s]"
            print(msg)

            del model
            torch.cuda.empty_cache()

            out_path = os.path.join(REPO_ROOT, args.output_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted 1,000-example Extreme Noise Resilience evaluation across all 34 models! Saved to {args.output_json}")


if __name__ == "__main__":
    main()
