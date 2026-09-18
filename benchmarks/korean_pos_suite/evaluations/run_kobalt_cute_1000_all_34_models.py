#!/usr/bin/env python3
"""
Fast batched evaluation of KoBALT & CUTE-Hangul (N=1,000 across 5 categories: 200 items each)
across all 34 models (17 AdamW + 17 Muon best_ckpt).
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
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.kobalt_cute_1000_dataset import build_full_kobalt_cute_1000
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

CATEGORIES = [
    "choseong_quiz",
    "allomorphic_particles",
    "phonological_assimilation",
    "vowel_harmony_ideophones",
    "cute_jamo_manipulation",
]


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


def pre_tokenize_kobalt(dataset: Dict[str, List[Dict[str, Any]]], tok_map: Dict[str, Any], block_size: int = 256) -> Dict[str, Dict[str, List[Any]]]:
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
    for key in base_keys:
        t0 = time.time()
        kind, enc = tok_map[key]
        cat_dict = {}
        for cat in CATEGORIES:
            items = dataset[cat]
            pre_list = []
            for item in items:
                ctx = item["ctx"]
                cands = item["candidates"]
                ans_idx = item["answer"]

                if kind == "single":
                    ctx_toks = enc(ctx)
                    opt_seqs = []
                    for cand in cands:
                        cand_toks = enc(cand)
                        full = (ctx_toks + cand_toks)[-block_size:]
                        start_idx = max(0, len(full) - len(cand_toks) - 1)
                        end_idx = len(full) - 1
                        opt_seqs.append({
                            "full": full,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "char_len": max(1, len(cand.strip())),
                        })
                    pre_list.append((opt_seqs, ans_idx))
                else:
                    ctx_lanes = enc.encode(ctx)
                    opt_seqs = []
                    for cand in cands:
                        cand_lanes = enc.encode(cand)
                        cand_len = len(cand_lanes[0])
                        full_lanes = [(ctx_lanes[i] + cand_lanes[i])[-block_size:] for i in range(len(ctx_lanes))]
                        full_len = len(full_lanes[0])
                        start_idx = max(0, full_len - cand_len - 1)
                        end_idx = full_len - 1
                        opt_seqs.append({
                            "lanes": full_lanes,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "char_len": max(1, len(cand.strip())),
                        })
                    pre_list.append((opt_seqs, ans_idx))
            cat_dict[cat] = pre_list
        cache[key] = cat_dict
        print(f"  Pre-tokenized {key} (1,000 items) in {time.time() - t0:.2f}s")

    for src_key, dst_key in [("mc_25_charhead_full", "mc_24_full"), ("mc_25_charhead_coarse", "mc_24_coarse")]:
        derived = {}
        for cat in CATEGORIES:
            cat_list = []
            for opt_seqs, ans_idx in cache[src_key][cat]:
                opt_24 = []
                for c in opt_seqs:
                    opt_24.append({
                        "lanes": c["lanes"][:24],
                        "start_idx": c["start_idx"],
                        "end_idx": c["end_idx"],
                        "char_len": c["char_len"],
                    })
                cat_list.append((opt_24, ans_idx))
            derived[cat] = cat_list
        cache[dst_key] = derived
        print(f"  Derived {dst_key} from {src_key}")

    return cache


def evaluate_single_kobalt_batched(model, cat_dict: Dict[str, List[Any]], batch_size: int = 64) -> Dict[str, Any]:
    vocab_size = getattr(model.config, "vocab_size", None)
    res = {}
    total_correct = 0
    total_count = 0
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        for cat in CATEGORIES:
            pre_tok = cat_dict[cat]
            flat_items = []
            for q_idx, (opt_seqs, ans_idx) in enumerate(pre_tok):
                for opt_idx, c in enumerate(opt_seqs):
                    full = c["full"]
                    if len(full) >= 2:
                        flat_items.append((q_idx, opt_idx, full[:-1], full[1:], c["start_idx"], c["end_idx"], c["char_len"]))

            scores_map = {}
            for b_start in range(0, len(flat_items), batch_size):
                batch = flat_items[b_start : b_start + batch_size]
                max_len = max(len(x[2]) for x in batch)
                b_in = torch.zeros((len(batch), max_len), dtype=torch.long, device="cuda")
                b_tg = torch.zeros((len(batch), max_len), dtype=torch.long, device="cuda")
                for row_i, item in enumerate(batch):
                    inp, tgt = item[2], item[3]
                    s_len = len(inp)
                    t_in = torch.tensor(inp, dtype=torch.long, device="cuda")
                    t_tg = torch.tensor(tgt, dtype=torch.long, device="cuda")
                    if vocab_size is not None:
                        t_in = torch.clamp(t_in, 0, vocab_size - 1)
                        t_tg = torch.clamp(t_tg, 0, vocab_size - 1)
                    b_in[row_i, :s_len] = t_in
                    b_tg[row_i, :s_len] = t_tg

                with ctx_autocast:
                    logits, _ = model(b_in, b_tg)
                logprobs = F.log_softmax(logits, dim=-1)
                for row_i, item in enumerate(batch):
                    q_idx, opt_idx, inp, tgt, start_idx, end_idx, char_len = item
                    target_slice = b_tg[row_i, start_idx:end_idx]
                    lp = logprobs[row_i, start_idx:end_idx, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
                    scores_map[(q_idx, opt_idx)] = lp.sum().item() / char_len

            correct = 0
            for q_idx, (opt_seqs, ans_idx) in enumerate(pre_tok):
                opt_scores = [scores_map.get((q_idx, opt_idx), -1e9) for opt_idx in range(len(opt_seqs))]
                pred = int(torch.tensor(opt_scores).argmax().item())
                if pred == ans_idx:
                    correct += 1
            acc = 100.0 * correct / len(pre_tok)
            res[cat] = {"accuracy": round(acc, 2), "correct": correct, "total": len(pre_tok)}
            total_correct += correct
            total_count += len(pre_tok)

    res["overall_accuracy"] = round(100.0 * total_correct / total_count, 2)
    return res


def evaluate_mc_kobalt_batched(model, cat_dict: Dict[str, List[Any]], tok_key: str, batch_size: int = 64) -> Dict[str, Any]:
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    is_hybrid = "nochar" in tok_key or "charhead" in tok_key
    has_charhead = "charhead" in tok_key
    n_lanes = len(cat_dict[CATEGORIES[0]][0][0][0]["lanes"])

    scoring_modes = ["primary"]
    if is_hybrid:
        scoring_modes.append("jamo_lvt")
    if has_charhead:
        scoring_modes.append("charhead_24")

    results_by_mode = {m: {"total_correct": 0, "total_count": 0, "cats": {}} for m in scoring_modes}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        for cat in CATEGORIES:
            pre_tok = cat_dict[cat]
            flat_items = []
            for q_idx, (opt_seqs, ans_idx) in enumerate(pre_tok):
                for opt_idx, c in enumerate(opt_seqs):
                    lanes = c["lanes"]
                    if len(lanes[0]) >= 2:
                        flat_items.append((
                            q_idx,
                            opt_idx,
                            [l[:-1] for l in lanes],
                            [l[1:] for l in lanes],
                            c["start_idx"],
                            c["end_idx"],
                            c["char_len"],
                        ))

            scores_map = {sm: {} for sm in scoring_modes}
            for b_start in range(0, len(flat_items), batch_size):
                batch = flat_items[b_start : b_start + batch_size]
                max_seq_len = max(len(x[2][0]) for x in batch)
                b_inputs = [torch.zeros((len(batch), max_seq_len), dtype=torch.long, device="cuda") for _ in range(n_lanes)]
                b_targets = [torch.zeros((len(batch), max_seq_len), dtype=torch.long, device="cuda") for _ in range(n_lanes)]
                for row_i, item in enumerate(batch):
                    inp_l, tgt_l = item[2], item[3]
                    s_len = len(inp_l[0])
                    for lane_i in range(n_lanes):
                        t_in = torch.tensor(inp_l[lane_i], dtype=torch.long, device="cuda")
                        t_tg = torch.tensor(tgt_l[lane_i], dtype=torch.long, device="cuda")
                        if vocab_sizes is not None and lane_i < len(vocab_sizes):
                            t_in = torch.clamp(t_in, 0, vocab_sizes[lane_i] - 1)
                            t_tg = torch.clamp(t_tg, 0, vocab_sizes[lane_i] - 1)
                        b_inputs[lane_i][row_i, :s_len] = t_in
                        b_targets[lane_i][row_i, :s_len] = t_tg

                token_dict = {f"lane_{i}": b_inputs[i] for i in range(n_lanes)}
                target_dict = {f"lane_{i}": b_targets[i] for i in range(n_lanes)}
                with ctx_autocast:
                    logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)

                if is_hybrid:
                    # Hybrid models (nochar / charhead): lane 0 is primary script/BPE head
                    lp0_all = F.log_softmax(logits_list[0], dim=-1)
                    lp1_all = F.log_softmax(logits_list[1], dim=-1)
                    lp2_all = F.log_softmax(logits_list[2], dim=-1)
                    lp3_all = F.log_softmax(logits_list[3], dim=-1)
                    if has_charhead:
                        lp24_all = F.log_softmax(logits_list[24], dim=-1)

                    for row_i, item in enumerate(batch):
                        q_idx, opt_idx, inp_l, tgt_l, start_idx, end_idx, char_len = item
                        tgt0 = b_targets[0][row_i, start_idx:end_idx]
                        lp0 = lp0_all[row_i, start_idx:end_idx, :].gather(-1, tgt0.unsqueeze(-1)).squeeze(-1)
                        scores_map["primary"][(q_idx, opt_idx)] = lp0.sum().item() / char_len

                        is_hangul = (tgt0 == 2).float()
                        t1 = b_targets[1][row_i, start_idx:end_idx]
                        t2 = b_targets[2][row_i, start_idx:end_idx]
                        t3 = b_targets[3][row_i, start_idx:end_idx]
                        lp1 = lp1_all[row_i, start_idx:end_idx, :].gather(-1, t1.unsqueeze(-1)).squeeze(-1)
                        lp2 = lp2_all[row_i, start_idx:end_idx, :].gather(-1, t2.unsqueeze(-1)).squeeze(-1)
                        lp3 = lp3_all[row_i, start_idx:end_idx, :].gather(-1, t3.unsqueeze(-1)).squeeze(-1)
                        lp_jamo_sum = (lp1 + lp2 + lp3) * is_hangul
                        scores_map["jamo_lvt"][(q_idx, opt_idx)] = (lp0 + lp_jamo_sum).sum().item() / char_len

                        if has_charhead:
                            t24 = b_targets[24][row_i, start_idx:end_idx]
                            lp24 = lp24_all[row_i, start_idx:end_idx, :].gather(-1, t24.unsqueeze(-1)).squeeze(-1)
                            lp_char_sum = lp24 * is_hangul
                            scores_map["charhead_24"][(q_idx, opt_idx)] = (lp0 + lp_jamo_sum + lp_char_sum).sum().item() / char_len
                else:
                    # 25-lane POS-MC or 3-Hot: combine jamo lanes (1, 2, 3) + surface char_byte lane (0 or 24)
                    active_lanes = [0, 1, 2, 3] if n_lanes == 4 else [1, 2, 3, 24]
                    lp_active = {l_idx: F.log_softmax(logits_list[l_idx], dim=-1) for l_idx in active_lanes if l_idx < len(logits_list)}
                    for row_i, item in enumerate(batch):
                        q_idx, opt_idx, inp_l, tgt_l, start_idx, end_idx, char_len = item
                        total_lp = 0.0
                        for l_idx, lp_tensor in lp_active.items():
                            tgt_slice = b_targets[l_idx][row_i, start_idx:end_idx]
                            gathered = lp_tensor[row_i, start_idx:end_idx, :].gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
                            total_lp += gathered.sum().item()
                        scores_map["primary"][(q_idx, opt_idx)] = total_lp / char_len

            for sm in scoring_modes:
                correct = 0
                for q_idx, (opt_seqs, ans_idx) in enumerate(pre_tok):
                    opt_scores = [scores_map[sm].get((q_idx, opt_idx), -1e9) for opt_idx in range(len(opt_seqs))]
                    pred = int(torch.tensor(opt_scores).argmax().item())
                    if pred == ans_idx:
                        correct += 1
                acc = 100.0 * correct / len(pre_tok)
                results_by_mode[sm]["cats"][cat] = {"accuracy": round(acc, 2), "correct": correct, "total": len(pre_tok)}
                results_by_mode[sm]["total_correct"] += correct
                results_by_mode[sm]["total_count"] += len(pre_tok)

    out = {}
    for sm in scoring_modes:
        cat_res = results_by_mode[sm]["cats"]
        cat_res["overall_accuracy"] = round(100.0 * results_by_mode[sm]["total_correct"] / results_by_mode[sm]["total_count"], 2)
        out[sm] = cat_res
    return out


def main():
    print("Building 1,000-example KoBALT & CUTE-Hangul dataset...")
    dataset = build_full_kobalt_cute_1000()
    tok_map = build_tokenizer_map()
    cache = pre_tokenize_kobalt(dataset, tok_map, block_size=256)

    all_results = {"adamw": {}, "muon": {}}

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== Evaluating {suite_name.upper()} Suite (17 models) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
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
            cat_dict = cache[tok_key]
            if tok_key.startswith("single_"):
                res = evaluate_single_kobalt_batched(model, cat_dict, batch_size=64)
                all_results[suite_name][m_id] = {"primary": res}
                dt = time.time() - t0
                print(
                    f"  [{suite_name.upper()}] {m_id:38s} | Overall={res['overall_accuracy']:5.2f}% "
                    f"(Chos={res['choseong_quiz']['accuracy']:5.1f}% Part={res['allomorphic_particles']['accuracy']:5.1f}% "
                    f"Assim={res['phonological_assimilation']['accuracy']:5.1f}% Harm={res['vowel_harmony_ideophones']['accuracy']:5.1f}% "
                    f"Jamo={res['cute_jamo_manipulation']['accuracy']:5.1f}%) [{dt:.1f}s]"
                )
            else:
                res = evaluate_mc_kobalt_batched(model, cat_dict, tok_key, batch_size=64)
                all_results[suite_name][m_id] = res
                dt = time.time() - t0
                p = res["primary"]
                msg = (
                    f"  [{suite_name.upper()}] {m_id:38s} | Primary={p['overall_accuracy']:5.2f}% "
                    f"(Chos={p['choseong_quiz']['accuracy']:5.1f}% Part={p['allomorphic_particles']['accuracy']:5.1f}% "
                    f"Assim={p['phonological_assimilation']['accuracy']:5.1f}% Harm={p['vowel_harmony_ideophones']['accuracy']:5.1f}% "
                    f"Jamo={p['cute_jamo_manipulation']['accuracy']:5.1f}%)"
                )
                if "jamo_lvt" in res:
                    msg += f" | JamoLVT={res['jamo_lvt']['overall_accuracy']:5.2f}%"
                if "charhead_24" in res:
                    msg += f" | CharHead24={res['charhead_24']['overall_accuracy']:5.2f}%"
                msg += f" [{dt:.1f}s]"
                print(msg)

            del model
            torch.cuda.empty_cache()

    out_path = os.path.join(REPO_ROOT, "results_kobalt_cute_1000_all_models.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all 1,000-example KoBALT & CUTE-Hangul results to {out_path}")


if __name__ == "__main__":
    main()
