#!/usr/bin/env python3
"""Evaluate 1,000 KorMedMCQA examples across all 34 models (17 AdamW + 17 Muon best_ckpt)."""
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


def load_kormedmcqa_1000(max_examples: int = 1000, seed: int = 1337) -> List[Dict[str, Any]]:
    """Load 1,000 balanced questions (250 per subject) or all 3,009 test questions if max_examples >= 3000."""
    use_all = max_examples >= 3000
    per_subj = 999999 if use_all else (max_examples // 4)
    questions = []
    for subj in ["doctor", "dentist", "nurse", "pharm"]:
        ds = load_dataset("sean0042/KorMedMCQA", subj, split="test")
        if not use_all:
            ds = ds.shuffle(seed=seed)
        cnt = 0
        for row in ds:
            if cnt >= per_subj:
                break
            q = row.get("question", "")
            opts = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", ""), row.get("E", "")]
            opts = [o for o in opts if o]
            raw_ans = row.get("answer")
            if q and opts and raw_ans is not None:
                ans_idx = int(raw_ans) - 1
                if 0 <= ans_idx < len(opts):
                    questions.append({
                        "subject": subj,
                        "question": q,
                        "options": opts,
                        "ans_idx": ans_idx,
                    })
                    cnt += 1
    print(f"Loaded {len(questions)} valid KorMedMCQA questions across 4 subjects.")
    return questions


def pre_tokenize_kormed(questions: List[Dict[str, Any]], tok_map: Dict[str, Any], block_size: int = 256) -> Dict[str, List[Any]]:
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
        pre_list = []
        for q_item in questions:
            q = q_item["question"]
            opts = q_item["options"]
            ans_idx = q_item["ans_idx"]
            subj = q_item["subject"]
            opt_seqs = []
            for opt in opts:
                prompt = f"질문: {q} 정답: {opt}"
                if kind == "single":
                    toks = enc(prompt)[:block_size]
                    opt_seqs.append(toks)
                else:
                    lanes = enc.encode(prompt)
                    lanes = [l[:block_size] for l in lanes]
                    opt_seqs.append(lanes)
            pre_list.append((opt_seqs, ans_idx, subj))
        cache[key] = pre_list
        print(f"  Pre-tokenized {key} ({len(questions)} questions) in {time.time() - t0:.2f}s")

    for src_key, dst_key in [("mc_25_charhead_full", "mc_24_full"), ("mc_25_charhead_coarse", "mc_24_coarse")]:
        derived = []
        for opt_seqs, ans_idx, subj in cache[src_key]:
            opt_24 = [lanes[:24] for lanes in opt_seqs]
            derived.append((opt_24, ans_idx, subj))
        cache[dst_key] = derived
        print(f"  Derived {dst_key} from {src_key}")

    return cache


def evaluate_single_kormed_batched(model, pre_tok: List[Any], batch_size: int = 64) -> Dict[str, Any]:
    vocab_size = getattr(model.config, "vocab_size", None)
    items = []
    for q_idx, (opt_seqs, ans_idx, subj) in enumerate(pre_tok):
        for opt_idx, toks in enumerate(opt_seqs):
            if len(toks) >= 2:
                items.append((q_idx, opt_idx, toks[:-1], toks[1:]))

    scores_map = {}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for b_start in range(0, len(items), batch_size):
            batch = items[b_start : b_start + batch_size]
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
                q_idx, opt_idx, inp, tgt = item
                s_len = len(inp)
                target_slice = b_tg[row_i, :s_len]
                lp = logprobs[row_i, :s_len, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
                scores_map[(q_idx, opt_idx)] = lp.mean().item()

    correct = 0
    subj_correct = {"doctor": 0, "dentist": 0, "nurse": 0, "pharm": 0}
    subj_total = {"doctor": 0, "dentist": 0, "nurse": 0, "pharm": 0}

    for q_idx, (opt_seqs, ans_idx, subj) in enumerate(pre_tok):
        opt_scores = [scores_map.get((q_idx, opt_idx), -1e9) for opt_idx in range(len(opt_seqs))]
        pred = int(torch.tensor(opt_scores).argmax().item())
        subj_total[subj] += 1
        if pred == ans_idx:
            correct += 1
            subj_correct[subj] += 1

    total = len(pre_tok)
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "by_subject": {s: subj_correct[s] / max(subj_total[s], 1) for s in subj_total},
    }


def evaluate_mc_kormed_batched(model, pre_tok: List[Any], tok_key: str, batch_size: int = 64) -> Dict[str, Any]:
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    is_hybrid = "nochar" in tok_key or "charhead" in tok_key or tok_key in ("mc_24_full", "mc_24_coarse")
    has_charhead = "charhead" in tok_key
    n_lanes = len(pre_tok[0][0][0])

    items = []
    for q_idx, (opt_seqs, ans_idx, subj) in enumerate(pre_tok):
        for opt_idx, lanes in enumerate(opt_seqs):
            if len(lanes[0]) >= 2:
                items.append((q_idx, opt_idx, [l[:-1] for l in lanes], [l[1:] for l in lanes]))

    scoring_modes = ["primary"]
    if is_hybrid:
        scoring_modes.append("jamo_lvt")
    if has_charhead:
        scoring_modes.append("charhead_24")

    scores_map = {sm: {} for sm in scoring_modes}
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        for b_start in range(0, len(items), batch_size):
            batch = items[b_start : b_start + batch_size]
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

            head_idx = 0 if is_hybrid else -1
            lp_primary_all = F.log_softmax(logits_list[head_idx], dim=-1)

            if is_hybrid:
                lp1_all = F.log_softmax(logits_list[1], dim=-1)
                lp2_all = F.log_softmax(logits_list[2], dim=-1)
                lp3_all = F.log_softmax(logits_list[3], dim=-1)
                log_Z1 = torch.logsumexp(logits_list[1][:, :, 1:20], dim=-1)
                log_Z2 = torch.logsumexp(logits_list[2][:, :, 1:22], dim=-1)
                log_Z3 = torch.logsumexp(logits_list[3][:, :, 0:28], dim=-1)
                lp1_norm_all = logits_list[1] - log_Z1.unsqueeze(-1)
                lp2_norm_all = logits_list[2] - log_Z2.unsqueeze(-1)
                lp3_norm_all = logits_list[3] - log_Z3.unsqueeze(-1)
            if has_charhead:
                lp24_all = F.log_softmax(logits_list[24], dim=-1)

            for row_i, item in enumerate(batch):
                q_idx, opt_idx, inp_l, tgt_l = item
                s_len = len(inp_l[0])
                tgt_primary = b_targets[head_idx][row_i, :s_len]
                lp_prim = lp_primary_all[row_i, :s_len, :].gather(-1, tgt_primary.unsqueeze(-1)).squeeze(-1)

                if is_hybrid:
                    is_hangul = (tgt_primary == 2)
                    t1 = b_targets[1][row_i, :s_len]
                    t2 = b_targets[2][row_i, :s_len]
                    t3 = b_targets[3][row_i, :s_len]
                    lp1 = lp1_all[row_i, :s_len, :].gather(-1, t1.unsqueeze(-1)).squeeze(-1)
                    lp2 = lp2_all[row_i, :s_len, :].gather(-1, t2.unsqueeze(-1)).squeeze(-1)
                    lp3 = lp3_all[row_i, :s_len, :].gather(-1, t3.unsqueeze(-1)).squeeze(-1)
                    lp_greedy = torch.where(is_hangul, lp_prim + lp1 + lp2 + lp3, lp_prim)
                    scores_map["primary"][(q_idx, opt_idx)] = lp_greedy.mean().item()

                    lp1_n = lp1_norm_all[row_i, :s_len, :].gather(-1, t1.unsqueeze(-1)).squeeze(-1)
                    lp2_n = lp2_norm_all[row_i, :s_len, :].gather(-1, t2.unsqueeze(-1)).squeeze(-1)
                    lp3_n = lp3_norm_all[row_i, :s_len, :].gather(-1, t3.unsqueeze(-1)).squeeze(-1)
                    lp_lvt = torch.where(is_hangul, lp_prim + lp1_n + lp2_n + lp3_n, lp_prim)
                    scores_map["jamo_lvt"][(q_idx, opt_idx)] = lp_lvt.mean().item()
                else:
                    scores_map["primary"][(q_idx, opt_idx)] = lp_prim.mean().item()

                if has_charhead:
                    t24 = b_targets[24][row_i, :s_len]
                    lp24 = lp24_all[row_i, :s_len, :].gather(-1, t24.unsqueeze(-1)).squeeze(-1)
                    lp_ch24 = torch.where(is_hangul, lp_prim + lp24, lp_prim)
                    scores_map["charhead_24"][(q_idx, opt_idx)] = lp_ch24.mean().item()

    total = len(pre_tok)
    out = {}
    for sm in scoring_modes:
        correct = 0
        subj_correct = {"doctor": 0, "dentist": 0, "nurse": 0, "pharm": 0}
        subj_total = {"doctor": 0, "dentist": 0, "nurse": 0, "pharm": 0}
        for q_idx, (opt_seqs, ans_idx, subj) in enumerate(pre_tok):
            opt_scores = [scores_map[sm].get((q_idx, opt_idx), -1e9) for opt_idx in range(len(opt_seqs))]
            pred = int(torch.tensor(opt_scores).argmax().item())
            subj_total[subj] += 1
            if pred == ans_idx:
                correct += 1
                subj_correct[subj] += 1
        out[sm] = {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "by_subject": {s: subj_correct[s] / max(subj_total[s], 1) for s in subj_total},
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="results_kormedmcqa_1000_all_models.json")
    args = parser.parse_args()

    questions = load_kormedmcqa_1000(max_examples=args.max_examples, seed=1337)
    tok_map = build_tokenizer_map()
    pre_tok_cache = pre_tokenize_kormed(questions, tok_map, block_size=256)

    all_results = {"adamw": {}, "muon": {}, "num_examples": len(questions)}

    for opt_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== EVALUATING {opt_name.upper()} MODELS ON KORMEDMCQA (N={len(questions)}) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
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
                res = evaluate_single_kormed_batched(model, pre_tok, batch_size=args.batch_size)
                entry = {"primary": res}
            else:
                entry = evaluate_mc_kormed_batched(model, pre_tok, tok_key, batch_size=args.batch_size)

            elapsed = time.time() - t0
            prim = entry["primary"]
            msg = f"  [{opt_name.upper()}] {m_id:38s} | primary: {prim['accuracy']*100:5.1f}% ({prim['correct']}/{prim['total']})"
            if "jamo_lvt" in entry:
                lvt = entry["jamo_lvt"]
                msg += f" || LVT: {lvt['accuracy']*100:5.1f}%"
            if "charhead_24" in entry:
                ch = entry["charhead_24"]
                msg += f" || CH24: {ch['accuracy']*100:5.1f}%"
            msg += f" ({elapsed:.1f}s)"
            print(msg)

            all_results[opt_name][m_id] = entry
            del model
            torch.cuda.empty_cache()

            out_path = os.path.join(REPO_ROOT, args.output_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved complete 1,000-example KorMedMCQA evaluation results to {args.output_json}")


if __name__ == "__main__":
    main()
