#!/usr/bin/env python3
"""
Evaluate Vocabulary Tail Perplexity across ALL 11,172 modern Hangul syllables (Deciles 1-10)
and OOV Stress Test (20 stress-test prompts) across all 34 models (17 AdamW + 17 Muon best_ckpt.pt).
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
from hangul_factorizer import HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer
from benchmarks.run_vocab_tail_perplexity import build_syllable_deciles
from benchmarks.run_oov_evaluations import get_prompts

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


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[GPT, Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    filtered_args = {k: v for k, v in model_args.items() if k in allowed}
    filtered_args["dropout"] = 0.0
    conf = GPTConfig(**filtered_args)
    model = GPT(conf)
    state_dict = checkpoint["model"]
    prefix = "_orig_mod."
    clean_sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval().to(device)
    return model, checkpoint


def pretokenize_tail_and_oov(all_hangul: List[str], oov_prompts: List[str]) -> Dict[str, Any]:
    print("Pre-tokenizing all 11,172 Hangul syllables and 20 OOV stress prompts...")
    cache = {}

    # 1. Single-channel tokenizers
    for key, folder in [
        ("single_char", "out_baseline_korean_pos"),
        ("single_bpe", "out_baseline_char_bpe"),
        ("single_byte", "out_baseline_pure_byte"),
    ]:
        with open(os.path.join(REPO_ROOT, folder, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        enc, _ = get_tokenizer_functions(meta)
        prefix_len = len(enc("단어: "))
        syl_items = []
        for c in all_hangul:
            ids = enc(f"단어: {c}")
            syl_items.append((ids, prefix_len))
        oov_items = [enc(p) for p in oov_prompts]
        cache[key] = {"syllables": syl_items, "oov": oov_items}

    # 2. Multi-channel 25-lane & 3-hot tokenizers
    tok_25_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
    tok_25_coarse = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")
    tok_three_hot = HangulPosFactorizedTokenizer(use_pos=False, pos_mode="full")

    for key, tok, sample_ckpt in [
        ("mc_three_hot", tok_three_hot, "out_three_hot_ind_char_byte/best_ckpt.pt"),
        ("mc_25_full", tok_25_full, "out_mc_full_pos_weighted_byte/best_ckpt.pt"),
        ("mc_25_coarse", tok_25_coarse, "out_mc_coarse_pos_weighted_byte/best_ckpt.pt"),
    ]:
        ck = torch.load(os.path.join(REPO_ROOT, sample_ckpt), map_location="cpu", weights_only=False)
        lanes = ck.get("config", {}).get("multicontext_datasets", [])
        stois = []
        for lp in lanes:
            with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
                stois.append(pickle.load(f)["stoi"])
        n_lanes = len(stois)

        def seq_to_lanes(seq):
            tl = [[] for _ in range(n_lanes)]
            for step in seq:
                ch = step["char"]
                idx = step["indices"]
                if n_lanes == 4:
                    byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                    tl[0].append(byte_id)
                    for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                        t_char = tok.token_for(factor_idx, idx[factor_idx])
                        tl[lane_i].append(stois[lane_i].get(t_char, 0))
                else:
                    for i in range(min(24, n_lanes - 1)):
                        t_char = tok.token_for(i, idx[i])
                        tl[i].append(stois[i].get(t_char, 0))
                    if n_lanes > 24:
                        byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                        tl[24].append(byte_id)
            return tl

        prefix_lanes = seq_to_lanes(tok.encode_text("단어: "))
        syl_items = []
        for c in all_hangul:
            # Fast single-syllable append without calling MeCab 11,172 times
            c_vals = tok.encode_char(c, pos_tag="NNG")
            c_step = {"char": c, "indices": c_vals}
            c_lane = seq_to_lanes([c_step])
            full_l = [prefix_lanes[i] + c_lane[i] for i in range(n_lanes)]
            syl_items.append(full_l)

        oov_items = [seq_to_lanes(tok.encode_text(p)) for p in oov_prompts]
        cache[key] = {"syllables": syl_items, "oov": oov_items}

    # 3. Hybrid 24/25-lane tokenizers
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")

    for key_25, key_24, tok in [
        ("mc_25_charhead_full", "mc_24_full", tok_24_full),
        ("mc_25_charhead_coarse", "mc_24_coarse", tok_24_coarse),
    ]:
        prefix_steps = tok.encode_text("단어: ")
        prefix_25 = [[] for _ in range(25)]
        for step in prefix_steps:
            for i in range(24):
                prefix_25[i].append(step[i])
            if step[0] == tok.hangul_id:
                cid = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
            else:
                cid = 0
            prefix_25[24].append(cid)

        syl_25 = []
        syl_24 = []
        for s_idx, c in enumerate(all_hangul):
            step = tok.encode_text(c, char_pos_tags={0: "NNG"})[0]
            c_step = list(step) + [s_idx + 1]
            full_25 = [prefix_25[i] + [c_step[i]] for i in range(25)]
            syl_25.append(full_25)
            syl_24.append(full_25[:24])

        def encode_oov_hybrid(p_str):
            steps = tok.encode_text(p_str)
            tl = [[] for _ in range(25)]
            for step in steps:
                for i in range(24):
                    tl[i].append(step[i])
                if step[0] == tok.hangul_id:
                    cid = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                else:
                    cid = 0
                tl[24].append(cid)
            return tl

        oov_25 = [encode_oov_hybrid(p) for p in oov_prompts]
        oov_24 = [x[:24] for x in oov_25]
        cache[key_25] = {"syllables": syl_25, "oov": oov_25}
        cache[key_24] = {"syllables": syl_24, "oov": oov_24}

    return cache


@torch.no_grad()
def evaluate_model_tail_and_oov(
    model: GPT,
    tok_key: str,
    data: Dict[str, Any],
    all_hangul: List[str],
    char_to_decile: Dict[str, int],
    device: str = "cuda",
    batch_size: int = 512,
) -> Dict[str, Any]:
    is_single = tok_key.startswith("single_")
    has_charhead = "charhead" in tok_key
    is_hybrid_24 = "mc_24" in tok_key
    vocab_size = getattr(model.config, "vocab_size", None)
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    syl_items = data["syllables"]
    oov_items = data["oov"]

    if is_single:
        decile_nlls = {d: [] for d in range(1, 11)}
        for b_start in range(0, len(syl_items), batch_size):
            batch = syl_items[b_start : b_start + batch_size]
            chars_b = all_hangul[b_start : b_start + batch_size]
            max_len = max(len(x[0]) - 1 for x in batch)
            inp_np = np.zeros((len(batch), max_len), dtype=np.int64)
            tgt_np = np.zeros((len(batch), max_len), dtype=np.int64)
            for i, (ids, _) in enumerate(batch):
                L = len(ids) - 1
                arr = np.array(ids, dtype=np.int64)
                inp_np[i, :L] = arr[:-1]
                tgt_np[i, :L] = arr[1:]
            inp = torch.from_numpy(inp_np).to(device)
            tgt = torch.from_numpy(tgt_np).to(device)
            if vocab_size is not None:
                inp.clamp_(0, vocab_size - 1)
                tgt.clamp_(0, vocab_size - 1)
            with ctx_autocast:
                logits, _ = model(inp, targets=tgt)
            lp = F.log_softmax(logits, dim=-1)
            lp_tgt = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
            for i, (ids, p_len) in enumerate(batch):
                # Target syllable tokens start at p_len - 1 in target sequence
                L = len(ids) - 1
                start_idx = max(0, p_len - 1)
                syl_nll = -lp_tgt[i, start_idx:L].mean().item()
                d = char_to_decile[chars_b[i]]
                decile_nlls[d].append(syl_nll)

        deciles_out = {f"decile_{d}": round(float(np.exp(np.mean(decile_nlls[d]))), 2) for d in range(1, 11)}

        oov_nlls = []
        for ids in oov_items:
            if len(ids) > 1:
                arr = torch.tensor(ids, dtype=torch.long, device=device)
                if vocab_size is not None:
                    arr.clamp_(0, vocab_size - 1)
                with ctx_autocast:
                    _, loss = model(arr[:-1].unsqueeze(0), targets=arr[1:].unsqueeze(0))
                if not torch.isnan(loss):
                    oov_nlls.append(loss.item())
        oov_ppl = round(float(np.exp(np.mean(oov_nlls))), 2)
        return {"primary": {"deciles": deciles_out, "oov_ppl": oov_ppl, "oov_loss": round(float(np.mean(oov_nlls)), 4)}}

    else:
        n_lanes = len(syl_items[0])
        modes = ["primary"]
        if is_hybrid_24:
            modes.append("jamo_lvt")
        if has_charhead:
            modes.extend(["charhead_24", "jamo_lvt"])

        decile_nlls = {m: {d: [] for d in range(1, 11)} for m in modes}

        # All syllable items have identical length (len("단어: ") + 1 = 5 tokens -> L=4)
        L = len(syl_items[0][0]) - 1
        for b_start in range(0, len(syl_items), batch_size):
            batch = syl_items[b_start : b_start + batch_size]
            chars_b = all_hangul[b_start : b_start + batch_size]
            B = len(batch)
            inp_np = np.zeros((n_lanes, B, L), dtype=np.int64)
            tgt_np = np.zeros((n_lanes, B, L), dtype=np.int64)
            for i, lanes in enumerate(batch):
                arr = np.array(lanes, dtype=np.int64)
                inp_np[:, i, :] = arr[:, :-1]
                tgt_np[:, i, :] = arr[:, 1:]
            inp_t = torch.from_numpy(inp_np).to(device)
            tgt_t = torch.from_numpy(tgt_np).to(device)
            for l_i in range(n_lanes):
                if vocab_sizes is not None and l_i < len(vocab_sizes):
                    inp_t[l_i].clamp_(0, vocab_sizes[l_i] - 1)
                    tgt_t[l_i].clamp_(0, vocab_sizes[l_i] - 1)

            td = {f"lane_{i}": inp_t[i] for i in range(n_lanes)}
            tg = {f"lane_{i}": tgt_t[i] for i in range(n_lanes)}
            with ctx_autocast:
                logits_list, _ = model(token_dict=td, target_dict=tg)

            # Final position (-1) predicts the target syllable c
            lp_last = [F.log_softmax(logits_list[i][:, -1, :], dim=-1) for i in range(n_lanes)]
            tgt_last = tgt_t[:, :, -1]  # (n_lanes, B)
            gathered = [lp_last[i].gather(-1, tgt_last[i].unsqueeze(-1)).squeeze(-1) for i in range(n_lanes)]

            if not is_hybrid_24 and not has_charhead:
                # mc_three_hot or mc_25_full / mc_25_coarse
                check_lanes = (1, 2, 3) if n_lanes == 4 else (1, 2, 3, 24)
                s_lp = sum(gathered[l_i] for l_i in check_lanes).cpu()
                for i in range(B):
                    d = char_to_decile[chars_b[i]]
                    decile_nlls["primary"][d].append(-s_lp[i].item())
            else:
                # Hybrid 24-lane nochar or 25-lane charhead
                # 1) primary (WITHOUT lm_head_24, Greedy): sum of logprobs over (script, L, V, T) / 4 (per-factor mean NLL)
                sum_lvt = (gathered[0] + gathered[1] + gathered[2] + gathered[3]) / 4.0
                s_prim = sum_lvt.cpu()

                # 2) jamo_lvt (WITHOUT lm_head_24, Joint Beam Search): renormalized over valid Hangul LxVxT
                log_Z_valid = (
                    torch.logsumexp(lp_last[1][:, 1:20], dim=-1)
                    + torch.logsumexp(lp_last[2][:, 1:22], dim=-1)
                    + torch.logsumexp(lp_last[3][:, 1:29], dim=-1)
                )
                sum_joint = (gathered[0] + gathered[1] + gathered[2] + gathered[3] - log_Z_valid) / 4.0
                s_jamo = sum_joint.cpu()

                for i in range(B):
                    d = char_to_decile[chars_b[i]]
                    decile_nlls["primary"][d].append(-s_prim[i].item())
                    decile_nlls["jamo_lvt"][d].append(-s_jamo[i].item())

                if has_charhead:
                    # 3) charhead_24 (WITH lm_head_24): mean of (script + char_24) / 2
                    sum_ch24 = (gathered[0] + gathered[24]) / 2.0
                    s_ch24 = sum_ch24.cpu()
                    for i in range(B):
                        d = char_to_decile[chars_b[i]]
                        decile_nlls["charhead_24"][d].append(-s_ch24[i].item())

        # OOV Stress Test
        oov_nlls = {m: [] for m in modes}
        for lanes in oov_items:
            seq_len = len(lanes[0])
            if seq_len > 1:
                arr = torch.tensor(lanes, dtype=torch.long, device=device)
                for l_i in range(n_lanes):
                    if vocab_sizes is not None and l_i < len(vocab_sizes):
                        arr[l_i].clamp_(0, vocab_sizes[l_i] - 1)
                td = {f"lane_{i}": arr[i, :-1].unsqueeze(0) for i in range(n_lanes)}
                tg = {f"lane_{i}": arr[i, 1:].unsqueeze(0) for i in range(n_lanes)}
                with ctx_autocast:
                    logits_list, losses = model(token_dict=td, target_dict=tg)
                if not has_charhead:
                    vl = [l.item() for l in losses if not torch.isnan(l)]
                    if vl:
                        loss_val = sum(vl) / len(vl)
                        oov_nlls["primary"].append(loss_val)
                        if "jamo_lvt" in oov_nlls:
                            oov_nlls["jamo_lvt"].append(loss_val)
                else:
                    vl_25 = [l.item() for l in losses[:25] if not torch.isnan(l)]
                    vl_24 = [l.item() for l in losses[:24] if not torch.isnan(l)]
                    if vl_25:
                        oov_nlls["charhead_24"].append(sum(vl_25) / len(vl_25))
                    if vl_24:
                        oov_nlls["primary"].append(sum(vl_24) / len(vl_24))
                        oov_nlls["jamo_lvt"].append(sum(vl_24) / len(vl_24))

        out = {}
        for m in modes:
            dec_m = {f"decile_{d}": round(float(np.exp(np.mean(decile_nlls[m][d]))), 2) for d in range(1, 11)}
            mean_oov = float(np.mean(oov_nlls[m]))
            out[m] = {
                "deciles": dec_m,
                "oov_loss": round(mean_oov, 4),
                "oov_ppl": round(float(np.exp(mean_oov)), 2),
            }
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=str, default="results_oov_tail_all_syllables_models.json")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus_txt = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "input.txt")
    char_to_decile, decile_meta = build_syllable_deciles(corpus_txt)
    all_hangul = [chr(code) for code in range(0xAC00, 0xD7A3 + 1)]
    oov_prompts = [p if isinstance(p, str) else p.get("prompt", "") for p in get_prompts()]

    print(f"Total Hangul syllables: {len(all_hangul)} across 10 deciles ({len(all_hangul)//10} per decile).")
    print(f"Total OOV Stress Test prompts: {len(oov_prompts)}.")

    pre_cache = pretokenize_tail_and_oov(all_hangul, oov_prompts)

    out_path = os.path.join(REPO_ROOT, args.output_json)
    all_results = {"adamw": {}, "muon": {}, "num_syllables": len(all_hangul), "num_oov_prompts": len(oov_prompts)}

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== EVALUATING {suite_name.upper()} MODELS ON ALL 11,172 SYLLABLES & OOV ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            t0 = time.time()
            model, _ = load_model_from_ckpt(ckpt_path, device=device)
            res = evaluate_model_tail_and_oov(
                model, tok_key, pre_cache[tok_key], all_hangul, char_to_decile, device=device, batch_size=args.batch_size
            )
            all_results[suite_name][m_id] = res
            dt = time.time() - t0
            p = res["primary"]
            d1 = p["deciles"]["decile_1"]
            d5 = p["deciles"]["decile_5"]
            d10 = p["deciles"]["decile_10"]
            msg = f"  [{suite_name.upper()}] {m_id:38s} | D1: {d1:6.2f} | D5: {d5:6.2f} | D10: {d10:6.2f} | OOV PPL: {p['oov_ppl']:6.2f} ({dt:.2f}s)"
            if "charhead_24" in res:
                ch = res["charhead_24"]
                j = res["jamo_lvt"]
                msg += f" || CH24 D10: {ch['deciles']['decile_10']:6.2f} OOV: {ch['oov_ppl']:5.2f} | LVT D10: {j['deciles']['decile_10']:6.2f}"
            print(msg)
            del model
            torch.cuda.empty_cache()

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved complete 11,172-syllable Vocabulary Tail & OOV results to {args.output_json}")


if __name__ == "__main__":
    main()
