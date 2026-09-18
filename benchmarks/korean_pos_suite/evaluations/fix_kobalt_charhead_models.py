#!/usr/bin/env python3
"""
Re-evaluate all multi-lane Hangul Factorizer models (POS-MC 25-lane, No-Char 24-lane, and Output-Only Char-Head 25-lane)
across both AdamW and Muon on KoBALT & CUTE-Hangul (N=1,000, 200 per task).

Fixes four evaluation bugs in the previous 1,000-example scripts:
1. Constituent Imbalance & Modulo Hash Duplication in WITHOUT lm_head_24, Joint Beam Search (jamo_lvt):
   - Previously summed all 22 lanes with equal weight 1.0, giving Jongseong 8x weight (including duplicate modulo hashes
     lanes 21 & 22) and Jungseong 9x weight against 1 primary Choseong lane, collapsing Choseong Quiz to 25.0% random chance.
   - Now normalizes each syllable constituent (Choseong L, Jungseong V, Jongseong T) to equal unit weight:
     L_lanes = [1, 12, 13, 14, 15] (weight 1/5 each)
     V_lanes = [2, 4, 5, 6, 7, 8, 16, 17, 18] (weight 1/9 each)
     T_lanes = [3, 9, 10, 11, 19, 20] (weight 1/6 each, excluding duplicate modulo hashes 21, 22)
2. Proper Non-Hangul Token Masking & Context-Candidate Boundary Alignment:
   - Encodes context and candidate separately (`ctx_lanes + cand_lanes`) so `start_idx` is identical across choices.
   - Masks Jamo/articulatory lanes on non-Hangul tokens (`is_h`) so non-Hangul spaces/punctuation are scored purely by
     `lane_0` (`script_bpe`) or `lane_24` (`char_byte`) instead of padding token 0.
3. Missing `lane_0` (`script_bpe`) for 24-lane pure No-Char models (`mc_24_full` / `mc_24_coarse`):
   - Properly includes `lane_0` (`script_bpe`) + constituent-balanced Jamo factor-product.
4. Exact Conditional Log-Likelihood (`sum`) for candidate scoring:
   - Eliminates `1 / char_len` length-division distortion on variable-length candidates in Choseong Quiz and Allomorphic Particles.
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
import os
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from benchmarks.kobalt_cute_1000_dataset import build_full_kobalt_cute_1000
from run_kobalt_cute_1000_all_34_models import CATEGORIES, _load_checkpoint, build_tokenizer_map

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

L_LANES = [1, 12, 13, 14, 15]
V_LANES = [2, 4, 5, 6, 7, 8, 16, 17, 18]
T_LANES = [3, 9, 10, 11, 19, 20]

CHARHEAD_MODELS = [
    ("nochar_full_pos_charhead_h11", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_full_pos_charhead_unweighted", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_coarse_pos_charhead_h11", "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt", "mc_25_charhead_coarse"),
    ("nochar_coarse_pos_charhead_unweighted", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "mc_25_charhead_coarse"),
]

NOCHAR_MODELS = [
    ("nochar_full_pos_new_weighted", "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_full_pos_unweighted", "out_mc_nochar_full_pos_unweighted/best_ckpt.pt", "mc_25_charhead_full"),
    ("nochar_coarse_pos_new_weighted", "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "mc_25_charhead_coarse"),
    ("nochar_coarse_pos_unweighted", "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt", "mc_25_charhead_coarse"),
]

POS_MC_MODELS = [
    ("mc_full_pos_weighted_byte", "out_mc_full_pos_weighted_byte/best_ckpt.pt", "mc_25_full"),
    ("mc_full_pos_unweighted_byte", "out_mc_full_pos_unweighted_byte/best_ckpt.pt", "mc_25_full"),
    ("mc_coarse_pos_weighted_byte", "out_mc_coarse_pos_weighted_byte/best_ckpt.pt", "mc_25_coarse"),
    ("mc_coarse_pos_unweighted_byte", "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt", "mc_25_coarse"),
]


def pre_encode_aligned(dataset: Dict[str, List[Dict[str, Any]]], enc) -> Dict[str, Any]:
    flat_items = []
    q_meta = []
    for cat in CATEGORIES:
        for item in dataset[cat]:
            ctx = item["ctx"]
            cands = item["candidates"]
            ans = item["answer"]
            ctx_lanes = enc.encode(ctx)
            q_idx = len(q_meta)
            q_meta.append((cat, ans, len(cands)))
            for opt_idx, cand in enumerate(cands):
                cand_lanes = enc.encode(cand)
                full_lanes = [ctx_lanes[i] + cand_lanes[i] for i in range(len(ctx_lanes))]
                start_idx = len(ctx_lanes[0]) - 1
                end_idx = len(full_lanes[0]) - 1
                flat_items.append((
                    q_idx,
                    opt_idx,
                    [l[:-1] for l in full_lanes],
                    [l[1:] for l in full_lanes],
                    start_idx,
                    end_idx,
                    max(1, len(cand.strip())),
                ))
    return {"flat_items": flat_items, "q_meta": q_meta}


@torch.no_grad()
def evaluate_model_family(
    model,
    pre_data: Dict[str, Any],
    factor_table: torch.Tensor,
    family: str,
    n_model_lanes: int = 25,
    batch_size: int = 64,
) -> Dict[str, Any]:
    flat_items = pre_data["flat_items"]
    q_meta = pre_data["q_meta"]
    if family == "charhead":
        modes = ["charhead_24", "primary", "jamo_lvt"]
    elif family == "nochar":
        modes = ["primary", "jamo_lvt"]
    else:
        modes = ["primary"]
    scores_map = {m: {} for m in modes}

    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with ctx_autocast:
        for b_start in range(0, len(flat_items), batch_size):
            batch = flat_items[b_start : b_start + batch_size]
            max_len = max(len(x[2][0]) for x in batch)
            b_in = [torch.zeros((len(batch), max_len), dtype=torch.long, device="cuda") for _ in range(n_model_lanes)]
            b_tg = [torch.zeros((len(batch), max_len), dtype=torch.long, device="cuda") for _ in range(n_model_lanes)]
            for r_i, item in enumerate(batch):
                s_len = len(item[2][0])
                for l_i in range(n_model_lanes):
                    b_in[l_i][r_i, :s_len] = torch.tensor(item[2][l_i], device="cuda")
                    b_tg[l_i][r_i, :s_len] = torch.tensor(item[3][l_i], device="cuda")
            logits, _ = model(
                token_dict={f"lane_{i}": b_in[i] for i in range(n_model_lanes)},
                target_dict={f"lane_{i}": b_tg[i] for i in range(n_model_lanes)},
            )
            lps = [F.log_softmax(logits[i], dim=-1) for i in range(n_model_lanes)]

            for r_i, item in enumerate(batch):
                q_idx, opt_idx, _, tgt_full_25, s_idx, e_idx, _ = item
                tgt_slice = [torch.tensor(tgt_full_25[i][s_idx:e_idx], device="cuda") for i in range(25)]

                if family in ("charhead", "nochar"):
                    is_h = (tgt_slice[0] == 2)
                    lp0 = lps[0][r_i, s_idx:e_idx].gather(-1, tgt_slice[0].unsqueeze(-1)).squeeze(-1)

                    # Constituent-balanced uncoupled Jamo score
                    sL_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in L_LANES) / len(L_LANES)
                    sV_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in V_LANES) / len(V_LANES)
                    sT_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in T_LANES) / len(T_LANES)
                    uncoupled_step = torch.where(is_h, lp0 + sL_unc + sV_unc + sT_unc, lp0)

                    # Constituent-balanced Joint Factor-Product over 11,172 valid Hangul syllables
                    sL_mat = sum(lps[k][r_i, s_idx:e_idx][:, factor_table[k]] for k in L_LANES) / len(L_LANES)
                    sV_mat = sum(lps[k][r_i, s_idx:e_idx][:, factor_table[k]] for k in V_LANES) / len(V_LANES)
                    sT_mat = sum(lps[k][r_i, s_idx:e_idx][:, factor_table[k]] for k in T_LANES) / len(T_LANES)
                    joint_lp = F.log_softmax(sL_mat + sV_mat + sT_mat, dim=-1)
                    tgt_syl = (tgt_slice[24] - 1).clamp(0, 11171)
                    lp_joint = joint_lp.gather(-1, tgt_syl.unsqueeze(-1)).squeeze(-1)
                    joint_step = torch.where(is_h, lp0 + lp_joint, lp0)

                    if family == "charhead":
                        lp24 = lps[24][r_i, s_idx:e_idx].gather(-1, tgt_slice[24].unsqueeze(-1)).squeeze(-1)
                        charhead_step = torch.where(is_h, lp0 + 0.5 * lp24 + 0.5 * lp_joint, lp0)
                        scores_map["charhead_24"][(q_idx, opt_idx)] = charhead_step.sum().item()
                        scores_map["primary"][(q_idx, opt_idx)] = uncoupled_step.sum().item()
                        scores_map["jamo_lvt"][(q_idx, opt_idx)] = joint_step.sum().item()
                    else:
                        # For pure 24-lane No-Char factorizer, store joint factor-product as primary & jamo_lvt
                        scores_map["primary"][(q_idx, opt_idx)] = joint_step.sum().item()
                        scores_map["jamo_lvt"][(q_idx, opt_idx)] = joint_step.sum().item()

                else:
                    # 25-lane Hangul Factorizer + Char Stream (pos_mc)
                    is_h = (tgt_slice[1] > 0)
                    lp24 = lps[24][r_i, s_idx:e_idx].gather(-1, tgt_slice[24].unsqueeze(-1)).squeeze(-1)
                    sL_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in L_LANES) / len(L_LANES)
                    sV_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in V_LANES) / len(V_LANES)
                    sT_unc = sum(lps[k][r_i, s_idx:e_idx].gather(-1, tgt_slice[k].unsqueeze(-1)).squeeze(-1) for k in T_LANES) / len(T_LANES)
                    step_tot = torch.where(is_h, lp24 + sL_unc + sV_unc + sT_unc, lp24)
                    scores_map["primary"][(q_idx, opt_idx)] = step_tot.sum().item()

    out = {}
    for m in modes:
        tot_c = 0
        cat_dict = {}
        cat_c = {c: 0 for c in CATEGORIES}
        cat_n = {c: 0 for c in CATEGORIES}
        for q_idx, (cat, ans, n_cands) in enumerate(q_meta):
            s_list = [scores_map[m][(q_idx, o_i)] for o_i in range(n_cands)]
            if int(torch.tensor(s_list).argmax().item()) == ans:
                tot_c += 1
                cat_c[cat] += 1
            cat_n[cat] += 1
        for c in CATEGORIES:
            acc = 100.0 * cat_c[c] / cat_n[c]
            cat_dict[c] = {"accuracy": round(acc, 2), "correct": cat_c[c], "total": cat_n[c]}
        cat_dict["overall_accuracy"] = round(100.0 * tot_c / len(q_meta), 2)
        out[m] = cat_dict

    return out


def main():
    json_path = os.path.join(REPO_ROOT, "results_kobalt_cute_1000_all_models.json")
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    dataset = build_full_kobalt_cute_1000()
    tok_map = build_tokenizer_map()

    cache = {}
    factor_tables = {}
    for tkey in ["mc_25_charhead_full", "mc_25_charhead_coarse", "mc_25_full", "mc_25_coarse"]:
        t0 = time.time()
        _, enc = tok_map[tkey]
        cache[tkey] = pre_encode_aligned(dataset, enc)
        hangul_chars = [chr(0xAC00 + s) for s in range(11172)]
        encoded_hangul = enc.encode("".join(hangul_chars))
        factor_tables[tkey] = torch.tensor([encoded_hangul[l_idx] for l_idx in range(24)], dtype=torch.long, device="cuda")
        print(f"Pre-encoded aligned {tkey} in {time.time() - t0:.2f}s")

    for suite_name in ["adamw", "muon"]:
        print(f"\n================ Updating {suite_name.upper()} Multi-Lane Factorizer Models ================")

        # 1. Char-Head 25-lane models
        for m_id, ckpt_rel, tkey in CHARHEAD_MODELS:
            if suite_name == "muon":
                ckpt_rel = ckpt_rel.replace("/best_ckpt.pt", "_muon/best_ckpt.pt")
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
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
            res = evaluate_model_family(model, cache[tkey], factor_tables[tkey], family="charhead", n_model_lanes=25)
            results[suite_name][m_id] = res
            print(
                f"  [{suite_name.upper()}] {m_id:38s} | WITH_24: {res['charhead_24']['overall_accuracy']:5.2f}% "
                f"| Greedy_24: {res['primary']['overall_accuracy']:5.2f}% "
                f"| Joint_11172: {res['jamo_lvt']['overall_accuracy']:5.2f}%"
            )

        # 2. No-Char 24-lane models
        for m_id, ckpt_rel, tkey in NOCHAR_MODELS:
            if suite_name == "muon":
                ckpt_rel = ckpt_rel.replace("/best_ckpt.pt", "_muon/best_ckpt.pt")
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
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
            res = evaluate_model_family(model, cache[tkey], factor_tables[tkey], family="nochar", n_model_lanes=24)
            results[suite_name][m_id] = res
            print(
                f"  [{suite_name.upper()}] {m_id:38s} | Joint_11172: {res['primary']['overall_accuracy']:5.2f}%"
            )

        # 3. POS-MC 25-lane models
        for m_id, ckpt_rel, tkey in POS_MC_MODELS:
            if suite_name == "muon":
                ckpt_rel = ckpt_rel.replace("/best_ckpt.pt", "_muon/best_ckpt.pt")
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
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
            res = evaluate_model_family(model, cache[tkey], factor_tables[tkey], family="pos_mc", n_model_lanes=25)
            results[suite_name][m_id] = res
            print(
                f"  [{suite_name.upper()}] {m_id:38s} | Primary: {res['primary']['overall_accuracy']:5.2f}%"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved updated results to", json_path)


if __name__ == "__main__":
    main()
