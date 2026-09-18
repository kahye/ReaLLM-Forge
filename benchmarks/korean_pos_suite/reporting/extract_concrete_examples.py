#!/usr/bin/env python3
"""Extract sentence-level concrete examples (high score, low score, high improvement) for Muon and AdamW."""
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
import os
import sys
import time
import torch
import torch.nn.functional as F
from dataclasses import fields
import sacrebleu
from jamo import h2j, j2hcj
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
KOREAN_DIR = os.path.join(REPO_ROOT, "data", "template", "utils", "korean")
sys.path.insert(0, KOREAN_DIR)

from model import GPT, GPTConfig
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def canonicalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    decomposed = j2hcj(h2j(text))
    stripped = re.sub(r"[^\w\s]", "", decomposed)
    return re.sub(r"\s+", " ", stripped).strip()

def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    filtered_args = {k: v for k, v in model_args.items() if k in allowed}
    filtered_args["dropout"] = 0.0
    conf = GPTConfig(**filtered_args)
    model = GPT(conf)
    state_dict = checkpoint["model"]
    clean_sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval().to(DEVICE)
    return model, checkpoint

# Load 40 test samples
TEST_JSONL = os.path.join(REPO_ROOT, "data", "korean_seq2seq_bench", "test.jsonl")
with open(TEST_JSONL, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()][:40]

spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
tok_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
tok_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")

MODELS = [
    # Muon models
    {"id": "muon_nochar_full_weighted_best", "name": "No-Char Full POS Weighted (Muon best_ckpt)", "path": "out_mc_nochar_full_pos_new_weighted_muon/best_ckpt.pt", "pos": "full", "use_lmhead": False},
    {"id": "muon_nochar_full_unweighted_best", "name": "No-Char Full POS Unweighted (Muon best_ckpt)", "path": "out_mc_nochar_full_pos_unweighted_muon/best_ckpt.pt", "pos": "full", "use_lmhead": False},
    {"id": "muon_nochar_coarse_weighted_15ep", "name": "No-Char Coarse POS Weighted (Muon 15_epochs)", "path": "out_mc_nochar_coarse_pos_new_weighted_muon/15_epochs.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "muon_nochar_coarse_unweighted_20ep", "name": "No-Char Coarse POS Unweighted (Muon 20_epochs)", "path": "out_mc_nochar_coarse_pos_unweighted_muon/20_epochs.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "muon_charhead_full_h11_20ep_no_lmhead", "name": "Char-Head Full POS H11 (Muon 20_epochs, no lm_head_24)", "path": "out_mc_nochar_full_pos_charhead_h11_muon/20_epochs.pt", "pos": "full", "use_lmhead": False},
    {"id": "muon_charhead_full_h11_20ep_with_lmhead", "name": "Char-Head Full POS H11 (Muon 20_epochs, with lm_head_24)", "path": "out_mc_nochar_full_pos_charhead_h11_muon/20_epochs.pt", "pos": "full", "use_lmhead": True},
    {"id": "muon_charhead_coarse_h11_20ep_no_lmhead", "name": "Char-Head Coarse POS H11 (Muon 20_epochs, no lm_head_24)", "path": "out_mc_nochar_coarse_pos_charhead_h11_muon/20_epochs.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "muon_charhead_coarse_h11_20ep_with_lmhead", "name": "Char-Head Coarse POS H11 (Muon 20_epochs, with lm_head_24)", "path": "out_mc_nochar_coarse_pos_charhead_h11_muon/20_epochs.pt", "pos": "coarse", "use_lmhead": True},

    # AdamW models
    {"id": "adamw_nochar_full_weighted_15ep", "name": "No-Char Full POS Weighted (AdamW 15_epochs)", "path": "out_mc_nochar_full_pos_new_weighted/15_epochs.pt", "pos": "full", "use_lmhead": False},
    {"id": "adamw_nochar_full_unweighted_15ep", "name": "No-Char Full POS Unweighted (AdamW 15_epochs)", "path": "out_mc_nochar_full_pos_unweighted/15_epochs.pt", "pos": "full", "use_lmhead": False},
    {"id": "adamw_nochar_coarse_weighted_best", "name": "No-Char Coarse POS Weighted (AdamW best_ckpt)", "path": "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "adamw_nochar_coarse_unweighted_15ep", "name": "No-Char Coarse POS Unweighted (AdamW 15_epochs)", "path": "out_mc_nochar_coarse_pos_unweighted/15_epochs.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "adamw_charhead_full_h11_20ep_no_lmhead", "name": "Char-Head Full POS H11 (AdamW 20_epochs, no lm_head_24)", "path": "out_mc_nochar_full_pos_charhead_h11/20_epochs.pt", "pos": "full", "use_lmhead": False},
    {"id": "adamw_charhead_full_h11_20ep_with_lmhead", "name": "Char-Head Full POS H11 (AdamW 20_epochs, with lm_head_24)", "path": "out_mc_nochar_full_pos_charhead_h11/20_epochs.pt", "pos": "full", "use_lmhead": True},
    {"id": "adamw_charhead_coarse_h11_3ep_no_lmhead", "name": "Char-Head Coarse POS H11 (AdamW 3_epoch, no lm_head_24)", "path": "out_gemma3_mc_nochar_coarse_pos_charhead_h11/3472.pt", "pos": "coarse", "use_lmhead": False},
    {"id": "adamw_charhead_coarse_h11_3ep_with_lmhead", "name": "Char-Head Coarse POS H11 (AdamW 3_epoch, with lm_head_24)", "path": "out_gemma3_mc_nochar_coarse_pos_charhead_h11/3472.pt", "pos": "coarse", "use_lmhead": True},
]

def generate_sentence(model, tok, n_lanes, prompt, mode="greedy", use_lmhead=False, max_new_tokens=40):
    newline_ids = set(tok.sp.encode_as_ids("\n"))
    if tok.sp.eos_id() != -1:
        newline_ids.add(tok.sp.eos_id())

    p_seq = tok.encode_text(prompt)
    token_lists = [[] for _ in range(n_lanes)]
    for step in p_seq:
        for i in range(min(24, len(step))):
            token_lists[i].append(step[i])
        if n_lanes >= 25:
            if step[0] == tok.hangul_id:
                char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
            else:
                char_id = 0
            token_lists[24].append(char_id)

    input_lanes = [torch.tensor(lane, device=DEVICE).unsqueeze(0) for lane in token_lists]
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

        if next_tokens[0] == tok.hangul_id:
            if mode == "joint_factor_product" or (n_lanes >= 25 and not use_lmhead):
                log_p_L = F.log_softmax(logits_list[1][0, -1, 1:20], dim=-1)
                log_p_V = F.log_softmax(logits_list[2][0, -1, 1:22], dim=-1)
                log_p_T = F.log_softmax(logits_list[3][0, -1, 0:28], dim=-1)

                joint_log_p = (log_p_L.view(19, 1, 1) + log_p_V.view(1, 21, 1) + log_p_T.view(1, 1, 28)).view(11172)
                best_syl_idx = torch.argmax(joint_log_p).item()

                syl_char = chr(0xAC00 + best_syl_idx)
                factored_step = tok.encode_text(syl_char)[0]
                for lane_idx in range(min(23, len(factored_step))):
                    next_tokens[lane_idx] = factored_step[lane_idx]
                if n_lanes >= 25:
                    next_tokens[24] = best_syl_idx + 1

            elif n_lanes >= 25 and use_lmhead:
                char_logits = logits_list[24][0, -1, 1:11173]
                best_syll_id = torch.argmax(char_logits).item() + 1
                s_idx = best_syll_id - 1
                syl_char = chr(0xAC00 + s_idx)
                factored_step = tok.encode_text(syl_char)[0]
                for lane_idx in range(min(23, len(factored_step))):
                    next_tokens[lane_idx] = factored_step[lane_idx]
                next_tokens[24] = best_syll_id

        for i in range(n_lanes):
            input_lanes[i] = torch.cat([input_lanes[i], torch.tensor([[next_tokens[i]]], device=DEVICE)], dim=1)

        if next_tokens[0] in newline_ids or "\n" in tok.sp.id_to_piece(next_tokens[0]):
            break

    gen_steps = []
    for step_idx in range(curr_len, input_lanes[0].size(1)):
        row = [input_lanes[i][0, step_idx].item() for i in range(min(24, n_lanes))]
        gen_steps.append(row)
    try:
        pred_str = tok.decode_sequence(gen_steps)
    except Exception:
        pred_str = ""
    return pred_str.strip()

out_records = {}

for m in MODELS:
    ckpt_path = os.path.join(REPO_ROOT, m["path"])
    if not os.path.exists(ckpt_path):
        print(f"Skipping {m['id']} (not found)")
        continue
    tok = tok_full if m["pos"] == "full" else tok_coarse
    model, ckpt_dict = load_model(ckpt_path)
    lane_names = ckpt_dict.get("config", {}).get("multicontext_datasets", [])
    n_lanes = len(lane_names)
    print(f"\nProcessing {m['id']}...")

    model_samples = []
    for s_idx, s in enumerate(samples):
        en = s["en"]
        ko = s["ko"]
        prompt = f"{en}\t"
        ref_canon = canonicalize(ko)

        # 1. Greedy
        hyp_greedy = generate_sentence(model, tok, n_lanes, prompt, mode="greedy", use_lmhead=m["use_lmhead"])
        hyp_greedy_canon = canonicalize(hyp_greedy)
        greedy_chrf = sacrebleu.sentence_chrf(hyp_greedy_canon, [ref_canon], char_order=18).score if hyp_greedy_canon else 0.0
        greedy_bleu = sacrebleu.sentence_bleu(hyp_greedy_canon, [ref_canon], smooth_method="exp").score if hyp_greedy_canon else 0.0

        # 2. Joint Factor Product
        hyp_joint = generate_sentence(model, tok, n_lanes, prompt, mode="joint_factor_product", use_lmhead=m["use_lmhead"])
        hyp_joint_canon = canonicalize(hyp_joint)
        joint_chrf = sacrebleu.sentence_chrf(hyp_joint_canon, [ref_canon], char_order=18).score if hyp_joint_canon else 0.0
        joint_bleu = sacrebleu.sentence_bleu(hyp_joint_canon, [ref_canon], smooth_method="exp").score if hyp_joint_canon else 0.0

        model_samples.append({
            "idx": s_idx + 1,
            "en": en,
            "ko": ko,
            "hyp_greedy": hyp_greedy,
            "greedy_chrf": round(greedy_chrf, 2),
            "greedy_bleu": round(greedy_bleu, 2),
            "hyp_joint": hyp_joint,
            "joint_chrf": round(joint_chrf, 2),
            "joint_bleu": round(joint_bleu, 2),
            "delta_chrf": round(joint_chrf - greedy_chrf, 2),
            "delta_bleu": round(joint_bleu - greedy_bleu, 2),
        })

    # Sort to pick top improved, top scoring, and low scoring
    sorted_by_delta = sorted(model_samples, key=lambda x: x["delta_chrf"], reverse=True)
    sorted_by_joint_chrf = sorted(model_samples, key=lambda x: x["joint_chrf"], reverse=True)
    sorted_by_low = sorted(model_samples, key=lambda x: x["joint_chrf"])

    out_records[m["id"]] = {
        "name": m["name"],
        "top_improved": sorted_by_delta[:3],
        "highest_scoring": sorted_by_joint_chrf[:3],
        "lowest_scoring": sorted_by_low[:3],
    }

    del model
    torch.cuda.empty_cache()

out_json_path = os.path.join(REPO_ROOT, "concrete_examples_muon_adamw.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(out_records, f, indent=2, ensure_ascii=False)
print(f"\nSaved concrete examples to {out_json_path}")
