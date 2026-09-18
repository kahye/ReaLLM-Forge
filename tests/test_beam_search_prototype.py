#!/usr/bin/env python3
"""Test prototype for Joint Factor-Product Beam Search on Hangul Factorizers."""
import os, os as _os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# Test on 5 samples
TEST_JSONL = os.path.join(REPO_ROOT, "data", "korean_seq2seq_bench", "test.jsonl")
with open(TEST_JSONL, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()][:5]

spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
tok = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
newline_ids = set(tok.sp.encode_as_ids("\n"))
if tok.sp.eos_id() != -1:
    newline_ids.add(tok.sp.eos_id())

ckpt_path = os.path.join(REPO_ROOT, "out_mc_nochar_coarse_pos_new_weighted_muon", "15_epochs.pt")
model, ckpt_dict = load_model(ckpt_path)
lane_names = ckpt_dict.get("config", {}).get("multicontext_datasets", [])
n_lanes = len(lane_names)
print(f"Loaded model from {ckpt_path} with {n_lanes} lanes.")

def generate_greedy_standard(prompt, max_new_tokens=40):
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

def generate_joint_factor_beam_search(prompt, beam_size=4, max_new_tokens=40):
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

    init_lanes = [torch.tensor(lane, device=DEVICE).unsqueeze(0) for lane in token_lists]
    curr_len = init_lanes[0].size(1)

    # A beam is a tuple: (cumulative_score, input_lanes, is_done)
    beams = [(0.0, init_lanes, False)]

    for step_num in range(max_new_tokens):
        candidates = []
        all_done = True

        for cum_score, lanes, is_done in beams:
            if is_done:
                candidates.append((cum_score, lanes, True))
                continue

            all_done = False
            cond_lanes = [lane[:, -256:] for lane in lanes]
            token_dict = {f"lane_{i}": lane for i, lane in enumerate(cond_lanes)}
            with torch.no_grad():
                logits_list, _ = model(token_dict=token_dict)

            # Lane 0 log probs
            log_p0 = F.log_softmax(logits_list[0][0, -1, :], dim=-1)
            hangul_logp = log_p0[tok.hangul_id].item()

            # 1. Joint Hangul Syllable Option over valid L x V x T
            log_p_L = F.log_softmax(logits_list[1][0, -1, 1:20], dim=-1)  # 19
            log_p_V = F.log_softmax(logits_list[2][0, -1, 1:22], dim=-1)  # 21
            log_p_T = F.log_softmax(logits_list[3][0, -1, 0:28], dim=-1)  # 28

            joint_syl_logp = (log_p_L.view(19, 1, 1) + log_p_V.view(1, 21, 1) + log_p_T.view(1, 1, 28)).view(11172)
            
            if n_lanes >= 25:
                log_p_char = F.log_softmax(logits_list[24][0, -1, 1:11173], dim=-1)
                joint_syl_logp = joint_syl_logp + log_p_char

            # Top K syllables
            top_syl_scores, top_syl_indices = torch.topk(joint_syl_logp, k=min(beam_size, 4))

            # Other lanes default (argmax)
            other_defaults = []
            for lane_i in range(4, min(24, n_lanes)):
                other_defaults.append(torch.argmax(logits_list[lane_i][0, -1, :]).item())

            for score_val, syl_idx in zip(top_syl_scores, top_syl_indices):
                s_idx = syl_idx.item()
                l_val = (s_idx // 588) + 1
                rem = s_idx % 588
                v_val = (rem // 28) + 1
                t_val = rem % 28

                # Synthesize factored syllable
                syl_char = chr(0xAC00 + s_idx)
                factored_step = tok.encode_text(syl_char)[0]

                next_tokens = [tok.hangul_id, l_val, v_val, t_val]
                for lane_idx in range(4, min(24, n_lanes)):
                    if lane_idx < len(factored_step):
                        next_tokens.append(factored_step[lane_idx])
                    else:
                        next_tokens.append(other_defaults[lane_idx - 4])
                if n_lanes >= 25:
                    next_tokens.append(s_idx + 1)

                new_lanes = [torch.cat([lanes[i], torch.tensor([[next_tokens[i]]], device=DEVICE)], dim=1) for i in range(n_lanes)]
                cand_score = cum_score + hangul_logp + score_val.item()
                candidates.append((cand_score, new_lanes, False))

            # 2. Non-Hangul Options (top non-Hangul tokens from Lane 0)
            log_p0_clone = log_p0.clone()
            log_p0_clone[tok.hangul_id] = -float("inf")
            top_non_scores, top_non_ids = torch.topk(log_p0_clone, k=min(beam_size, 3))

            for score_val, non_id in zip(top_non_scores, top_non_ids):
                tok_id = non_id.item()
                next_tokens = [tok_id]
                for lane_i in range(1, n_lanes):
                    next_tokens.append(0)  # PAD for non-Hangul

                new_lanes = [torch.cat([lanes[i], torch.tensor([[next_tokens[i]]], device=DEVICE)], dim=1) for i in range(n_lanes)]
                cand_score = cum_score + score_val.item()
                done_flag = (tok_id in newline_ids) or ("\n" in tok.sp.id_to_piece(tok_id))
                candidates.append((cand_score, new_lanes, done_flag))

        if all_done:
            break

        # Sort all candidates and prune to beam_size
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    # Best beam
    best_lanes = beams[0][1]
    gen_steps = []
    for step_idx in range(curr_len, best_lanes[0].size(1)):
        row = [best_lanes[i][0, step_idx].item() for i in range(min(24, n_lanes))]
        gen_steps.append(row)
    try:
        pred_str = tok.decode_sequence(gen_steps)
    except Exception:
        pred_str = ""
    return pred_str.strip()

print("\nRunning comparison on 5 sample sentences...")
for idx, s in enumerate(samples):
    en = s["en"]
    ko = s["ko"]
    prompt = f"{en}\t"

    t0 = time.time()
    greedy_res = generate_greedy_standard(prompt)
    t_greedy = (time.time() - t0) * 1000

    t0 = time.time()
    beam_res = generate_joint_factor_beam_search(prompt, beam_size=4)
    t_beam = (time.time() - t0) * 1000

    print(f"\n--- Sample {idx+1} ---")
    print(f"EN: {en[:70]}...")
    print(f"KO Ref: {ko[:70]}...")
    print(f"Greedy ({t_greedy:.1f}ms): {greedy_res}")
    print(f"Joint Beam ({t_beam:.1f}ms): {beam_res}")
