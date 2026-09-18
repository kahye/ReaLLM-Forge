#!/usr/bin/env python3
"""Evaluate remaining No-Char checkpoints under Standard Greedy vs Joint Factor Product."""
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

TEST_JSONL = os.path.join(REPO_ROOT, "data", "korean_seq2seq_bench", "test.jsonl")
with open(TEST_JSONL, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()][:100]

canon_refs = [canonicalize(s["ko"]) for s in samples]

spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
tok_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
tok_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")

MODELS = [
    # Muon No-Char Remaining
    ("muon_nochar_full_unweighted_best", "out_mc_nochar_full_pos_unweighted_muon/best_ckpt.pt", "full"),
    ("muon_nochar_coarse_unweighted_20ep", "out_mc_nochar_coarse_pos_unweighted_muon/20_epochs.pt", "coarse"),
    # AdamW No-Char Remaining
    ("adamw_nochar_full_weighted_15ep", "out_mc_nochar_full_pos_new_weighted/15_epochs.pt", "full"),
    ("adamw_nochar_full_unweighted_15ep", "out_mc_nochar_full_pos_unweighted/15_epochs.pt", "full"),
    ("adamw_nochar_coarse_unweighted_15ep", "out_mc_nochar_coarse_pos_unweighted/15_epochs.pt", "coarse"),
    # AdamW Char-Head Remaining (best_ckpt)
    ("adamw_charhead_coarse_h11_3ep", "out_gemma3_mc_nochar_coarse_pos_charhead_h11/3472.pt", "coarse"),
]

def generate(model, tok, n_lanes, samples, mode="greedy", max_new_tokens=40):
    newline_ids = set(tok.sp.encode_as_ids("\n"))
    if tok.sp.eos_id() != -1:
        newline_ids.add(tok.sp.eos_id())

    hyps = []
    t0 = time.time()

    for s in samples:
        prompt = f"{s['en']}\t"
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

            if next_tokens[0] == tok.hangul_id and mode == "joint_factor_product":
                log_p_L = F.log_softmax(logits_list[1][0, -1, 1:20], dim=-1)  # 19
                log_p_V = F.log_softmax(logits_list[2][0, -1, 1:22], dim=-1)  # 21
                log_p_T = F.log_softmax(logits_list[3][0, -1, 0:28], dim=-1)  # 28

                joint_log_p = (log_p_L.view(19, 1, 1) + log_p_V.view(1, 21, 1) + log_p_T.view(1, 1, 28)).view(11172)
                best_syl_idx = torch.argmax(joint_log_p).item()

                syl_char = chr(0xAC00 + best_syl_idx)
                factored_step = tok.encode_text(syl_char)[0]
                for lane_idx in range(min(23, len(factored_step))):
                    next_tokens[lane_idx] = factored_step[lane_idx]
                if n_lanes >= 25:
                    next_tokens[24] = best_syl_idx + 1

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
        hyps.append(pred_str.strip())

    latency_ms = (time.time() - t0) * 1000.0 / len(samples)
    canon_hyps = [canonicalize(h) for h in hyps]
    filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
    filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]

    bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
    chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

    return {"bleu": bleu, "chrf": chrf, "latency_ms": latency_ms, "sample_pred": hyps[0]}

results = {}
for model_id, rel_path, pos_type in MODELS:
    ckpt_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(ckpt_path):
        print(f"Skipping {ckpt_path} (not found)")
        continue
    tok = tok_full if pos_type == "full" else tok_coarse
    model, ckpt_dict = load_model(ckpt_path)
    lane_names = ckpt_dict.get("config", {}).get("multicontext_datasets", [])
    n_lanes = len(lane_names)
    print(f"\nEvaluating {model_id} ({rel_path})...")

    r_greedy = generate(model, tok, n_lanes, samples, mode="greedy")
    print(f"  [Standard Greedy] BLEU: {r_greedy['bleu']:.2f} | chrF: {r_greedy['chrf']:.2f} | Latency: {r_greedy['latency_ms']:.1f}ms")

    r_joint = generate(model, tok, n_lanes, samples, mode="joint_factor_product")
    print(f"  [Joint Factor Product] BLEU: {r_joint['bleu']:.2f} | chrF: {r_joint['chrf']:.2f} | Latency: {r_joint['latency_ms']:.1f}ms")

    results[model_id] = {
        "standard_greedy": r_greedy,
        "joint_factor_product": r_joint
    }

    del model
    torch.cuda.empty_cache()

out_file = os.path.join(REPO_ROOT, "results_remaining_nochar_decoding.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved results to {out_file}")
