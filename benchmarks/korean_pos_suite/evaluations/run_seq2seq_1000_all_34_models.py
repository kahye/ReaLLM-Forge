#!/usr/bin/env python3
"""
Fast Batched Evaluation of English-to-Korean Seq2Seq Translation Benchmark (N=1,000 test examples)
across all 34 Best-Validation-Loss Checkpoints (17 AdamW + 17 Muon).

Computes on N=1,000 examples:
1. Teacher-Forced Metrics:
   - Teacher-Forced BPJ (Bits-Per-Jamo)
   - Target Loss
   - Target Perplexity (PPL)
   (For Char-Head models, computes both WITH lm_head_24 [25 lanes] and WITHOUT lm_head_24 [24 lanes])
2. Autoregressive Generation Metrics (max_new_tokens=40, batched right-padded GPU inference):
   - Corpus BLEU-4 (on canonicalized Compatibility Jamo strings)
   - Corpus chrF (char_order=18 on canonicalized Compatibility Jamo strings)
   (For Char-Head models, evaluates all 3 decoding modes on N=1,000 examples:
    WITH lm_head_24 [charhead_24], WITHOUT lm_head_24 Greedy [primary], and WITHOUT lm_head_24 Joint Beam Search [jamo_lvt])
3. Single-Sentence Generation Latency (ms / sent):
   - Measured with batch_size=1 on 20 sentences with fused input embeddings and active decision heads.
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
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from jamo import h2j, j2hcj
import sacrebleu
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
from hangul_factorizer import (
    HangulPosFactorizedTokenizer,
    S_BASE,
    V_COUNT,
    T_COUNT,
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
)
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

TEST_JSONL = os.path.join(REPO_ROOT, "data", "korean_seq2seq_bench", "test.jsonl")

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


def canonicalize(text: str) -> str:
    """Canonicalize text according to EACL 2023 Appendix A & Section 4."""
    if not isinstance(text, str):
        return ""
    decomposed = j2hcj(h2j(text))
    stripped = re.sub(r"[^\w\s]", "", decomposed)
    return re.sub(r"\s+", " ", stripped).strip()


def count_jamos(text: str) -> int:
    canon = canonicalize(text)
    return max(1, len(canon.replace(" ", "")))


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[GPT, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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


def prepare_fused_model(model: GPT, active_heads: List[int], device: str = "cuda") -> None:
    """Pre-fuses input embedding tables and active LM decision heads for fast single-kernel inference."""
    if getattr(model.config, "vocab_sizes", None) is not None:
        if not hasattr(model, "_W_wte_concat"):
            output_only = set(getattr(model.config, "output_only_lanes", []) or [])
            active_in_lanes = [i for i in range(len(model.config.vocab_sizes)) if i not in output_only]
            wte_weights = [model.transformer[f"wte_{i}"].weight for i in active_in_lanes]
            model._active_in_lanes = torch.tensor(active_in_lanes, dtype=torch.long, device=device)
            model._wte_offsets = torch.tensor([0] + [w.size(0) for w in wte_weights[:-1]], device=device).cumsum(0).view(-1, 1, 1)
            model._W_wte_concat = torch.cat(wte_weights, dim=0)

        if not getattr(model.config, "three_hot_conditional", False):
            hw = [model.apply_lm_head_norm(model.transformer[f"lm_head_{h}"].weight) for h in active_heads]
            model._fused_head_sizes = [w.size(0) for w in hw]
            model._W_head_fused = torch.cat(hw, dim=0)
            model._active_heads = active_heads


@torch.no_grad()
def forward_fused_step(
    model: GPT,
    last_indices: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    buf_lanes: Optional[torch.Tensor] = None,
    max_c: Optional[int] = None,
) -> Any:
    """
    Runs Transformer backbone using fused multi-channel embeddings and projects only active decision heads.
    """
    if buf_lanes is not None:
        device = buf_lanes.device
        active_tokens = buf_lanes[model._active_in_lanes, :, :max_c]
        b = active_tokens.size(1)
        t = max_c
        flat_ids = active_tokens + model._wte_offsets
        x = F.embedding(flat_ids, model._W_wte_concat).sum(dim=0)

        if model.config.norm_variant_wte is not None:
            x = model.transformer.post_embedding_norm(x)
        if model.config.use_embedding_scale:
            x = x * model.embedding_scale
        if model.config.use_abs_pos_embeddings:
            pos_emb = model.transformer.wpe(t, device=device, training=False)
            x = model.transformer.drop(x + pos_emb)
        else:
            x = model.transformer.drop(x)

        if model.use_lsv and model.config.apply_lsv_at_layer_idx == 0:
            x = model.lsv_matrix(x)
        if model.use_ln_f_input_mixer:
            layer_outputs = [x]

        layer_idx = 1
        blocks = model.transformer.h
        if model.attention_residual_variant == "full":
            x = model._forward_full_attention_residual(x, None)
            blocks = ()
        for block in blocks:
            x = block(x, None)
            if model.use_lsv and layer_idx == model.config.apply_lsv_at_layer_idx:
                x = model.lsv_matrix(x)
            if model.use_ln_f_input_mixer:
                layer_outputs.append(x)
            layer_idx += 1

        if model.use_ln_f_input_mixer:
            x = model.ln_f_mixer(layer_outputs)
        x = model.transformer.ln_f(x)
        if model.n_embd_wte:
            x = F.linear(x, model.transformer.scale_down.weight.t())

        row_idx = torch.arange(b, device=device)
        x_last = x[row_idx, last_indices, :]

        if getattr(model.config, "three_hot_conditional", False):
            x_last_3d = x_last.unsqueeze(1)
            logits_0 = model.compute_lm_head_logits(x_last_3d, model.transformer["lm_head_0"])
            d = x_last_3d.size(-1)
            h0_exp = model.cond_h0.expand(b, 1, d)
            h_i = torch.tanh(model.cond_W_e(x_last_3d) + model.cond_W_h(h0_exp))
            logits_1 = model.compute_lm_head_logits(h_i, model.transformer["lm_head_1"])
            idx_i = torch.argmax(logits_1, dim=-1)
            emb_i = model.transformer["wte_1"](idx_i)
            h_v = torch.tanh(model.cond_W_e(emb_i) + model.cond_W_h(h_i))
            logits_2 = model.compute_lm_head_logits(h_v, model.transformer["lm_head_2"])
            idx_v = torch.argmax(logits_2, dim=-1)
            emb_v = model.transformer["wte_2"](idx_v)
            h_f = torch.tanh(model.cond_W_e(emb_v) + model.cond_W_h(h_v))
            logits_3 = model.compute_lm_head_logits(h_f, model.transformer["lm_head_3"])
            out = {0: logits_0[:, 0, :], 1: logits_1[:, 0, :], 2: logits_2[:, 0, :], 3: logits_3[:, 0, :]}
            if model.config.final_logit_softcapping is not None:
                sc = model.config.final_logit_softcapping
                out = {k: torch.tanh(v / sc) * sc for k, v in out.items()}
            return out
        else:
            all_logits = F.linear(x_last, model._W_head_fused)
            if model.config.final_logit_softcapping is not None:
                sc = model.config.final_logit_softcapping
                all_logits = torch.tanh(all_logits / sc) * sc
            splits = torch.split(all_logits, model._fused_head_sizes, dim=-1)
            return {h: s for h, s in zip(model._active_heads, splits)}

    else:
        device = idx.device
        b, t = idx.size()
        tok_emb = model.transformer.wte(idx)
        tok_emb = model.add_embedding_gaussian_noise(tok_emb, iter_num=None)
        if model.n_embd_wte:
            tok_emb = model.transformer.scale_up(tok_emb)
        if model.config.use_embedding_scale:
            tok_emb = tok_emb * model.embedding_scale
        if model.config.norm_variant_wte is not None:
            tok_emb = model.transformer.post_embedding_norm(tok_emb)

        if model.config.use_abs_pos_embeddings:
            pos_emb = model.transformer.wpe(t, device=device, training=False)
            x = tok_emb + pos_emb
            if model.config.norm_variant_abs is not None:
                x = model.transformer.post_abs_norm(x)
            x = model.transformer.drop(x)
        else:
            x = model.transformer.drop(tok_emb)

        if model.use_lsv and model.config.apply_lsv_at_layer_idx == 0:
            x = model.lsv_matrix(x)
        if model.use_ln_f_input_mixer:
            layer_outputs = [x]

        layer_idx = 1
        blocks = model.transformer.h
        if model.attention_residual_variant == "full":
            x = model._forward_full_attention_residual(x, None)
            blocks = ()
        for block in blocks:
            x = block(x, None)
            if model.use_lsv and layer_idx == model.config.apply_lsv_at_layer_idx:
                x = model.lsv_matrix(x)
            if model.use_ln_f_input_mixer:
                layer_outputs.append(x)
            layer_idx += 1

        if model.use_ln_f_input_mixer:
            x = model.ln_f_mixer(layer_outputs)
        x = model.transformer.ln_f(x)
        if model.n_embd_wte:
            x = F.linear(x, model.transformer.scale_down.weight.t())

        row_idx = torch.arange(b, device=device)
        x_last = x[row_idx, last_indices, :].unsqueeze(1)
        logits = model.compute_lm_head_logits(x_last, model.lm_head)
        if model.config.final_logit_softcapping is not None:
            sc = model.config.final_logit_softcapping
            logits = torch.tanh(logits / sc) * sc
        return logits[:, 0, :]


class Seq2SeqTokenizerBundle:
    """Holds pre-initialized tokenizers, vocabs, and GPU lookup tables for fast batched seq2seq eval."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Initializing tokenizers and syllable factor lookup tables...")

        # 1. Single-channel tokenizers
        self.single_meta = {}
        self.single_enc = {}
        self.single_dec = {}
        self.single_stop_ids = {}
        self.single_stop_tensors = {}

        for key, folder in [
            ("single_char", "out_baseline_korean_pos"),
            ("single_bpe", "out_baseline_char_bpe"),
            ("single_byte", "out_baseline_pure_byte"),
        ]:
            with open(os.path.join(REPO_ROOT, folder, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            enc, dec = get_tokenizer_functions(meta)
            self.single_meta[key] = meta
            self.single_enc[key] = enc
            self.single_dec[key] = dec
            if key == "single_byte":
                stops = {ord("\n")}
            else:
                stops = set()
                itos = meta.get("itos", {})
                for tid, piece in itos.items():
                    if (isinstance(piece, bytes) and b"\n" in piece) or (isinstance(piece, str) and "\n" in piece):
                        stops.add(tid)
            self.single_stop_ids[key] = stops
            self.single_stop_tensors[key] = torch.tensor(sorted(list(stops)), dtype=torch.long, device=device)

        # 2. Multi-channel 25-lane & 3-hot tokenizers
        self.tok_25_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
        self.tok_25_coarse = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")
        self.tok_three_hot = HangulPosFactorizedTokenizer(use_pos=False, pos_mode="full")

        self.mc25_stois = {}
        self.mc25_itos = {}
        for key, sample_ckpt in [
            ("mc_three_hot", "out_three_hot_ind_char_byte/best_ckpt.pt"),
            ("mc_25_full", "out_mc_full_pos_weighted_byte/best_ckpt.pt"),
            ("mc_25_coarse", "out_mc_coarse_pos_weighted_byte/best_ckpt.pt"),
        ]:
            ck = torch.load(os.path.join(REPO_ROOT, sample_ckpt), map_location="cpu", weights_only=False)
            lanes = ck.get("config", {}).get("multicontext_datasets", [])
            stois, itos = [], []
            for lp in lanes:
                with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
                    m = pickle.load(f)
                    stois.append(m["stoi"])
                    itos.append(m["itos"])
            self.mc25_stois[key] = stois
            self.mc25_itos[key] = itos

        # Build GPU factor tables for mc_three_hot, mc_25_full, mc_25_coarse
        self.mc25_tables = {}
        self.mc25_pads = {}
        self.mc25_lvt_maps = {}
        for key, tok in [
            ("mc_three_hot", self.tok_three_hot),
            ("mc_25_full", self.tok_25_full),
            ("mc_25_coarse", self.tok_25_coarse),
        ]:
            stois = self.mc25_stois[key]
            itos = self.mc25_itos[key]
            n_lanes = len(stois)
            l_map = torch.zeros(len(itos[1]), dtype=torch.long, device=device)
            for idx_i, ch in itos[1].items():
                if len(ch) == 1 and ord(ch) >= 0xE101:
                    l_map[idx_i] = min(18, max(0, ord(ch) - 0xE101))
            v_map = torch.zeros(len(itos[2]), dtype=torch.long, device=device)
            for idx_i, ch in itos[2].items():
                if len(ch) == 1 and ord(ch) >= 0xE201:
                    v_map[idx_i] = min(20, max(0, ord(ch) - 0xE201))
            t_map = torch.zeros(len(itos[3]), dtype=torch.long, device=device)
            for idx_i, ch in itos[3].items():
                if len(ch) == 1 and ord(ch) >= 0xE300:
                    t_map[idx_i] = min(27, max(0, ord(ch) - 0xE300))
            self.mc25_lvt_maps[key] = (l_map, v_map, t_map)
            if n_lanes >= 25:
                tbl = torch.zeros((11172, 25), dtype=torch.long)
                for s_idx in range(11172):
                    syl_char = chr(0xAC00 + s_idx)
                    vals = tok.encode_char(syl_char, pos_tag="UNK")
                    for li in range(23):
                        tbl[s_idx, li] = stois[li].get(tok.token_for(li, vals[li]), 0)
                    tbl[s_idx, 24] = ord(syl_char.encode("utf-8")[0:1])
                self.mc25_tables[key] = tbl.to(device)
                pad_vec = torch.zeros(23, dtype=torch.long)
                for li in range(23):
                    val_id = 2 if li == 0 else 0
                    pad_vec[li] = stois[li].get(tok.token_for(li, val_id), 0)
                self.mc25_pads[key] = pad_vec.to(device)
            else:
                lead_bytes = torch.tensor(
                    [ord(chr(0xAC00 + s).encode("utf-8")[0:1]) for s in range(11172)],
                    dtype=torch.long,
                    device=device,
                )
                self.mc25_tables[key] = lead_bytes

        # 3. Hybrid 24/25-lane tokenizers (nochar & charhead)
        spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
        self.tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
        self.tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
        self.hangul_id = self.tok_24_full.hangul_id

        self.hybrid_stop_ids = set(self.tok_24_full.sp.encode_as_ids("\n"))
        if self.tok_24_full.sp.eos_id() != -1:
            self.hybrid_stop_ids.add(self.tok_24_full.sp.eos_id())
        for tid in range(self.tok_24_full.sp.get_piece_size()):
            if "\n" in self.tok_24_full.sp.id_to_piece(tid):
                self.hybrid_stop_ids.add(tid)
        self.hybrid_stop_tensor = torch.tensor(sorted(list(self.hybrid_stop_ids)), dtype=torch.long, device=device)

        syl_table = torch.zeros((11172, 23), dtype=torch.long)
        for s_idx in range(11172):
            syl_char = chr(0xAC00 + s_idx)
            step = self.tok_24_full.encode_text(syl_char)[0]
            for l_i in range(min(23, len(step))):
                syl_table[s_idx, l_i] = step[l_i]
        self.syl_table_gpu = syl_table.to(device)
        print("Initialized all tokenizers and precomputed 11,172 Hangul syllable factor tables on GPU.")


def _worker_seq2seq_mc25_chunk(args):
    key, chunk = args
    bundle = Seq2SeqTokenizerBundle(device="cpu")
    tok = {
        "mc_three_hot": bundle.tok_three_hot,
        "mc_25_full": bundle.tok_25_full,
        "mc_25_coarse": bundle.tok_25_coarse,
    }[key]
    stois = bundle.mc25_stois[key]
    n_lanes = len(stois)
    items = []
    for s in chunk:
        en, ko = s["en"], s["ko"]
        prompt = f"{en}\t"
        full = f"{en}\t{ko}\n"
        p_seq = tok.encode_text(prompt)
        f_seq = tok.encode_text(full)[:256]

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

        items.append({
            "prompt_lanes": seq_to_lanes(p_seq),
            "full_lanes": seq_to_lanes(f_seq),
            "n_jamos": count_jamos(ko),
        })
    return items


def _worker_seq2seq_hybrid_chunk(args):
    key, chunk = args
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode=("full" if "full" in key else "coarse"))
    items_25 = []
    items_24 = []
    for s in chunk:
        en, ko = s["en"], s["ko"]
        prompt = f"{en}\t"
        full = f"{en}\t{ko}\n"
        p_seq = tok.encode_text(prompt)
        f_seq = tok.encode_text(full)[:256]

        def hybrid_to_lanes(seq):
            tl = [[] for _ in range(25)]
            for step in seq:
                for i in range(24):
                    tl[i].append(step[i])
                if step[0] == tok.hangul_id:
                    char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                else:
                    char_id = 0
                tl[24].append(char_id)
            return tl

        p_25 = hybrid_to_lanes(p_seq)
        f_25 = hybrid_to_lanes(f_seq)
        nj = count_jamos(ko)
        items_25.append({"prompt_lanes": p_25, "full_lanes": f_25, "n_jamos": nj})
        items_24.append({"prompt_lanes": p_25[:24], "full_lanes": f_25[:24], "n_jamos": nj})
    return items_25, items_24


def pretokenize_all_1000(samples: List[Dict[str, str]], bundle: Seq2SeqTokenizerBundle) -> Dict[str, Any]:
    import multiprocessing as mp
    cache = {}
    n_workers = min(12, mp.cpu_count() or 4)
    print(f"Pre-tokenizing {len(samples)} parallel sentences across all tokenizer configurations using {n_workers} workers...")

    # 1. Single-channel
    for key in ["single_char", "single_bpe", "single_byte"]:
        t0 = time.time()
        enc = bundle.single_enc[key]
        items = []
        for s in samples:
            en, ko = s["en"], s["ko"]
            prompt = f"{en}\t"
            full = f"{en}\t{ko}\n"
            p_ids = enc(prompt)
            f_ids = enc(full)[:256]
            items.append({
                "prompt_ids": p_ids,
                "full_ids": f_ids,
                "n_jamos": count_jamos(ko),
            })
        cache[key] = items
        print(f"  Pre-tokenized {key} in {time.time() - t0:.2f}s")

    # 2. mc_three_hot, mc_25_full, mc_25_coarse
    chunk_size = max(1, (len(samples) + n_workers - 1) // n_workers)
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    for key in ["mc_three_hot", "mc_25_full", "mc_25_coarse"]:
        t0 = time.time()
        with mp.get_context("spawn").Pool(n_workers) as pool:
            res_chunks = pool.map(_worker_seq2seq_mc25_chunk, [(key, c) for c in chunks])
        cache[key] = [item for sub in res_chunks for item in sub]
        print(f"  Pre-tokenized {key} in {time.time() - t0:.2f}s")

    # 3. mc_25_charhead_full, mc_25_charhead_coarse (and derive mc_24_full, mc_24_coarse)
    for key, dst_24 in [
        ("mc_25_charhead_full", "mc_24_full"),
        ("mc_25_charhead_coarse", "mc_24_coarse"),
    ]:
        t0 = time.time()
        with mp.get_context("spawn").Pool(n_workers) as pool:
            res_chunks = pool.map(_worker_seq2seq_hybrid_chunk, [(key, c) for c in chunks])
        cache[key] = [item for sub_25, _ in res_chunks for item in sub_25]
        cache[dst_24] = [item for _, sub_24 in res_chunks for item in sub_24]
        print(f"  Pre-tokenized {key} & derived {dst_24} in {time.time() - t0:.2f}s")

    return cache


_SEQ2SEQ_TF_CACHE = {}
_SEQ2SEQ_GEN_CACHE = {}


@torch.no_grad()
def evaluate_teacher_forced_batched(
    model: GPT,
    items: List[Dict[str, Any]],
    tok_key: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    import numpy as np
    is_single = tok_key.startswith("single_")
    has_charhead = "charhead" in tok_key
    vocab_size = getattr(model.config, "vocab_size", None)
    vocab_sizes = getattr(model.config, "vocab_sizes", None)

    total_nll_25 = 0.0
    total_nll_24 = 0.0
    total_tokens = 0
    total_jamos = 0

    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    if is_single:
        cache_key = (id(items), batch_size, "single")
        if cache_key not in _SEQ2SEQ_TF_CACHE:
            valid_items = [x for x in items if len(x["full_ids"]) > len(x["prompt_ids"])]
            batches = []
            tot_j = 0
            for b_start in range(0, len(valid_items), batch_size):
                batch = valid_items[b_start : b_start + batch_size]
                B = len(batch)
                max_len = max(len(x["full_ids"]) - 1 for x in batch)
                inp_np = np.zeros((B, max_len), dtype=np.int64)
                tgt_np = np.full((B, max_len), -1, dtype=np.int64)
                for i, x in enumerate(batch):
                    f_ids = x["full_ids"]
                    p_len = len(x["prompt_ids"])
                    L = len(f_ids) - 1
                    mask_len = min(p_len - 1, L)
                    arr = np.array(f_ids, dtype=np.int64)
                    inp_np[i, :L] = arr[:-1]
                    tgt_np[i, :L] = arr[1:]
                    tgt_np[i, :mask_len] = -1
                    tot_j += x["n_jamos"]
                batches.append((torch.from_numpy(inp_np), torch.from_numpy(tgt_np)))
            _SEQ2SEQ_TF_CACHE[cache_key] = (batches, tot_j)

        batches, total_jamos = _SEQ2SEQ_TF_CACHE[cache_key]
        for inp_cpu, tgt_cpu in batches:
            if vocab_size is not None:
                inp = inp_cpu.clamp(0, vocab_size - 1).to(device, non_blocking=True)
                tgt = torch.where(tgt_cpu != -1, tgt_cpu.clamp(0, vocab_size - 1), tgt_cpu).to(device, non_blocking=True)
            else:
                inp = inp_cpu.to(device, non_blocking=True)
                tgt = tgt_cpu.to(device, non_blocking=True)

            with ctx_autocast:
                logits, _ = model(inp, targets=tgt)
            loss_flat = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1, reduction="sum")
            n_tgt = (tgt != -1).sum().item()
            total_nll_25 += loss_flat.item()
            total_tokens += n_tgt

        avg_loss = total_nll_25 / max(1, total_tokens)
        bpj = total_nll_25 / (math.log(2.0) * max(1, total_jamos))
        ppl = math.exp(min(20.0, avg_loss))
        return {"primary": {"bpj": bpj, "target_loss": avg_loss, "target_ppl": ppl}}

    else:
        n_lanes = len(items[0]["full_lanes"])
        cache_key = (id(items), batch_size, "mc")
        if cache_key not in _SEQ2SEQ_TF_CACHE:
            valid_items = [x for x in items if len(x["full_lanes"][0]) > len(x["prompt_lanes"][0])]
            batches = []
            tot_j = 0
            for b_start in range(0, len(valid_items), batch_size):
                batch = valid_items[b_start : b_start + batch_size]
                B = len(batch)
                max_len = max(len(x["full_lanes"][0]) - 1 for x in batch)
                inp_np = np.zeros((n_lanes, B, max_len), dtype=np.int64)
                tgt_np = np.full((n_lanes, B, max_len), -1, dtype=np.int64)
                for i, x in enumerate(batch):
                    f_lanes = x["full_lanes"]
                    p_len = len(x["prompt_lanes"][0])
                    L = len(f_lanes[0]) - 1
                    mask_len = min(p_len - 1, L)
                    arr = np.array(f_lanes, dtype=np.int64)
                    inp_np[:, i, :L] = arr[:, :-1]
                    tgt_np[:, i, :L] = arr[:, 1:]
                    tgt_np[:, i, :mask_len] = -1
                    tot_j += x["n_jamos"]
                inp_t = torch.from_numpy(inp_np)
                tgt_t = torch.from_numpy(tgt_np)
                batches.append(([inp_t[l] for l in range(n_lanes)], [tgt_t[l] for l in range(n_lanes)]))
            _SEQ2SEQ_TF_CACHE[cache_key] = (batches, tot_j)

        batches, total_jamos = _SEQ2SEQ_TF_CACHE[cache_key]
        for b_inputs_cpu, b_targets_cpu in batches:
            b_inputs = []
            b_targets = []
            for l_idx in range(n_lanes):
                if vocab_sizes is not None and l_idx < len(vocab_sizes):
                    b_inputs.append(b_inputs_cpu[l_idx].clamp(0, vocab_sizes[l_idx] - 1).to(device, non_blocking=True))
                    t_c = b_targets_cpu[l_idx]
                    b_targets.append(torch.where(t_c != -1, t_c.clamp(0, vocab_sizes[l_idx] - 1), t_c).to(device, non_blocking=True))
                else:
                    b_inputs.append(b_inputs_cpu[l_idx].to(device, non_blocking=True))
                    b_targets.append(b_targets_cpu[l_idx].to(device, non_blocking=True))

            token_dict = {f"lane_{i}": b_inputs[i] for i in range(n_lanes)}
            target_dict = {f"lane_{i}": b_targets[i] for i in range(n_lanes)}
            with ctx_autocast:
                logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)

            lane_nlls = []
            for l_idx in range(n_lanes):
                logits_l = logits_list[l_idx]
                tgt_l = b_targets[l_idx]
                l_sum = F.cross_entropy(
                    logits_l.view(-1, logits_l.size(-1)),
                    tgt_l.view(-1),
                    ignore_index=-1,
                    reduction="sum",
                ).item()
                lane_nlls.append(l_sum)

            n_tgt = (b_targets[0] != -1).sum().item()
            total_tokens += n_tgt

            avg_nll_all = sum(lane_nlls) / n_lanes
            total_nll_25 += avg_nll_all

            if has_charhead:
                avg_nll_24 = sum(lane_nlls[:24]) / 24.0
                total_nll_24 += avg_nll_24

        avg_loss_25 = total_nll_25 / max(1, total_tokens)
        bpj_25 = total_nll_25 / (math.log(2.0) * max(1, total_jamos))
        ppl_25 = math.exp(min(20.0, avg_loss_25))

        res = {"primary": {"bpj": bpj_25, "target_loss": avg_loss_25, "target_ppl": ppl_25}}
        if has_charhead:
            avg_loss_24 = total_nll_24 / max(1, total_tokens)
            bpj_24 = total_nll_24 / (math.log(2.0) * max(1, total_jamos))
            ppl_24 = math.exp(min(20.0, avg_loss_24))
            res["charhead_24"] = {"bpj": bpj_25, "target_loss": avg_loss_25, "target_ppl": ppl_25}
            res["primary"] = {"bpj": bpj_24, "target_loss": avg_loss_24, "target_ppl": ppl_24}
            res["jamo_lvt"] = {"bpj": bpj_24, "target_loss": avg_loss_24, "target_ppl": ppl_24}
        return res


@torch.no_grad()
def generate_batched_1000(
    model: GPT,
    items: List[Dict[str, Any]],
    tok_key: str,
    bundle: Seq2SeqTokenizerBundle,
    mode: str = "primary",
    max_new_tokens: int = 40,
    batch_size: int = 64,
    device: str = "cuda",
) -> List[str]:
    import numpy as np
    is_single = tok_key.startswith("single_")
    vocab_size = getattr(model.config, "vocab_size", None)
    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    ctx_autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    hypotheses = []

    if is_single:
        stop_tensor = bundle.single_stop_tensors[tok_key]
        dec_fn = bundle.single_dec[tok_key]
        cache_key = (id(items), len(items), tok_key, max_new_tokens, batch_size, "single")
        if cache_key not in _SEQ2SEQ_GEN_CACHE:
            batches = []
            for b_start in range(0, len(items), batch_size):
                batch = items[b_start : b_start + batch_size]
                B = len(batch)
                prompt_lens = [max(1, min(len(x["prompt_ids"]), 256 - max_new_tokens)) for x in batch]
                max_p = max(prompt_lens)
                max_total_len = max_p + max_new_tokens
                buf_np = np.zeros((B, max_total_len), dtype=np.int64)
                for i, x in enumerate(batch):
                    L = prompt_lens[i]
                    buf_np[i, :L] = np.array(x["prompt_ids"][-L:], dtype=np.int64)
                batches.append((torch.from_numpy(buf_np), torch.tensor(prompt_lens, dtype=torch.long), prompt_lens, max_p, max_total_len))
            _SEQ2SEQ_GEN_CACHE[cache_key] = batches

        for buf_cpu, curr_lens_cpu, prompt_lens, max_p, max_total_len in _SEQ2SEQ_GEN_CACHE[cache_key]:
            B = len(prompt_lens)
            buf = buf_cpu.to(device, non_blocking=True)
            if vocab_size is not None:
                buf[:, :max_p].clamp_(0, vocab_size - 1)
            curr_lens = curr_lens_cpu.to(device, non_blocking=True)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            row_idx = torch.arange(B, device=device)

            for step in range(max_new_tokens):
                if done.all():
                    break
                max_c = max_p + step
                with ctx_autocast:
                    next_logits = forward_fused_step(model, last_indices=curr_lens - 1, idx=buf[:, :max_c])
                next_toks = torch.argmax(next_logits, dim=-1)
                if vocab_size is not None:
                    next_toks.clamp_(0, vocab_size - 1)

                write_pos = curr_lens.clamp(max=max_total_len - 1)
                buf[row_idx, write_pos] = torch.where(done, buf[row_idx, write_pos], next_toks)
                is_stop = torch.isin(next_toks, stop_tensor)
                curr_lens = curr_lens + (~done).long()
                done = done | is_stop

            max_e_len = int(curr_lens.max().item())
            cpu_buf = buf[:, :max_e_len].cpu().tolist()
            cpu_curr_lens = curr_lens.cpu().tolist()
            for i in range(B):
                gen_ids = cpu_buf[i][prompt_lens[i] : cpu_curr_lens[i]]
                if tok_key == "single_byte":
                    pred_str = bytes([b for b in gen_ids if 0 <= b < 256]).decode("utf-8", errors="ignore").strip()
                else:
                    pred_str = dec_fn(gen_ids).strip()
                hypotheses.append(pred_str)

    else:
        n_lanes = len(items[0]["prompt_lanes"])
        is_hybrid = "mc_24" in tok_key or "charhead" in tok_key
        is_mc25 = "mc_25_full" in tok_key or "mc_25_coarse" in tok_key
        is_3hot = "mc_three_hot" in tok_key

        if is_hybrid:
            active_heads = [0, 23, 24] if mode == "charhead_24" else [0, 1, 2, 3, 23]
        elif is_mc25:
            active_heads = [0, 1, 2, 3, 23, 24]
        else:
            active_heads = [0, 1, 2, 3]

        prepare_fused_model(model, active_heads, device=device)

        cache_key = (id(items), len(items), tok_key, max_new_tokens, batch_size, "mc")
        if cache_key not in _SEQ2SEQ_GEN_CACHE:
            batches = []
            for b_start in range(0, len(items), batch_size):
                batch = items[b_start : b_start + batch_size]
                B = len(batch)
                prompt_lens = [max(1, min(len(x["prompt_lanes"][0]), 256 - max_new_tokens)) for x in batch]
                max_p = max(prompt_lens)
                max_total_len = max_p + max_new_tokens
                buf_np = np.zeros((n_lanes, B, max_total_len), dtype=np.int64)
                for i, x in enumerate(batch):
                    L = prompt_lens[i]
                    arr = np.array([x["prompt_lanes"][l][-L:] for l in range(n_lanes)], dtype=np.int64)
                    buf_np[:, i, :L] = arr
                batches.append((torch.from_numpy(buf_np), torch.tensor(prompt_lens, dtype=torch.long), prompt_lens, max_p, max_total_len))
            _SEQ2SEQ_GEN_CACHE[cache_key] = batches

        for buf_cpu, curr_lens_cpu, prompt_lens, max_p, max_total_len in _SEQ2SEQ_GEN_CACHE[cache_key]:
            B = len(prompt_lens)
            buf_lanes = buf_cpu.to(device, non_blocking=True)
            curr_lens = curr_lens_cpu.to(device, non_blocking=True)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            row_idx = torch.arange(B, device=device)

            for step in range(max_new_tokens):
                if done.all():
                    break
                max_c = max_p + step
                with ctx_autocast:
                    logits_dict = forward_fused_step(
                        model,
                        last_indices=curr_lens - 1,
                        buf_lanes=buf_lanes,
                        max_c=max_c,
                    )

                preds = {h: torch.argmax(lv, dim=-1) for h, lv in logits_dict.items()}

                if is_hybrid:
                    is_hangul = preds[0] == bundle.hangul_id
                    if mode == "jamo_lvt":
                        l1 = logits_dict[1][:, 1:20]
                        l2 = logits_dict[2][:, 1:22]
                        l3 = logits_dict[3][:, 0:28]
                        lp_L = F.log_softmax(l1, dim=-1)
                        lp_V = F.log_softmax(l2, dim=-1)
                        lp_T = F.log_softmax(l3, dim=-1)
                        joint_lp = (
                            lp_L.unsqueeze(2).unsqueeze(3)
                            + lp_V.unsqueeze(1).unsqueeze(3)
                            + lp_T.unsqueeze(1).unsqueeze(2)
                        ).view(B, 11172)
                        best_s_idx = torch.argmax(joint_lp, dim=-1)
                    elif mode == "charhead_24" and n_lanes >= 25:
                        l24 = logits_dict[24][:, 1:11173]
                        best_s_idx = torch.argmax(l24, dim=-1)
                    else:
                        c_l = preds[1].clamp(1, 19) - 1
                        c_v = preds[2].clamp(1, 21) - 1
                        c_t = preds[3].clamp(0, 27)
                        best_s_idx = c_l * 588 + c_v * 28 + c_t

                    h_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    h_toks[:, :23] = bundle.syl_table_gpu[best_s_idx]
                    h_toks[:, 23] = preds[23]
                    if n_lanes >= 25:
                        h_toks[:, 24] = best_s_idx + 1

                    nh_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    nh_toks[:, 0] = preds[0]
                    nh_toks[:, 23] = preds[23]

                    next_tokens = torch.where(is_hangul.unsqueeze(1), h_toks, nh_toks)
                    is_stop = torch.isin(next_tokens[:, 0], bundle.hybrid_stop_tensor)

                elif is_mc25:
                    is_hangul = preds[0] == 0
                    l_map, v_map, t_map = bundle.mc25_lvt_maps[tok_key]
                    s_idx = (l_map[preds[1]] * 21 + v_map[preds[2]]) * 28 + t_map[preds[3]]
                    tbl_rows = bundle.mc25_tables[tok_key][s_idx]

                    h_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    h_toks[:, :23] = tbl_rows[:, :23]
                    h_toks[:, 23] = preds[23]
                    h_toks[:, 24] = tbl_rows[:, 24]

                    nh_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    nh_toks[:, :23] = bundle.mc25_pads[tok_key].unsqueeze(0).expand(B, -1)
                    nh_toks[:, 23] = preds[23]
                    nh_toks[:, 24] = preds[24]

                    next_tokens = torch.where(is_hangul.unsqueeze(1), h_toks, nh_toks)
                    is_stop = (~is_hangul) & (next_tokens[:, 24] == 10)

                elif is_3hot:
                    l_map, v_map, t_map = bundle.mc25_lvt_maps[tok_key]
                    is_hangul = (preds[1] > 0) & (preds[2] > 0)
                    s_idx = (l_map[preds[1]] * 21 + v_map[preds[2]]) * 28 + t_map[preds[3]]

                    h_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    h_toks[:, 0] = bundle.mc25_tables[tok_key][s_idx]
                    h_toks[:, 1] = preds[1]
                    h_toks[:, 2] = preds[2]
                    h_toks[:, 3] = preds[3]

                    nh_toks = torch.zeros((B, n_lanes), dtype=torch.long, device=device)
                    nh_toks[:, 0] = preds[0]

                    next_tokens = torch.where(is_hangul.unsqueeze(1), h_toks, nh_toks)
                    is_stop = (~is_hangul) & (next_tokens[:, 0] == 10)

                write_pos = curr_lens.clamp(max=max_total_len - 1)
                buf_lanes[:, row_idx, write_pos] = torch.where(
                    done.unsqueeze(0),
                    buf_lanes[:, row_idx, write_pos],
                    next_tokens.t(),
                )
                curr_lens = curr_lens + (~done).long()
                done = done | is_stop

            max_e_len = int(curr_lens.max().item())
            cpu_curr_lens = curr_lens.cpu().tolist()

            if is_hybrid:
                cpu_lanes = buf_lanes[:4, :, :max_e_len].cpu().tolist()
                tok = bundle.tok_24_full if "full" in tok_key else bundle.tok_24_coarse
                for i in range(B):
                    s_len = prompt_lens[i]
                    e_len = cpu_curr_lens[i]
                    gen_steps = [
                        [cpu_lanes[l_idx][i][step_idx] for l_idx in range(4)]
                        for step_idx in range(s_len, e_len)
                    ]
                    try:
                        pred_str = tok.decode_sequence(gen_steps)
                    except Exception:
                        pred_str = ""
                    hypotheses.append(pred_str.strip())
            else:
                char_lane_idx = 0 if n_lanes == 4 else 24
                dec_lanes = [0, 1, 2, 3] if n_lanes == 4 else [0, 1, 2, 3, 24]
                cpu_lanes = buf_lanes[dec_lanes, :, :max_e_len].cpu().tolist()
                itos = bundle.mc25_itos[tok_key]
                for i in range(B):
                    s_len = prompt_lens[i]
                    e_len = cpu_curr_lens[i]
                    decoded_chars = []
                    for step_idx in range(s_len, e_len):
                        r0 = cpu_lanes[0][i][step_idx]
                        r1 = cpu_lanes[1][i][step_idx]
                        r2 = cpu_lanes[2][i][step_idx]
                        r3 = cpu_lanes[3][i][step_idx]
                        rc = cpu_lanes[0 if n_lanes == 4 else 4][i][step_idx]
                        if (n_lanes >= 25 and r0 == 0) or (n_lanes == 4 and r1 > 0 and r2 > 0):
                            t1 = itos[1].get(r1, "")
                            t2 = itos[2].get(r2, "")
                            t3 = itos[3].get(r3, "")
                            if len(t1) > 0 and len(t2) > 0 and len(t3) > 0:
                                l_val = (ord(t1) - 0xE100) - 1
                                v_val = (ord(t2) - 0xE200) - 1
                                t_val = ord(t3) - 0xE300
                                if 0 <= l_val < len(CHOSEONG) and 0 <= v_val < len(JUNGSEONG) and 0 <= t_val < len(JONGSEONG):
                                    decoded_chars.append(chr(S_BASE + (l_val * V_COUNT + v_val) * T_COUNT + t_val))
                                else:
                                    decoded_chars.append(chr(rc) if 0 <= rc < 0x110000 else "")
                            else:
                                decoded_chars.append(chr(rc) if 0 <= rc < 0x110000 else "")
                        else:
                            decoded_chars.append(chr(rc) if 0 <= rc < 0x110000 else "")
                    hypotheses.append("".join(decoded_chars).strip())

    return hypotheses


@torch.no_grad()
def measure_single_stream_latency_ms(
    model: GPT,
    items: List[Dict[str, Any]],
    tok_key: str,
    bundle: Seq2SeqTokenizerBundle,
    mode: str = "primary",
    num_latency_samples: int = 20,
    max_new_tokens: int = 40,
    device: str = "cuda",
) -> float:
    """Measures single-sentence (batch_size=1) generation latency (ms / sent) on num_latency_samples."""
    warmup_items = items[:2]
    sub_items = items[:num_latency_samples]
    _ = generate_batched_1000(
        model=model,
        items=warmup_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        device=device,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    _ = generate_batched_1000(
        model=model,
        items=sub_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        device=device,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000.0 / max(1, len(sub_items))
    return elapsed_ms


def compute_bleu_chrf(hyps: List[str], canon_refs: List[str]) -> Tuple[float, float]:
    canon_hyps = [canonicalize(h) for h in hyps]
    filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
    filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]
    bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
    chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score
    return float(bleu), float(chrf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="results_seq2seq_1000_all_models.json")
    parser.add_argument("--force", action="store_true", help="Re-evaluate all models even if present in output_json")
    parser.add_argument("--latency_only", action="store_true", help="Re-measure single-stream latency for all models in output_json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    samples = all_samples[: (30 if args.latency_only else args.max_examples)]
    canon_refs = [canonicalize(s["ko"]) for s in samples]
    print(f"Loaded {len(samples)} English-to-Korean parallel test sentences from {TEST_JSONL}.")

    bundle = Seq2SeqTokenizerBundle(device=device)
    pre_cache = pretokenize_all_1000(samples, bundle)

    out_path = os.path.join(REPO_ROOT, args.output_json)
    if os.path.exists(out_path) and not args.force:
        with open(out_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded existing {out_path}: AdamW={len(all_results.get('adamw', {}))}, Muon={len(all_results.get('muon', {}))}")
    else:
        all_results = {"adamw": {}, "muon": {}, "num_examples": len(samples)}

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== Evaluating {suite_name.upper()} Suite (17 models) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            if not args.force and not args.latency_only and m_id in all_results.get(suite_name, {}):
                print(f"  [SKIP ALREADY DONE] [{suite_name.upper()}] {m_id}")
                continue
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"  [MISSING] {ckpt_path}")
                continue

            t0 = time.time()
            model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=device)
            iter_num = int(ckpt_dict.get("iter_num", 0))
            items = pre_cache[tok_key]

            has_charhead = "charhead" in tok_key
            is_nochar_24 = "mc_24" in tok_key

            if has_charhead:
                modes_to_eval = ["charhead_24", "primary", "jamo_lvt"]
            elif is_nochar_24:
                modes_to_eval = ["primary", "jamo_lvt"]
            else:
                modes_to_eval = ["primary"]

            if args.latency_only and m_id in all_results.get(suite_name, {}):
                model_entry = all_results[suite_name][m_id]
                for mode_name in modes_to_eval:
                    lat_ms = measure_single_stream_latency_ms(
                        model=model,
                        items=items,
                        tok_key=tok_key,
                        bundle=bundle,
                        mode=mode_name,
                        num_latency_samples=20,
                        max_new_tokens=40,
                        device=device,
                    )
                    model_entry[mode_name]["latency_ms"] = round(lat_ms, 1)
            else:
                tf_res = evaluate_teacher_forced_batched(model, items, tok_key, device=device, batch_size=args.batch_size)
                model_entry = {"iter_num": iter_num}
                for mode_name in modes_to_eval:
                    hyps = generate_batched_1000(
                        model=model,
                        items=items,
                        tok_key=tok_key,
                        bundle=bundle,
                        mode=mode_name,
                        max_new_tokens=40,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    bleu, chrf = compute_bleu_chrf(hyps, canon_refs)
                    lat_ms = measure_single_stream_latency_ms(
                        model=model,
                        items=items,
                        tok_key=tok_key,
                        bundle=bundle,
                        mode=mode_name,
                        num_latency_samples=20,
                        max_new_tokens=40,
                        device=device,
                    )
                    tf_m = tf_res.get(mode_name, tf_res["primary"])
                    model_entry[mode_name] = {
                        "bpj": round(tf_m["bpj"], 4),
                        "target_loss": round(tf_m["target_loss"], 4),
                        "target_ppl": round(tf_m["target_ppl"], 2),
                        "bleu": round(bleu, 2),
                        "chrf": round(chrf, 2),
                        "latency_ms": round(lat_ms, 1),
                    }

            all_results[suite_name][m_id] = model_entry
            dt = time.time() - t0
            p = model_entry["primary"]
            msg = (
                f"  [{suite_name.upper()}] {m_id:38s} | BPJ={p['bpj']:.4f} PPL={p['target_ppl']:5.2f} "
                f"BLEU={p['bleu']:4.2f} chrF={p['chrf']:5.2f} Lat={p['latency_ms']:5.1f}ms"
            )
            if "charhead_24" in model_entry:
                c = model_entry["charhead_24"]
                j = model_entry["jamo_lvt"]
                msg += f" || WITH_24: BPJ={c['bpj']:.4f} chrF={c['chrf']:5.2f} Lat={c['latency_ms']:5.1f}ms | JointLVT: chrF={j['chrf']:5.2f} Lat={j['latency_ms']:5.1f}ms"
            elif "jamo_lvt" in model_entry:
                j = model_entry["jamo_lvt"]
                msg += f" || JointLVT: chrF={j['chrf']:5.2f} Lat={j['latency_ms']:5.1f}ms"
            msg += f" [{dt:.1f}s]"
            print(msg)

            del model
            torch.cuda.empty_cache()

            out_path = os.path.join(REPO_ROOT, args.output_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted evaluation across all 34 models! Saved to {args.output_json}")


if __name__ == "__main__":
    main()
