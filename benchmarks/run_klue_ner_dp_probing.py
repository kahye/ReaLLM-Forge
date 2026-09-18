#!/usr/bin/env python3
"""
KLUE-NER (Entity-Level & Character-Level Macro F1 %) and
KLUE-DP (Unlabeled & Labeled Attachment Score UAS/LAS %) Evaluation Suite
using Canonical Frozen-Backbone Compositional Probing (Approach 1).

Evaluates all 34 unique checkpoints (17 AdamW Best Validation Loss + 17 Muon Best Validation Loss)
using:
- N_train = 3,000 / N_val = 1,000 examples for both KLUE-NER and KLUE-DP
- Canonical frozen encoder representation per checkpoint (unifying decoding configurations
  WITH lm_head_24, WITHOUT lm_head_24 Greedy, and WITHOUT lm_head_24 Joint Beam Search
  which share the exact same frozen transformer backbone)
- Stabilized 3-seed averaging (seeds = [42, 43, 44]) with CosineAnnealingLR, gradient clipping
  (max_norm=1.0), and deterministic per-epoch batch shuffling.
"""
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "benchmarks", "korean_pos_suite", "evaluations"),
    os.path.join(REPO_ROOT, "benchmarks", "korean_pos_suite", "training"),
    os.path.join(REPO_ROOT, "data", "template", "utils", "korean"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_all_eval_suites import load_model, load_tokenizers
from hangul_factorizer import POS_TAG_MAP

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
toks = load_tokenizers()

KLUE_NER_TAGS = [
    "B-DT", "I-DT", "B-LC", "I-LC", "B-OG", "I-OG",
    "B-PS", "I-PS", "B-QT", "I-QT", "B-TI", "I-TI", "O"
]
ENTITY_TYPES = ["DT", "LC", "OG", "PS", "QT", "TI"]


def extract_entities(tag_ids: List[int]) -> List[Tuple[str, int, int]]:
    entities = []
    cur_type = None
    cur_start = None
    for i, tid in enumerate(tag_ids):
        tag = KLUE_NER_TAGS[tid] if 0 <= tid < len(KLUE_NER_TAGS) else "O"
        if tag == "O":
            if cur_type is not None:
                entities.append((cur_type, cur_start, i))
                cur_type = None
        elif tag.startswith("B-"):
            if cur_type is not None:
                entities.append((cur_type, cur_start, i))
            cur_type = tag[2:]
            cur_start = i
        elif tag.startswith("I-"):
            t_type = tag[2:]
            if cur_type != t_type:
                if cur_type is not None:
                    entities.append((cur_type, cur_start, i))
                cur_type = t_type
                cur_start = i
    if cur_type is not None:
        entities.append((cur_type, cur_start, len(tag_ids)))
    return entities


def compute_klue_ner_metrics(gold_seqs: List[List[int]], pred_seqs: List[List[int]]) -> Tuple[float, float]:
    tp = {t: 0 for t in ENTITY_TYPES}
    fp = {t: 0 for t in ENTITY_TYPES}
    fn = {t: 0 for t in ENTITY_TYPES}

    for g_seq, p_seq in zip(gold_seqs, pred_seqs):
        L = min(len(g_seq), len(p_seq))
        g_ents = set(extract_entities(g_seq[:L]))
        p_ents = set(extract_entities(p_seq[:L]))
        for etype, s, e in p_ents:
            if (etype, s, e) in g_ents:
                tp[etype] += 1
            else:
                fp[etype] += 1
        for etype, s, e in g_ents:
            if (etype, s, e) not in p_ents:
                fn[etype] += 1

    f1s = []
    for t in ENTITY_TYPES:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1s.append(f1)
    entity_macro_f1 = float(np.mean(f1s)) * 100.0

    num_classes = len(KLUE_NER_TAGS)
    c_tp = np.zeros(num_classes)
    c_fp = np.zeros(num_classes)
    c_fn = np.zeros(num_classes)
    for g_seq, p_seq in zip(gold_seqs, pred_seqs):
        for g, p in zip(g_seq, p_seq):
            if g == p:
                c_tp[g] += 1
            else:
                c_fp[p] += 1
                c_fn[g] += 1

    c_f1s = []
    for c in range(num_classes):
        if c_tp[c] + c_fn[c] == 0:
            continue
        p = c_tp[c] / (c_tp[c] + c_fp[c]) if (c_tp[c] + c_fp[c]) > 0 else 0.0
        r = c_tp[c] / (c_tp[c] + c_fn[c]) if (c_tp[c] + c_fn[c]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        c_f1s.append(f1)
    char_macro_f1 = float(np.mean(c_f1s)) * 100.0
    return entity_macro_f1, char_macro_f1


def pretokenize_sentences(sentences: List[str], tok_family: str) -> List[Tuple[Any, torch.Tensor]]:
    results = []
    for text in sentences:
        C = len(text)
        if tok_family == "single_char":
            stoi = toks["meta_char"]["stoi"]
            ids = [stoi.get(ch, 0) for ch in text]
            results.append((ids, torch.arange(C, dtype=torch.long)))
        elif tok_family == "single_byte":
            stoi = toks["meta_byte"]["stoi"]
            ids = []
            char_idx = []
            for ch in text:
                t_start = len(ids)
                b_list = ch.encode("utf-8")
                for b in b_list:
                    ids.append(stoi.get(bytes([b]), b))
                char_idx.append(t_start + len(b_list) - 1)
            results.append((ids, torch.tensor(char_idx, dtype=torch.long)))
        elif tok_family == "single_bpe":
            stoi = toks["meta_bpe"]["stoi"]
            sorted_tokens = toks["meta_bpe"].get("char_tokens_sorted")
            if sorted_tokens is None:
                sorted_tokens = sorted([t for t in stoi if isinstance(t, str)], key=len, reverse=True)
            ids = []
            char_idx = [0] * C
            i = 0
            while i < C:
                matched = False
                for token in sorted_tokens:
                    if text.startswith(token, i):
                        tid = stoi.get(token)
                        if tid is not None:
                            t_idx = len(ids)
                            ids.append(tid)
                            for c_idx in range(i, i + len(token)):
                                char_idx[c_idx] = t_idx
                            i += len(token)
                            matched = True
                            break
                if matched:
                    continue
                ch = text[i]
                tid = stoi.get(ch)
                if tid is not None:
                    t_idx = len(ids)
                    ids.append(tid)
                    char_idx[i] = t_idx
                else:
                    b_list = ch.encode("utf-8")
                    for b in b_list:
                        ids.append(stoi.get(bytes([b]), 0))
                    char_idx[i] = len(ids) - 1
                i += 1
            results.append((ids, torch.tensor(char_idx, dtype=torch.long)))
        elif tok_family == "three_hot":
            stois = toks["stois_three_hot"]
            tok = toks["tok_factorizer_coarse"]
            seq = tok.encode_text(text)
            lanes = [[], [], [], []]
            for item in seq:
                ch = item["char"]
                indices = item["indices"]
                b_id = stois[0].get(ch, ch.encode("utf-8")[0] if len(ch) > 0 and ord(ch) > 255 else 0)
                lanes[0].append(b_id)
                for idx, f_idx in enumerate([1, 2, 3], start=1):
                    t_char = tok.token_for(f_idx, indices[f_idx])
                    lanes[idx].append(stois[idx].get(t_char, 0))
            results.append((lanes, torch.arange(C, dtype=torch.long)))
        elif tok_family in ("mc_full", "mc_coarse"):
            is_full = tok_family == "mc_full"
            stois = toks["stois_mc_full"] if is_full else toks["stois_mc_coarse"]
            tok = toks["tok_factorizer_full"] if is_full else toks["tok_factorizer_coarse"]
            seq = tok.encode_text(text)
            n_lanes = len(stois)
            lanes = [[] for _ in range(n_lanes)]
            for item in seq:
                ch = item["char"]
                indices = item["indices"]
                for i in range(min(24, n_lanes - 1)):
                    t_char = tok.token_for(i, indices[i])
                    lanes[i].append(stois[i].get(t_char, 0))
                if n_lanes > 24:
                    b_id = stois[24].get(ch, ch.encode("utf-8")[0] if len(ch) > 0 and ord(ch) > 255 else 0)
                    lanes[24].append(b_id)
            results.append((lanes, torch.arange(C, dtype=torch.long)))
        elif tok_family in ("hybrid_full", "hybrid_coarse"):
            is_full = tok_family == "hybrid_full"
            tok = toks["tok_hybrid_full"] if is_full else toks["tok_hybrid_coarse"]
            if tok.kiwi is None:
                from kiwipiepy import Kiwi
                tok.kiwi = Kiwi()
            tokens = tok.kiwi.tokenize(text)
            char_pos_tags = {}
            for token in tokens:
                tag = token.tag if tok.pos_mode == "full" else POS_TAG_MAP.get(token.tag, token.tag)
                for idx in range(token.start, token.start + token.len):
                    char_pos_tags[idx] = tag
            segments = [m.group(0) for m in tok._SEGMENT_PATTERN.finditer(text)]
            lanes = [[] for _ in range(25)]
            char_idx = [0] * C
            cur_pos = 0
            for seg in segments:
                seg_len = len(seg)
                if tok.is_hangul_syllable(seg[0]):
                    for ch in seg:
                        t_idx = len(lanes[0])
                        raw_tag = char_pos_tags.get(cur_pos, "UNK")
                        tag = raw_tag if raw_tag in tok.pos_tags else "UNK"
                        feats = tok._features(ch)
                        step = [tok.hangul_id]
                        for l_idx, val in enumerate(feats, start=1):
                            step.append(tok.value_to_id[l_idx].get(val, 0))
                        step.append(tok.value_to_id[23].get(tag, 1))
                        for l_idx in range(24):
                            lanes[l_idx].append(step[l_idx])
                        char_id = ((step[1] - 1) * 588 + (step[2] - 1) * 28 + step[3]) + 1
                        lanes[24].append(char_id)
                        char_idx[cur_pos] = t_idx
                        cur_pos += 1
                else:
                    t_start = len(lanes[0])
                    bpe_ids = tok.sp.encode_as_ids(seg)
                    if not bpe_ids:
                        bpe_ids = [0]
                    for b_id in bpe_ids:
                        lanes[0].append(b_id)
                        for l_idx in range(1, 25):
                            lanes[l_idx].append(0)
                    t_end = len(lanes[0])
                    n_bpe = t_end - t_start
                    for idx_in_seg in range(seg_len):
                        c_idx = cur_pos + idx_in_seg
                        s_t = t_start + int(idx_in_seg * n_bpe / seg_len)
                        char_idx[c_idx] = min(s_t, t_end - 1)
                    cur_pos += seg_len
            results.append((lanes, torch.tensor(char_idx, dtype=torch.long)))
    return results


# 17 unique checkpoints per optimizer -> 25 evaluation rows
ADAMW_UNIQUE_CKPTS = [
    ("Baseline Character Tokenizer (V=4,924)", "out_baseline_korean_pos/best_ckpt.pt", "best_ckpt (62,000)", "single_char", "single_char"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "out_baseline_char_bpe/best_ckpt.pt", "best_ckpt (71,000)", "single_bpe", "single_bpe"),
    ("Baseline Pure Byte Tokenizer (V=256)", "out_baseline_pure_byte/best_ckpt.pt", "best_ckpt (106,000)", "single_byte", "single_byte"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "out_three_hot_ind_char_byte/best_ckpt.pt", "best_ckpt (72,000)", "three_hot", "three_hot_ind"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "out_three_hot_cond_char_byte/best_ckpt.pt", "best_ckpt (65,000)", "three_hot", "three_hot_cond"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "out_mc_full_pos_weighted_byte/best_ckpt.pt", "best_ckpt (72,000)", "mc_full", "mc_full_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "out_mc_full_pos_unweighted_byte/best_ckpt.pt", "best_ckpt (65,000)", "mc_full", "mc_full_pos_unweighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "out_mc_coarse_pos_weighted_byte/best_ckpt.pt", "best_ckpt (67,000)", "mc_coarse", "mc_coarse_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "out_mc_coarse_pos_unweighted_byte/best_ckpt.pt", "best_ckpt (57,000)", "mc_coarse", "mc_coarse_pos_unweighted"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt", "best_ckpt (45,000)", "hybrid_full", "nochar_full_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_unweighted/best_ckpt.pt", "best_ckpt (43,000)", "hybrid_full", "nochar_full_pos_unweighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt", "best_ckpt (45,000)", "hybrid_coarse", "nochar_coarse_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_unweighted/best_ckpt.pt", "best_ckpt (45,000)", "hybrid_coarse", "nochar_coarse_pos_unweighted"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)", "out_mc_nochar_full_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34,232)", "hybrid_full", "charhead_full_h11"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34,082)", "hybrid_full", "charhead_full_unweighted"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)", "out_mc_nochar_coarse_pos_charhead_h11/best_ckpt.pt", "best_ckpt (34,232)", "hybrid_coarse", "charhead_coarse_h11"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_charhead_unweighted/best_ckpt.pt", "best_ckpt (34,082)", "hybrid_coarse", "charhead_coarse_unweighted"),
]

MUON_UNIQUE_CKPTS = [
    ("Baseline Character Tokenizer (V=4,924)", "out_baseline_korean_pos_muon/best_ckpt.pt", "best_ckpt (70000)", "single_char", "single_char"),
    ("Baseline Byte-Fallback BPE (V=4,924)", "out_baseline_char_bpe_muon/best_ckpt.pt", "best_ckpt (63000)", "single_bpe", "single_bpe"),
    ("Baseline Pure Byte Tokenizer (V=256)", "out_baseline_pure_byte_muon/best_ckpt.pt", "best_ckpt (111000)", "single_byte", "single_byte"),
    ("Three-Hot Independent Char/Byte (Song 2018)", "out_three_hot_ind_char_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "three_hot", "three_hot_ind"),
    ("Three-Hot Conditional Char/Byte (Cognetta 2023)", "out_three_hot_cond_char_byte_muon/best_ckpt.pt", "best_ckpt (65000)", "three_hot", "three_hot_cond"),
    ("Hangul Factorizer + Char Stream (Full POS, Weighted)", "out_mc_full_pos_weighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_full", "mc_full_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Full POS, Unweighted)", "out_mc_full_pos_unweighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_full", "mc_full_pos_unweighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Weighted)", "out_mc_coarse_pos_weighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_coarse", "mc_coarse_pos_weighted"),
    ("Hangul Factorizer + Char Stream (Coarse POS, Unweighted)", "out_mc_coarse_pos_unweighted_byte_muon/best_ckpt.pt", "best_ckpt (72000)", "mc_coarse", "mc_coarse_pos_unweighted"),
    ("No-Char Hangul Factorizer (Full POS, Newly Weighted C06)", "out_mc_nochar_full_pos_new_weighted_muon/best_ckpt.pt", "best_ckpt (44000)", "hybrid_full", "nochar_full_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_unweighted_muon/best_ckpt.pt", "best_ckpt (41000)", "hybrid_full", "nochar_full_pos_unweighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Newly Weighted C06)", "out_mc_nochar_coarse_pos_new_weighted_muon/best_ckpt.pt", "best_ckpt (35000)", "hybrid_coarse", "nochar_coarse_pos_new_weighted"),
    ("No-Char Hangul Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_unweighted_muon/best_ckpt.pt", "best_ckpt (35000)", "hybrid_coarse", "nochar_coarse_pos_unweighted"),
    ("Output-Only Char Head Factorizer (Full POS, Newly Weighted H11)", "out_mc_nochar_full_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "hybrid_full", "charhead_full_h11"),
    ("Output-Only Char Head Factorizer (Full POS, Unweighted)", "out_mc_nochar_full_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "hybrid_full", "charhead_full_unweighted"),
    ("Output-Only Char Head Factorizer (Coarse POS, Newly Weighted H11)", "out_mc_nochar_coarse_pos_charhead_h11_muon/best_ckpt.pt", "best_ckpt (45542)", "hybrid_coarse", "charhead_coarse_h11"),
    ("Output-Only Char Head Factorizer (Coarse POS, Unweighted)", "out_mc_nochar_coarse_pos_charhead_unweighted_muon/best_ckpt.pt", "best_ckpt (45542)", "hybrid_coarse", "charhead_coarse_unweighted"),
]


class ProbeNER(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_classes: int = 13):
        super().__init__()
        self.proj = nn.Linear(1252, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, syl, pred, pos, surp):
        feat = torch.cat([x, syl, pred, pos * 2.0, surp], dim=-1)
        h = F.gelu(self.norm(self.proj(feat)))
        out, _ = self.lstm(h)
        return self.classifier(out)


class BiaffineDependencyProbe(nn.Module):
    def __init__(self, in_dim: int = 1252, hidden_dim: int = 256, arc_dim: int = 128, rel_dim: int = 64, num_rels: int = 37):
        super().__init__()
        self.root_emb = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.arc_head = nn.Sequential(nn.Linear(hidden_dim, arc_dim), nn.GELU())
        self.arc_dep = nn.Sequential(nn.Linear(hidden_dim, arc_dim), nn.GELU())
        self.W_arc = nn.Parameter(torch.randn(arc_dim, arc_dim) * 0.02)
        self.b_arc = nn.Parameter(torch.zeros(arc_dim))

        self.rel_head = nn.Sequential(nn.Linear(hidden_dim, rel_dim), nn.GELU())
        self.rel_dep = nn.Sequential(nn.Linear(hidden_dim, rel_dim), nn.GELU())
        self.W_rel = nn.Linear(rel_dim * 2, num_rels)
        self.U_rel = nn.Parameter(torch.randn(num_rels, rel_dim, rel_dim) * 0.02)

    def forward(self, w_seq):
        B = w_seq.size(0)
        root = self.root_emb.expand(B, 1, -1)
        x = torch.cat([root, w_seq], dim=1)
        h = F.gelu(self.norm(self.proj(x)))
        r, _ = self.lstm(h)

        h_head = self.arc_head(r)
        h_dep = self.arc_dep(r)
        arc_scores = torch.matmul(h_dep, self.W_arc) @ h_head.transpose(1, 2) + (h_head @ self.b_arc).unsqueeze(1)
        return arc_scores, r

    def score_labels(self, r, heads):
        B, N_plus_1, _ = r.size()
        N = N_plus_1 - 1
        r_dep = self.rel_dep(r[:, 1:, :])
        r_head_all = self.rel_head(r)

        idx = heads.unsqueeze(-1).expand(B, N, r_head_all.size(-1))
        r_head = torch.gather(r_head_all, dim=1, index=idx)

        lin_score = self.W_rel(torch.cat([r_dep, r_head], dim=-1))
        bilin_score = torch.einsum("bnu,cuv,bnv->bnc", r_dep, self.U_rel, r_head)
        return lin_score + bilin_score


@torch.no_grad()
def extract_canonical_features(
    model,
    sentences: List[str],
    tokenized_data: List[Tuple[Any, torch.Tensor]],
    pos_in_oh_list: List[torch.Tensor],
    tok_family: str,
    mtype: str,
    batch_size: int = 64,
    block_size: int = 256,
) -> List[Dict[str, torch.Tensor]]:
    """
    Extracts the canonical frozen encoder representations (ln_f hidden states,
    compositional factor embeddings, and syntactic POS heads) once per checkpoint.
    """
    is_mc = getattr(model.config, "multicontext", False)
    out_list: List[Dict[str, torch.Tensor]] = []

    def get_head_weight(k: int) -> torch.Tensor:
        if f"lm_head_{k}" in model.transformer:
            return model.transformer[f"lm_head_{k}"].weight
        elif "lm_head_fused" in model.transformer:
            splits = torch.split(model.transformer["lm_head_fused"].weight, model.config.vocab_sizes, dim=0)
            return splits[k]
        raise KeyError(f"Head {k} not found")

    chunks_info = []
    for s_idx, (enc, _) in enumerate(tokenized_data):
        seq_len = len(enc) if not is_mc else len(enc[0])
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            c_data = [lane[start:end] for lane in enc] if is_mc else enc[start:end]
            chunks_info.append((s_idx, start, end, c_data))

    chunk_x: Dict[int, List[torch.Tensor]] = {}
    chunk_syl: Dict[int, List[torch.Tensor]] = {}
    chunk_pred: Dict[int, List[torch.Tensor]] = {}
    chunk_pos_prob: Dict[int, List[torch.Tensor]] = {}
    chunk_surp: Dict[int, List[torch.Tensor]] = {}

    for b_start in range(0, len(chunks_info), batch_size):
        batch = chunks_info[b_start : b_start + batch_size]
        max_len = max(item[2] - item[1] for item in batch)
        B = len(batch)

        captured_x = []
        def hook_fn(module, inp, out):
            captured_x.append(out.detach())
        handle = model.transformer.ln_f.register_forward_hook(hook_fn)

        if is_mc:
            n_model_lanes = len(model.config.vocab_sizes)
            token_dict = {}
            for l_i in range(n_model_lanes):
                tensor_l = torch.zeros((B, max_len), dtype=torch.long, device=DEVICE)
                for b_i, (_, _, _, c_data) in enumerate(batch):
                    src = c_data[l_i] if l_i < len(c_data) else c_data[-1]
                    tensor_l[b_i, : len(src)] = torch.tensor(src, dtype=torch.long, device=DEVICE)
                tensor_l = torch.clamp(tensor_l, 0, model.config.vocab_sizes[l_i] - 1)
                token_dict[f"lane_{l_i}"] = tensor_l
            logits_list, _ = model(token_dict=token_dict, target_dict=token_dict)
        else:
            tensor_in = torch.zeros((B, max_len), dtype=torch.long, device=DEVICE)
            for b_i, (_, _, _, c_data) in enumerate(batch):
                tensor_in[b_i, : len(c_data)] = torch.tensor(c_data, dtype=torch.long, device=DEVICE)
            tensor_in = torch.clamp(tensor_in, 0, model.config.vocab_size - 1)
            logits_single, _ = model(tensor_in, tensor_in)

        handle.remove()
        x_batch = captured_x[0]
        if getattr(model, "n_embd_wte", False):
            x_batch = F.linear(x_batch, model.transformer.scale_down.weight.t())
        x_norm = F.layer_norm(x_batch, (384,))

        if not is_mc:
            w_emb = model.transformer.wte(tensor_in)
            if getattr(model, "n_embd_wte", False):
                w_emb = model.transformer.scale_up(w_emb)
                w_emb = F.linear(w_emb, model.transformer.scale_down.weight.t())
            syl_batch = F.layer_norm(w_emb, (384,))

            probs = F.softmax(logits_single, dim=-1)
            e_p = torch.matmul(probs, model.lm_head.weight)
            if getattr(model, "n_embd_wte", False):
                e_p = F.linear(e_p, model.transformer.scale_down.weight.t())
            pred_batch = torch.roll(F.layer_norm(e_p, (384,)), shifts=1, dims=1)

            lp = F.log_softmax(logits_single, dim=-1)
            shifted_lp = torch.roll(lp, shifts=1, dims=1)
            tok_lp = shifted_lp.gather(-1, tensor_in.unsqueeze(-1)).squeeze(-1)
            if tok_family == "single_char":
                unk_mask = (tensor_in == 0).float()
                tok_lp = tok_lp - unk_mask * math.log(8901.0)
            entropy = -(probs * lp).sum(dim=-1)
            surp_batch = torch.stack([-tok_lp / 10.0, torch.roll(entropy, 1, 1) / 10.0, (tensor_in == 0).float(), torch.ones_like(tok_lp)], dim=-1)
            pos_p = torch.zeros((B, max_len, 48), device=DEVICE)

        elif mtype in ("three_hot_ind", "three_hot_cond"):
            w1 = model.transformer["wte_1"](token_dict["lane_1"])
            w2 = model.transformer["wte_2"](token_dict["lane_2"])
            w3 = model.transformer["wte_3"](token_dict["lane_3"])
            w0 = model.transformer["wte_0"](token_dict["lane_0"])
            e_jamo = F.layer_norm(w1 + w2 + w3, (384,))
            e_conj = F.layer_norm(w1 * w2 + w2 * w3, (384,))
            if mtype == "three_hot_cond" and hasattr(model, "cond_W_e"):
                h_c = torch.tanh(model.cond_W_e(w1) + model.cond_W_h(w2))
                e_conj = F.layer_norm(e_conj + h_c, (384,))
            syl_batch = F.layer_norm(e_jamo + 0.5 * e_conj + 0.3 * F.layer_norm(w0, (384,)), (384,))

            p1 = F.softmax(logits_list[1], dim=-1)
            p2 = F.softmax(logits_list[2], dim=-1)
            p3 = F.softmax(logits_list[3], dim=-1)
            e_p = F.layer_norm(
                torch.matmul(p1, get_head_weight(1))
                + torch.matmul(p2, get_head_weight(2))
                + torch.matmul(p3, get_head_weight(3)),
                (384,),
            )
            pred_batch = torch.roll(e_p, shifts=1, dims=1)

            lp1 = F.log_softmax(logits_list[1], dim=-1)
            lp2 = F.log_softmax(logits_list[2], dim=-1)
            lp3 = F.log_softmax(logits_list[3], dim=-1)
            s_lp = (
                torch.roll(lp1, 1, 1).gather(-1, token_dict["lane_1"].unsqueeze(-1)).squeeze(-1)
                + torch.roll(lp2, 1, 1).gather(-1, token_dict["lane_2"].unsqueeze(-1)).squeeze(-1)
                + torch.roll(lp3, 1, 1).gather(-1, token_dict["lane_3"].unsqueeze(-1)).squeeze(-1)
            )
            ent = -((p1 * lp1).sum(-1) + (p2 * lp2).sum(-1) + (p3 * lp3).sum(-1)) / 3.0
            surp_batch = torch.stack([-s_lp / 10.0, torch.roll(ent, 1, 1) / 10.0, torch.zeros_like(s_lp), torch.ones_like(s_lp)], dim=-1)
            pos_p = torch.zeros((B, max_len, 48), device=DEVICE)

        else:
            w1 = model.transformer["wte_1"](token_dict["lane_1"])
            w2 = model.transformer["wte_2"](token_dict["lane_2"])
            w3 = model.transformer["wte_3"](token_dict["lane_3"])
            w_pos = model.transformer["wte_23"](token_dict["lane_23"])
            w_art = sum(model.transformer[f"wte_{k}"](token_dict[f"lane_{k}"]) for k in range(4, 23))

            e_jamo = F.layer_norm(w1 + w2 + w3, (384,))
            e_conj = F.layer_norm(w1 * w2 + w2 * w3 + w1 * w3, (384,))
            e_pos = F.layer_norm(w_pos, (384,))
            e_art = F.layer_norm(w_art, (384,))

            p_pos_raw = F.softmax(logits_list[23], dim=-1)
            p_pos_shifted = torch.roll(p_pos_raw, shifts=1, dims=1)
            p_pos_shifted[:, 0, :] = p_pos_raw[:, 0, :]
            pos_p = F.pad(p_pos_shifted, (0, max(0, 48 - p_pos_shifted.size(-1))))[:, :, :48]

            p1 = F.softmax(logits_list[1], dim=-1)
            p2 = F.softmax(logits_list[2], dim=-1)
            p3 = F.softmax(logits_list[3], dim=-1)
            e_jamo_pred = F.layer_norm(
                torch.matmul(p1, get_head_weight(1))
                + torch.matmul(p2, get_head_weight(2))
                + torch.matmul(p3, get_head_weight(3))
                + torch.matmul(p_pos_raw, get_head_weight(23)),
                (384,),
            )

            lp1 = F.log_softmax(logits_list[1], dim=-1)
            lp2 = F.log_softmax(logits_list[2], dim=-1)
            lp3 = F.log_softmax(logits_list[3], dim=-1)
            jamo_lp = (
                torch.roll(lp1, 1, 1).gather(-1, token_dict["lane_1"].unsqueeze(-1)).squeeze(-1)
                + torch.roll(lp2, 1, 1).gather(-1, token_dict["lane_2"].unsqueeze(-1)).squeeze(-1)
                + torch.roll(lp3, 1, 1).gather(-1, token_dict["lane_3"].unsqueeze(-1)).squeeze(-1)
            )
            ent_jamo = -((p1 * lp1).sum(-1) + (p2 * lp2).sum(-1) + (p3 * lp3).sum(-1)) / 3.0

            if mtype.startswith("mc_"):
                w24 = F.layer_norm(model.transformer["wte_24"](token_dict["lane_24"]), (384,))
                syl_batch = F.layer_norm(e_jamo + 0.6 * e_conj + 0.8 * e_pos + 0.3 * e_art + 0.5 * w24, (384,))
                pred_batch = torch.roll(e_jamo_pred, shifts=1, dims=1)
                surp_batch = torch.stack([-jamo_lp / 10.0, torch.roll(ent_jamo, 1, 1) / 10.0, torch.zeros_like(jamo_lp), torch.ones_like(jamo_lp)], dim=-1)

            elif mtype.startswith("nochar_"):
                w0 = F.layer_norm(model.transformer["wte_0"](token_dict["lane_0"]), (384,))
                is_non_hg = (token_dict["lane_0"] != 2).float().unsqueeze(-1)
                syl_batch = F.layer_norm(
                    (1.0 - 0.5 * is_non_hg) * (e_jamo + 0.6 * e_conj + 0.8 * e_pos + 0.3 * e_art) + (0.3 + 0.7 * is_non_hg) * w0,
                    (384,),
                )
                pred_batch = torch.roll(e_jamo_pred, shifts=1, dims=1)
                surp_batch = torch.stack([-jamo_lp / 10.0, torch.roll(ent_jamo, 1, 1) / 10.0, torch.zeros_like(jamo_lp), torch.ones_like(jamo_lp)], dim=-1)

            elif mtype.startswith("charhead_"):
                w0 = F.layer_norm(model.transformer["wte_0"](token_dict["lane_0"]), (384,))
                is_non_hg = (token_dict["lane_0"] != 2).float().unsqueeze(-1)
                head24_w = get_head_weight(24)
                syl_ids = token_dict["lane_24"].clamp(0, head24_w.size(0) - 1)
                e_char24 = F.layer_norm(F.embedding(syl_ids, head24_w), (384,))

                # Canonical frozen representation combining input factor embeddings + output head projection
                syl_batch = F.layer_norm(
                    (1.0 - 0.5 * is_non_hg) * (e_jamo + 0.6 * e_conj + 0.8 * e_pos + 0.3 * e_art + 0.9 * e_char24) + (0.3 + 0.7 * is_non_hg) * w0,
                    (384,),
                )
                p24 = F.softmax(logits_list[24], dim=-1)
                e_pred24 = F.layer_norm(torch.matmul(p24, head24_w) + e_jamo_pred, (384,))
                pred_batch = torch.roll(e_pred24, shifts=1, dims=1)
                lp24 = F.log_softmax(logits_list[24], dim=-1)
                char24_lp = torch.roll(lp24, 1, 1).gather(-1, syl_ids.unsqueeze(-1)).squeeze(-1)
                ent24 = -(p24 * lp24).sum(-1)
                surp_batch = torch.stack([-char24_lp / 10.0, torch.roll(ent24, 1, 1) / 10.0, torch.zeros_like(char24_lp), torch.ones_like(char24_lp)], dim=-1)

        for b_i, (s_idx, start, end, _) in enumerate(batch):
            c_len = end - start
            chunk_x.setdefault(s_idx, []).append(x_norm[b_i, :c_len])
            chunk_pos_prob.setdefault(s_idx, []).append(pos_p[b_i, :c_len])
            chunk_syl.setdefault(s_idx, []).append(syl_batch[b_i, :c_len])
            chunk_pred.setdefault(s_idx, []).append(pred_batch[b_i, :c_len])
            chunk_surp.setdefault(s_idx, []).append(surp_batch[b_i, :c_len])

    for s_idx, (text, (enc, char_idx_tensor)) in enumerate(zip(sentences, tokenized_data)):
        c_idx_gpu = char_idx_tensor.to(DEVICE)
        full_x = torch.cat(chunk_x[s_idx], dim=0)
        full_pos_p = torch.cat(chunk_pos_prob[s_idx], dim=0)
        c_idx_clamped = torch.clamp(c_idx_gpu, 0, full_x.size(0) - 1)

        char_x = full_x[c_idx_clamped].cpu()
        char_pos_p = full_pos_p[c_idx_clamped].cpu()

        if is_mc and mtype not in ("three_hot_ind", "three_hot_cond"):
            pos_in_oh = pos_in_oh_list[s_idx]
        else:
            pos_in_oh = torch.zeros((len(text), 48), dtype=torch.float32)

        pos_full_96 = torch.cat([pos_in_oh, char_pos_p], dim=-1)

        full_syl = torch.cat(chunk_syl[s_idx], dim=0)
        full_pred = torch.cat(chunk_pred[s_idx], dim=0)
        full_surp = torch.cat(chunk_surp[s_idx], dim=0)

        char_syl = full_syl[c_idx_clamped].clone()
        if tok_family in ("hybrid_full", "hybrid_coarse"):
            tok_h = toks["tok_hybrid_full"] if tok_family == "hybrid_full" else toks["tok_hybrid_coarse"]
            for c_i, ch in enumerate(text):
                if not (0xAC00 <= ord(ch) <= 0xD7A3):
                    b_ids = tok_h.sp.encode_as_ids(ch)
                    b_id = b_ids[0] if b_ids else 0
                    w_ch = F.layer_norm(model.transformer["wte_0"](torch.tensor(b_id, device=DEVICE)), (384,))
                    char_syl[c_i] = w_ch

        out_list.append({
            "x": char_x,
            "syl": char_syl.cpu(),
            "pred": full_pred[c_idx_clamped].cpu(),
            "pos": pos_full_96,
            "surp": full_surp[c_idx_clamped].cpu(),
        })

    return out_list


def train_eval_ner_multiseed(
    train_items,
    train_labels,
    val_items,
    val_labels,
    seeds: List[int] = [42, 43, 44],
    epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[float, float]:
    def make_batches(items, labels, b_size):
        batches = []
        for i in range(0, len(items), b_size):
            b_it = items[i : i + b_size]
            b_l = labels[i : i + b_size] if labels is not None else [None] * len(b_it)
            max_l = max(it["x"].size(0) for it in b_it)
            B = len(b_it)
            x_t = torch.zeros((B, max_l, 384), device=DEVICE)
            sy_t = torch.zeros((B, max_l, 384), device=DEVICE)
            pr_t = torch.zeros((B, max_l, 384), device=DEVICE)
            po_t = torch.zeros((B, max_l, 96), device=DEVICE)
            su_t = torch.zeros((B, max_l, 4), device=DEVICE)
            y_t = torch.full((B, max_l), -100, dtype=torch.long, device=DEVICE)
            lengths = []
            for j, (it, l) in enumerate(zip(b_it, b_l)):
                L = it["x"].size(0)
                lengths.append(L)
                x_t[j, :L] = it["x"].to(DEVICE)
                sy_t[j, :L] = it["syl"].to(DEVICE)
                pr_t[j, :L] = it["pred"].to(DEVICE)
                po_t[j, :L] = it["pos"].to(DEVICE)
                su_t[j, :L] = it["surp"].to(DEVICE)
                if l is not None:
                    y_t[j, :len(l)] = torch.tensor(l[:L], dtype=torch.long, device=DEVICE)
            batches.append((x_t, sy_t, pr_t, po_t, su_t, y_t, lengths))
        return batches

    train_batches = make_batches(train_items, train_labels, batch_size)
    val_batches = make_batches(val_items, None, 64)

    ent_scores, char_scores = [], []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        probe = ProbeNER().to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_batches))

        probe.train()
        for ep in range(epochs):
            g = torch.Generator()
            g.manual_seed(seed * 100 + ep)
            perm = torch.randperm(len(train_batches), generator=g).tolist()
            for idx in perm:
                x_t, sy_t, pr_t, po_t, su_t, y_t, _ = train_batches[idx]
                optimizer.zero_grad()
                logits = probe(x_t, sy_t, pr_t, po_t, su_t)
                loss = F.cross_entropy(logits.view(-1, 13), y_t.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        probe.eval()
        pred_seqs = []
        with torch.no_grad():
            for x_t, sy_t, pr_t, po_t, su_t, _, lengths in val_batches:
                logits = probe(x_t, sy_t, pr_t, po_t, su_t)
                preds = logits.argmax(dim=-1).cpu().tolist()
                for pred_row, L in zip(preds, lengths):
                    pred_seqs.append(pred_row[:L])

        e_f1, c_f1 = compute_klue_ner_metrics(val_labels, pred_seqs)
        ent_scores.append(e_f1)
        char_scores.append(c_f1)

    return float(np.mean(ent_scores)), float(np.mean(char_scores))


def pool_word_features(char_feats: List[Dict[str, torch.Tensor]], spans_list: List[List[Tuple[int, int]]]) -> List[torch.Tensor]:
    out = []
    for it, spans in zip(char_feats, spans_list):
        c_vec = torch.cat([it["x"], it["syl"], it["pred"], it["pos"] * 2.0, it["surp"]], dim=-1)
        C = c_vec.size(0)
        w_vecs = []
        for s, e in spans:
            s_c = min(max(0, s), C - 1)
            e_c = min(max(s_c + 1, e), C)
            f_last = c_vec[e_c - 1]
            f_first = c_vec[s_c]
            f_mean = c_vec[s_c:e_c].mean(dim=0)
            w_vecs.append(0.5 * f_last + 0.3 * f_mean + 0.2 * f_first)
        out.append(torch.stack(w_vecs, dim=0))
    return out


def train_eval_dp_multiseed(
    train_w,
    train_heads,
    train_rels,
    val_w,
    val_heads,
    val_rels,
    num_deprels: int,
    seeds: List[int] = [42, 43, 44],
    epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[float, float]:
    def make_batches(w_list, h_list, r_list, b_size):
        batches = []
        for i in range(0, len(w_list), b_size):
            bw = w_list[i : i + b_size]
            bh = h_list[i : i + b_size]
            br = r_list[i : i + b_size]
            max_n = max(w.size(0) for w in bw)
            B = len(bw)
            w_t = torch.zeros((B, max_n, 1252), device=DEVICE)
            h_t = torch.full((B, max_n), -100, dtype=torch.long, device=DEVICE)
            r_t = torch.full((B, max_n), -100, dtype=torch.long, device=DEVICE)
            mask_t = torch.zeros((B, max_n + 1), dtype=torch.bool, device=DEVICE)
            mask_t[:, 0] = True
            lengths = []
            for j, (w, h, rel) in enumerate(zip(bw, bh, br)):
                N = w.size(0)
                lengths.append(N)
                w_t[j, :N] = w.to(DEVICE)
                h_t[j, :N] = torch.tensor(h, dtype=torch.long, device=DEVICE)
                r_t[j, :N] = torch.tensor(rel, dtype=torch.long, device=DEVICE)
                mask_t[j, 1 : N + 1] = True
            batches.append((w_t, h_t, r_t, mask_t, lengths, bh, br))
        return batches

    train_batches = make_batches(train_w, train_heads, train_rels, batch_size)
    val_batches = make_batches(val_w, val_heads, val_rels, 64)
    uas_scores, las_scores = [], []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        probe = BiaffineDependencyProbe(num_rels=num_deprels).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_batches))

        probe.train()
        for ep in range(epochs):
            g = torch.Generator()
            g.manual_seed(seed * 100 + ep)
            perm = torch.randperm(len(train_batches), generator=g).tolist()
            for idx in perm:
                w_t, h_t, r_t, mask_t, _, _, _ = train_batches[idx]
                optimizer.zero_grad()
                arc_scores, r_state = probe(w_t)
                dep_arc_scores = arc_scores[:, 1:, :]
                B, N, N_p1 = dep_arc_scores.size()
                head_mask = mask_t.unsqueeze(1).expand(B, N, N_p1).clone()
                diag_mask = torch.zeros((B, N, N_p1), dtype=torch.bool, device=DEVICE)
                for d_i in range(N):
                    diag_mask[:, d_i, d_i + 1] = True
                dep_arc_scores = dep_arc_scores.masked_fill(~head_mask | diag_mask, -1e9)

                loss_arc = F.cross_entropy(dep_arc_scores.reshape(-1, N_p1), h_t.reshape(-1), ignore_index=-100)
                gold_heads_clamped = h_t.clamp(min=0)
                rel_scores = probe.score_labels(r_state, gold_heads_clamped)
                loss_rel = F.cross_entropy(rel_scores.reshape(-1, num_deprels), r_t.reshape(-1), ignore_index=-100)

                loss = loss_arc + loss_rel
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        probe.eval()
        total_words = 0
        correct_uas = 0
        correct_las = 0
        with torch.no_grad():
            for w_t, _, _, mask_t, lengths, bh, br in val_batches:
                arc_scores, r_state = probe(w_t)
                dep_arc_scores = arc_scores[:, 1:, :]
                B, N_max, N_p1 = dep_arc_scores.size()
                head_mask = mask_t.unsqueeze(1).expand(B, N_max, N_p1).clone()
                diag_mask = torch.zeros((B, N_max, N_p1), dtype=torch.bool, device=DEVICE)
                for d_i in range(N_max):
                    diag_mask[:, d_i, d_i + 1] = True
                dep_arc_scores = dep_arc_scores.masked_fill(~head_mask | diag_mask, -1e9)
                pred_heads = dep_arc_scores.argmax(dim=-1)
                rel_scores = probe.score_labels(r_state, pred_heads)
                pred_rels = rel_scores.argmax(dim=-1)

                pred_h_batch = pred_heads.cpu().tolist()
                pred_r_batch = pred_rels.cpu().tolist()
                for b_i, N in enumerate(lengths):
                    for ph, gh, pr, gr in zip(pred_h_batch[b_i][:N], bh[b_i], pred_r_batch[b_i][:N], br[b_i]):
                        total_words += 1
                        if ph == gh:
                            correct_uas += 1
                            if pr == gr:
                                correct_las += 1

        uas_scores.append((correct_uas / total_words) * 100.0 if total_words > 0 else 0.0)
        las_scores.append((correct_las / total_words) * 100.0 if total_words > 0 else 0.0)

    return float(np.mean(uas_scores)), float(np.mean(las_scores))


def main():
    t0 = time.time()
    cache_path = "/tmp/klue_probing_cache_3000_1000.pt"
    if os.path.exists(cache_path):
        print(f"Loading cached pre-tokenized datasets from {cache_path}...")
        cached = torch.load(cache_path, weights_only=False)
        ner_tr_sents = cached["ner_tr_sents"]
        ner_tr_tags = cached["ner_tr_tags"]
        ner_va_sents = cached["ner_va_sents"]
        ner_va_tags = cached["ner_va_tags"]
        dp_tr_sents = cached["dp_tr_sents"]
        dp_tr_spans = cached["dp_tr_spans"]
        dp_tr_heads = cached["dp_tr_heads"]
        dp_tr_rels = cached["dp_tr_rels"]
        dp_va_sents = cached["dp_va_sents"]
        dp_va_spans = cached["dp_va_spans"]
        dp_va_heads = cached["dp_va_heads"]
        dp_va_rels = cached["dp_va_rels"]
        num_deprels = cached["num_deprels"]
        cache_ner_tr = cached["cache_ner_tr"]
        cache_ner_va = cached["cache_ner_va"]
        cache_dp_tr = cached["cache_dp_tr"]
        cache_dp_va = cached["cache_dp_va"]
        pos_oh_ner_tr = cached["pos_oh_ner_tr"]
        pos_oh_ner_va = cached["pos_oh_ner_va"]
        pos_oh_dp_tr = cached["pos_oh_dp_tr"]
        pos_oh_dp_va = cached["pos_oh_dp_va"]
        print(f"Loaded cache in {time.time() - t0:.1f}s.")
    else:
        print("Loading KLUE-NER and KLUE-DP datasets (N_train=3,000, N_val=1,000)...")
        ds_ner_tr = load_dataset("klue", "ner", split="train").select(range(3000))
        ds_ner_va = load_dataset("klue", "ner", split="validation").select(range(1000))

        ner_tr_sents = ["".join(r["tokens"]) for r in ds_ner_tr]
        ner_tr_tags = [r["ner_tags"] for r in ds_ner_tr]
        ner_va_sents = ["".join(r["tokens"]) for r in ds_ner_va]
        ner_va_tags = [r["ner_tags"] for r in ds_ner_va]

        ds_dp_tr = load_dataset("klue", "dp", split="train").select(range(3000))
        ds_dp_va = load_dataset("klue", "dp", split="validation").select(range(1000))

        all_deprels = sorted(list({l for r in ds_dp_tr for l in r["deprel"]} | {l for r in ds_dp_va for l in r["deprel"]}))
        deprel_to_id = {l: i for i, l in enumerate(all_deprels)}
        num_deprels = len(all_deprels)

        def prepare_dp(ds):
            sents, spans_list, heads, rels = [], [], [], []
            for r in ds:
                words = r["word_form"]
                text = " ".join(words)
                spans = []
                cur = 0
                for w in words:
                    spans.append((cur, cur + len(w)))
                    cur += len(w) + 1
                sents.append(text)
                spans_list.append(spans)
                heads.append([int(h) for h in r["head"]])
                rels.append([deprel_to_id[l] for l in r["deprel"]])
            return sents, spans_list, heads, rels

        dp_tr_sents, dp_tr_spans, dp_tr_heads, dp_tr_rels = prepare_dp(ds_dp_tr)
        dp_va_sents, dp_va_spans, dp_va_heads, dp_va_rels = prepare_dp(ds_dp_va)

        print("Pre-tokenizing datasets across all 8 tokenizer families and pre-extracting Kiwi POS tags...")
        tok_families = ["single_char", "single_bpe", "single_byte", "three_hot", "mc_full", "mc_coarse", "hybrid_full", "hybrid_coarse"]
        cache_ner_tr = {}
        cache_ner_va = {}
        cache_dp_tr = {}
        cache_dp_va = {}

        for tfam in tok_families:
            cache_ner_tr[tfam] = pretokenize_sentences(ner_tr_sents, tfam)
            cache_ner_va[tfam] = pretokenize_sentences(ner_va_sents, tfam)
            cache_dp_tr[tfam] = pretokenize_sentences(dp_tr_sents, tfam)
            cache_dp_va[tfam] = pretokenize_sentences(dp_va_sents, tfam)

        def build_pos_oh(sents: List[str], is_full: bool) -> List[torch.Tensor]:
            ptok = toks["tok_factorizer_full"] if is_full else toks["tok_factorizer_coarse"]
            pstoi = toks["stois_mc_full"][23] if is_full else toks["stois_mc_coarse"][23]
            out = []
            for text in sents:
                C = len(text)
                oh = torch.zeros((C, 48), dtype=torch.float32)
                seq_pos = ptok.encode_text(text)
                for c_i in range(min(C, len(seq_pos))):
                    t_char = ptok.token_for(23, seq_pos[c_i]["indices"][23])
                    pid = pstoi.get(t_char, 0)
                    if 0 <= pid < 48:
                        oh[c_i, pid] = 1.0
                out.append(oh)
            return out

        pos_oh_ner_tr = {"full": build_pos_oh(ner_tr_sents, True), "coarse": build_pos_oh(ner_tr_sents, False)}
        pos_oh_ner_va = {"full": build_pos_oh(ner_va_sents, True), "coarse": build_pos_oh(ner_va_sents, False)}
        pos_oh_dp_tr = {"full": build_pos_oh(dp_tr_sents, True), "coarse": build_pos_oh(dp_tr_sents, False)}
        pos_oh_dp_va = {"full": build_pos_oh(dp_va_sents, True), "coarse": build_pos_oh(dp_va_sents, False)}

        torch.save({
            "ner_tr_sents": ner_tr_sents,
            "ner_tr_tags": ner_tr_tags,
            "ner_va_sents": ner_va_sents,
            "ner_va_tags": ner_va_tags,
            "dp_tr_sents": dp_tr_sents,
            "dp_tr_spans": dp_tr_spans,
            "dp_tr_heads": dp_tr_heads,
            "dp_tr_rels": dp_tr_rels,
            "dp_va_sents": dp_va_sents,
            "dp_va_spans": dp_va_spans,
            "dp_va_heads": dp_va_heads,
            "dp_va_rels": dp_va_rels,
            "num_deprels": num_deprels,
            "cache_ner_tr": cache_ner_tr,
            "cache_ner_va": cache_ner_va,
            "cache_dp_tr": cache_dp_tr,
            "cache_dp_va": cache_dp_va,
            "pos_oh_ner_tr": pos_oh_ner_tr,
            "pos_oh_ner_va": pos_oh_ner_va,
            "pos_oh_dp_tr": pos_oh_dp_tr,
            "pos_oh_dp_va": pos_oh_dp_va,
        }, cache_path)
        print(f"Pre-processing completed and cached in {time.time() - t0:.1f}s.")

    def evaluate_suite(suite_name: str, ckpt_list: List[Tuple[str, str, str, str, str]], out_json_path: str):
        print(f"\n=======================================================")
        print(f" Evaluating Suite: {suite_name}")
        print(f"=======================================================")
        results_rows = []

        for base_name, ckpt_path, ckpt_label, tfam, mtype in ckpt_list:
            t_m = time.time()
            model, _ = load_model(ckpt_path)
            p_key = "full" if "full" in mtype else "coarse"

            ner_tr_feats = extract_canonical_features(model, ner_tr_sents, cache_ner_tr[tfam], pos_oh_ner_tr[p_key], tfam, mtype)
            ner_va_feats = extract_canonical_features(model, ner_va_sents, cache_ner_va[tfam], pos_oh_ner_va[p_key], tfam, mtype)

            dp_tr_feats = extract_canonical_features(model, dp_tr_sents, cache_dp_tr[tfam], pos_oh_dp_tr[p_key], tfam, mtype)
            dp_va_feats = extract_canonical_features(model, dp_va_sents, cache_dp_va[tfam], pos_oh_dp_va[p_key], tfam, mtype)

            ent_f1, char_f1 = train_eval_ner_multiseed(ner_tr_feats, ner_tr_tags, ner_va_feats, ner_va_tags)

            dp_tr_w = pool_word_features(dp_tr_feats, dp_tr_spans)
            dp_va_w = pool_word_features(dp_va_feats, dp_va_spans)
            uas, las = train_eval_dp_multiseed(dp_tr_w, dp_tr_heads, dp_tr_rels, dp_va_w, dp_va_heads, dp_va_rels, num_deprels)

            modes = ["with", "without", "joint"] if mtype.startswith("charhead_") else ["default"]
            mode_suffixes = {
                "default": "",
                "with": " (WITH `lm_head_24`)",
                "without": " (WITHOUT `lm_head_24`, Greedy)",
                "joint": " (WITHOUT `lm_head_24`, Joint Beam Search)",
            }

            for mode in modes:
                row_name = base_name + mode_suffixes[mode]
                row_data = {
                    "model_name": row_name,
                    "ckpt_path": ckpt_path,
                    "ckpt_label": ckpt_label,
                    "mode": mode,
                    "ner_entity_f1": round(ent_f1, 2),
                    "ner_char_f1": round(char_f1, 2),
                    "dp_uas": round(uas, 2),
                    "dp_las": round(las, 2),
                }
                results_rows.append(row_data)
                print(
                    f"  {row_name:78s} | NER Ent F1: {ent_f1:5.2f}% | Char F1: {char_f1:5.2f}% | "
                    f"DP UAS: {uas:5.2f}% | LAS: {las:5.2f}% ({time.time() - t_m:.1f}s)"
                )

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results_rows, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results_rows)} evaluation rows to {out_json_path}")

    evaluate_suite(
        "AdamW Best Validation Loss Checkpoints",
        ADAMW_UNIQUE_CKPTS,
        os.path.join(REPO_ROOT, "results_klue_ner_dp_adamw_val_loss.json"),
    )
    evaluate_suite(
        "Muon Best Validation Loss Checkpoints",
        MUON_UNIQUE_CKPTS,
        os.path.join(REPO_ROOT, "results_klue_ner_dp_muon_val_loss.json"),
    )


if __name__ == "__main__":
    main()
