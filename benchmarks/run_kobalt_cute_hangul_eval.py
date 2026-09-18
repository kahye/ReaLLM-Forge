#!/usr/bin/env python3
"""KoBALT & CUTE-Hangul Diagnostic Benchmark Suite for Hangul Factorizers.

Evaluates 5 literature-grounded (2023-2025) Korean sub-syllable & linguistic tasks:
1. Choseong Quiz & Sub-Syllable Probing (CUTE 2024 / CharBench 2025)
2. Allomorphic Particle Minimal Pairs (KoBALT 2025 Morphology - 은/는, 이/가, 을/를, 로/으로)
3. Korean Phonological Assimilation Rules (KoBALT 2025 Phonology - 비음화, 유음화, 구개음화, 격음화)
4. Korean Vowel Harmony & Ideophones (KoBALT 2025 Sound Symbolism - 양성/음성 모음조화)
5. CUTE-Hangul Jamo Surgery & Orthographic Editing (CUTE 2024 - Jamo substitution/deletion)
"""
import argparse
import json
import math
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions

sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))
from hangul_factorizer import HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer


# ==============================================================================
# 1. Curated Diagnostic Datasets (KoBALT 2025 & CUTE-Hangul 2024)
# ==============================================================================

TASK_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    "choseong_quiz": [
        {
            "ctx": "초성 퀴즈: 대한민국의 수도 (초성: ㅅㅇ) 정답:",
            "candidates": [" 서울", " 부산", " 수원", " 세종"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 겨울에 하늘에서 내리는 하얀 결정 (초성: ㄴ) 정답:",
            "candidates": [" 눈", " 비", " 바람", " 안개"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 사과, 배, 포도 등을 통틀어 이르는 말 (초성: ㄱㅇ) 정답:",
            "candidates": [" 과일", " 채소", " 곡식", " 나무"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 밤하늘에 밝게 빛나는 천체 (초성: ㄷ) 정답:",
            "candidates": [" 달", " 해", " 구름", " 은하"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 한글을 창제하신 조선의 제4대 왕 (초성: ㅅㅈㄷㅇ) 정답:",
            "candidates": [" 세종대왕", " 이순신", " 태조이성계", " 정조대왕"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 학교에서 학생들을 가르치는 직업 (초성: ㅅㅅㄴ) 정답:",
            "candidates": [" 선생님", " 의사", " 경찰관", " 소방관"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 아침, 점심, 저녁에 먹는 음식 (초성: ㅅㅅ) 정답:",
            "candidates": [" 식사", " 간식", " 음료", " 요리"],
            "answer": 0,
        },
        {
            "ctx": "초성 퀴즈: 컴퓨터와 인터넷을 통해 정보를 주고받는 공간 (초성: ㅇㄹㅇ) 정답:",
            "candidates": [" 온라인", " 도서관", " 우체국", " 방송국"],
            "answer": 0,
        },
    ],
    "allomorphic_particles": [
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 책이 책상 위에 있다.", " 책가 책상 위에 있다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 사과를 맛있게 먹었다.", " 사과을 맛있게 먹었다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오 (ㄹ 받침 규칙):",
            "candidates": [" 서울로 가는 기차를 탔다.", " 서울으로 가는 기차를 탔다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 부산으로 가는 버스를 탔다.", " 부산로 가는 버스를 탔다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 하늘은 높고 푸르다.", " 하늘는 높고 푸르다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 바다는 넓고 깊다.", " 바다은 넓고 깊다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 친구와 함께 영화관에 갔다.", " 친구과 함께 영화관에 갔다."],
            "answer": 0,
        },
        {
            "ctx": "문법이 올바른 문장을 고르시오:",
            "candidates": [" 가족과 함께 여행을 떠났다.", " 가족와 함께 여행을 떠났다."],
            "answer": 0,
        },
    ],
    "phonological_assimilation": [
        {
            "ctx": "음운 변동 규칙 (비음화): '국물'의 올바른 표준 발음은?",
            "candidates": [" [궁물]", " [국물]", " [군물]", " [구물]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (유음화): '신라'의 올바른 표준 발음은?",
            "candidates": [" [실라]", " [신라]", " [시라]", " [신나]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (구개음화): '같이'의 올바른 표준 발음은?",
            "candidates": [" [가치]", " [가티]", " [바치]", " [갈치]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (격음화): '좋다'의 올바른 표준 발음은?",
            "candidates": [" [조타]", " [조다]", " [졷타]", " [조차]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (비음화): '독립'의 올바른 표준 발음은?",
            "candidates": [" [동닙]", " [독립]", " [독닙]", " [동립]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (구개음화): '해돋이'의 올바른 표준 발음은?",
            "candidates": [" [해도지]", " [해도디]", " [해돋이]", " [해도체]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (격음화): '축하'의 올바른 표준 발음은?",
            "candidates": [" [추카]", " [축하]", " [추가]", " [추하]"],
            "answer": 0,
        },
        {
            "ctx": "음운 변동 규칙 (유음화): '칼날'의 올바른 표준 발음은?",
            "candidates": [" [칼랄]", " [칼날]", " [칸날]", " [카랄]"],
            "answer": 0,
        },
    ],
    "vowel_harmony_ideophones": [
        {
            "ctx": "모음조화 비례식: '퐁당퐁당'이 '풍덩풍덩'이 될 때, '반짝반짝'에 대응하는 음성모음 의태어는?",
            "candidates": [" 번쩍번쩍", " 빈찍빈찍", " 분쩍분쩍", " 본짝본짝"],
            "answer": 0,
        },
        {
            "ctx": "모음조화 비례식: '찰랑찰랑'이 '출렁출렁'이 될 때, '알록달록'에 대응하는 음성모음 의태어는?",
            "candidates": [" 얼룩덜룩", " 일룩딜룩", " 울룩둘룩", " 올록돌록"],
            "answer": 0,
        },
        {
            "ctx": "모음조화 비례식: '살랑살랑'이 '설렁설렁'이 될 때, '깡충깡충'에 대응하는 양성모음 의태어는?",
            "candidates": [" 껑충껑충", " 낑충낑충", " 꿍충꿍충", " 꽁충꽁충"],
            "answer": 3,
        },
        {
            "ctx": "자음 세기 변화: '감감하다 -> 캄캄하다'일 때, '단단하다'보다 더 강하고 단단한 느낌을 주는 된소리 단어는?",
            "candidates": [" 딴딴하다", " 탄탄하다", " 난난하다", " 잔잔하다"],
            "answer": 0,
        },
        {
            "ctx": "자음 세기 변화: '빙글빙글'의 거센소리(격음) 형태는?",
            "candidates": [" 핑글핑글", " 삥글삥글", " 밍글밍글", " 싱글싱글"],
            "answer": 0,
        },
        {
            "ctx": "모음조화: '졸졸' 흐르는 시냇물보다 더 크고 깊게 흐르는 물을 나타내는 음성모음 의성어는?",
            "candidates": [" 줄줄", " 잘잘", " 질질", " 절절"],
            "answer": 0,
        },
    ],
    "cute_jamo_manipulation": [
        {
            "ctx": "자모 연산: '가다'의 첫 글자에 받침 'ㄹ'을 넣으면 어떤 단어가 됩니까?",
            "candidates": [" 갈다", " 간다", " 감다", " 갑다"],
            "answer": 0,
        },
        {
            "ctx": "자모 연산: '살'의 모음 'ㅏ'를 'ㅗ'로 바꾸면 어떤 글자가 됩니까?",
            "candidates": [" 솔", " 술", " 실", " 설"],
            "answer": 0,
        },
        {
            "ctx": "자모 연산: '강산'의 두 글자에서 받침(종성)을 모두 제거하면?",
            "candidates": [" 가사", " 간사", " 강사", " 가산"],
            "answer": 0,
        },
        {
            "ctx": "자모 연산: '구름'의 첫 글자 초성을 'ㅂ'으로 바꾸면?",
            "candidates": [" 부름", " 주름", " 수름", " 무름"],
            "answer": 0,
        },
        {
            "ctx": "자모 연산: '바다'의 두 글자 순서를 거꾸로 뒤집으면?",
            "candidates": [" 다바", " 바다", " 다다", " 바바"],
            "answer": 0,
        },
        {
            "ctx": "자모 연산: '달'의 받침 'ㄹ'을 'ㄱ'으로 바꾸면?",
            "candidates": [" 닭", " 닥", " 담", " 답"],
            "answer": 1,
        },
    ],
}


# ==============================================================================
# 2. Model Loading & Scoring Infrastructure
# ==============================================================================

DEFAULT_MODELS = {
    "baseline_character": {
        "name": "Baseline Character (V=4,924)",
        "path": "out_baseline_korean_pos/best_ckpt.pt",
        "meta": "out_baseline_korean_pos/meta.pkl",
        "type": "single",
    },
    "baseline_char_bpe": {
        "name": "Baseline BPE (V=4,924)",
        "path": "out_baseline_char_bpe/best_ckpt.pt",
        "meta": "out_baseline_char_bpe/meta.pkl",
        "type": "single",
    },
    "baseline_pure_byte": {
        "name": "Baseline Pure Byte (V=256)",
        "path": "out_baseline_pure_byte/best_ckpt.pt",
        "meta": "out_baseline_pure_byte/meta.pkl",
        "type": "single",
    },
    "mc_25_full_weighted": {
        "name": "25-Lane Factorizer + Char Stream (Full POS, Weighted)",
        "path": "out_mc_full_pos_weighted_byte/best_ckpt.pt",
        "type": "mc_25",
    },
    "mc_24_nochar_full_weighted": {
        "name": "24-Lane No-Char (Full POS, Old Weights: wbpe=1.0, wfactors=0.05, wpos=0.5)",
        "path": "out_mc_nochar_full_pos_weighted/best_ckpt.pt",
        "type": "mc_24",
    },
    "mc_24_nochar_full_new_weighted": {
        "name": "24-Lane No-Char (Full POS, Winning Weights: wbpe=1.0, wjamo=1.0, waux=0.05, wpos=0.5)",
        "path": "out_mc_nochar_full_pos_new_weighted/best_ckpt.pt",
        "type": "mc_24",
    },
    "mc_24_nochar_coarse_new_weighted": {
        "name": "24-Lane No-Char (Coarse POS, Winning Weights: wbpe=1.0, wjamo=1.0, waux=0.05, wpos=0.5)",
        "path": "out_mc_nochar_coarse_pos_new_weighted/best_ckpt.pt",
        "type": "mc_24",
    },
}


def load_gpt_model(ckpt_path: str, device: str) -> Tuple[GPT, dict]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    model_args = checkpoint.get("model_args", {})
    allowed_keys = set(GPTConfig.__dataclass_fields__.keys())
    model_args = {k: v for k, v in model_args.items() if k in allowed_keys}
    model_args["dropout"] = 0.0
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, checkpoint_config


def score_candidate_single(
    model: GPT,
    encode_fn,
    ctx: str,
    candidate: str,
    device: str,
) -> float:
    ctx_ids = encode_fn(ctx)
    cand_ids = encode_fn(candidate)
    if not cand_ids:
        return -math.inf
    full_ids = ctx_ids + cand_ids
    block_size = model.config.block_size
    if len(full_ids) > block_size:
        full_ids = full_ids[-block_size:]
    start_idx = max(0, len(full_ids) - len(cand_ids) - 1)
    inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(inp, targets=tgt)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_slice = tgt[:, start_idx:]
        token_logprobs = logprobs[:, start_idx:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
        # Normalize by character count of candidate for fair comparison across tokenizers
        return token_logprobs.sum().item() / max(1, len(candidate.strip()))


def score_candidate_mc24(
    model: GPT,
    tok_24: HangulHybridPosTokenizer,
    ctx: str,
    candidate: str,
    device: str,
) -> float:
    ctx_steps = tok_24.encode_text(ctx)
    cand_steps = tok_24.encode_text(candidate)
    if not cand_steps:
        return -math.inf
    full_steps = ctx_steps + cand_steps
    block_size = model.config.block_size
    if len(full_steps) > block_size:
        full_steps = full_steps[-block_size:]
    start_idx = max(0, len(full_steps) - len(cand_steps) - 1)

    n_lanes = 24
    token_lists = [[step[i] for step in full_steps] for i in range(n_lanes)]
    input_lanes = [torch.tensor([lane[:-1]], dtype=torch.long, device=device) for lane in token_lists]
    target_lanes = [torch.tensor([lane[1:]], dtype=torch.long, device=device) for lane in token_lists]
    token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
    target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

    with torch.no_grad():
        logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)
        # Lane 0 (script/BPE) at all positions
        lp_0 = torch.log_softmax(logits_list[0], dim=-1)
        tgt_0 = target_lanes[0][:, start_idx:]
        total_lp = lp_0[:, start_idx:, :].gather(-1, tgt_0.unsqueeze(-1)).squeeze(-1).sum().item()

        # Lanes 1..3 (choseong, jungseong, jongseong) at Hangul syllable positions
        is_hangul_mask = (tgt_0 == tok_24.hangul_id).float()
        for lane_idx in (1, 2, 3):
            lp = torch.log_softmax(logits_list[lane_idx], dim=-1)
            tgt_slice = target_lanes[lane_idx][:, start_idx:]
            gathered = lp[:, start_idx:, :].gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
            total_lp += (gathered * is_hangul_mask).sum().item()

        # If model has output-only char head (Lane 24), include its syllable log-likelihood
        if len(logits_list) >= 25:
            lp_char = torch.log_softmax(logits_list[24], dim=-1)
            c_t = target_lanes[1][:, start_idx:].clamp(min=1) - 1
            v_t = target_lanes[2][:, start_idx:].clamp(min=1) - 1
            f_t = target_lanes[3][:, start_idx:]
            tgt_char = ((c_t * 588 + v_t * 28 + f_t) + 1) * (tgt_0 == tok_24.hangul_id).long()
            gathered_char = lp_char[:, start_idx:, :].gather(-1, tgt_char.unsqueeze(-1)).squeeze(-1)
            total_lp += (gathered_char * is_hangul_mask).sum().item()

    return total_lp / max(1, len(candidate.strip()))


def score_candidate_mc25(
    model: GPT,
    tok_25: HangulPosFactorizedTokenizer,
    stois_25: List[Dict[str, int]],
    ctx: str,
    candidate: str,
    device: str,
) -> float:
    ctx_seq = tok_25.encode_text(ctx)
    cand_seq = tok_25.encode_text(candidate)
    if not cand_seq:
        return -math.inf
    full_seq = ctx_seq + cand_seq
    block_size = model.config.block_size
    if len(full_seq) > block_size:
        full_seq = full_seq[-block_size:]
    start_idx = max(0, len(full_seq) - len(cand_seq) - 1)

    n_lanes = len(stois_25)
    token_lists = [[] for _ in range(n_lanes)]
    for item in full_seq:
        ch = item["char"]
        idx = item["indices"]
        if n_lanes == 4:
            byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
            token_lists[0].append(byte_id)
            for lane_i, factor_idx in enumerate([1, 2, 3], start=1):
                t_char = tok_25.token_for(factor_idx, idx[factor_idx])
                token_lists[lane_i].append(stois_25[lane_i].get(t_char, 0))
        else:
            for i in range(min(24, n_lanes - 1)):
                t_char = tok_25.token_for(i, idx[i])
                token_lists[i].append(stois_25[i].get(t_char, 0))
            if n_lanes > 24:
                byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
                token_lists[24].append(byte_id)

    vocab_sizes = getattr(model.config, "vocab_sizes", None)
    input_lanes = [torch.tensor([lane[:-1]], dtype=torch.long, device=device) for lane in token_lists]
    target_lanes = [torch.tensor([lane[1:]], dtype=torch.long, device=device) for lane in token_lists]
    if vocab_sizes is not None and len(vocab_sizes) >= n_lanes:
        input_lanes = [input_lanes[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_lanes)]
        target_lanes = [target_lanes[i].clamp(0, vocab_sizes[i] - 1) for i in range(n_lanes)]
    token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
    target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

    with torch.no_grad():
        logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)
        total_lp = 0.0
        for lane_idx in (1, 2, 3, 24):
            if lane_idx < len(logits_list):
                lp = torch.log_softmax(logits_list[lane_idx], dim=-1)
                tgt_slice = target_lanes[lane_idx][:, start_idx:]
                gathered = lp[:, start_idx:, :].gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
                total_lp += gathered.sum().item()

    return total_lp / max(1, len(candidate.strip()))


def evaluate_all_models(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
    tok_24_full = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")
    tok_24_coarse = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="coarse")
    tok_25_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")

    results: Dict[str, Any] = {}

    for model_id, mcfg in DEFAULT_MODELS.items():
        ckpt_path = os.path.join(REPO_ROOT, mcfg["path"])
        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\nEvaluating {mcfg['name']} ({model_id})...")
        model, ckpt_config = load_gpt_model(ckpt_path, device)
        m_type = mcfg["type"]

        encode_fn = None
        stois_25 = []
        if m_type == "single":
            meta_path = os.path.join(REPO_ROOT, mcfg["meta"])
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            encode_fn, _ = get_tokenizer_functions(meta)
        elif m_type == "mc_25":
            lane_names = ckpt_config.get("multicontext_datasets", [])
            for lp in lane_names:
                with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
                    stois_25.append(pickle.load(f)["stoi"])

        model_scores: Dict[str, Any] = {}
        total_correct = 0
        total_items = 0

        for task_name, items in TASK_DATASETS.items():
            correct = 0
            for item in items:
                ctx = item["ctx"]
                cands = item["candidates"]
                ans = item["answer"]

                cand_scores = []
                for cand in cands:
                    if m_type == "single":
                        s = score_candidate_single(model, encode_fn, ctx, cand, device)
                    elif m_type == "mc_24":
                        tok_24 = tok_24_coarse if "coarse" in model_id else tok_24_full
                        s = score_candidate_mc24(model, tok_24, ctx, cand, device)
                    elif m_type == "mc_25":
                        s = score_candidate_mc25(model, tok_25_full, stois_25, ctx, cand, device)
                    else:
                        s = -math.inf
                    cand_scores.append(s)

                pred = int(max(range(len(cand_scores)), key=lambda i: cand_scores[i]))
                if pred == ans:
                    correct += 1

            acc = correct / max(1, len(items))
            model_scores[task_name] = {
                "accuracy": round(acc * 100.0, 2),
                "correct": correct,
                "total": len(items),
            }
            total_correct += correct
            total_items += len(items)
            print(f"  - {task_name:<28}: {acc*100.0:5.1f}% ({correct}/{len(items)})")

        overall_acc = total_correct / max(1, total_items)
        model_scores["overall_accuracy"] = round(overall_acc * 100.0, 2)
        print(f"  => Overall Diagnostic Accuracy: {overall_acc*100.0:5.1f}% ({total_correct}/{total_items})")
        results[model_id] = {
            "name": mcfg["name"],
            "scores": model_scores,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run KoBALT & CUTE-Hangul Diagnostic Evaluation Suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results_kobalt_cute_hangul_eval.json")
    args = parser.parse_args()

    results = evaluate_all_models(device=args.device)
    out_path = os.path.join(REPO_ROOT, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved diagnostic benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
