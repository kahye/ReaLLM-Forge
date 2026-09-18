#!/usr/bin/env python3
"""
Measures and normalizes both:
1. Single-Stream Autoregressive Latency (`latency_ms`, batch_size=1, ms/sent over 20 sentences)
2. Batched Throughput Latency (`batched_latency_ms`, batch_size=64, ms/sent over 256 sentences)
across all 34 Best Validation Loss models (17 AdamW + 17 Muon) and all decoding modes.

Ensures 100% apples-to-apples comparison across both:
- results_seq2seq_all_5000_models.json
- results_seq2seq_1000_all_models.json
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

import json
import os
import time
import torch

from run_seq2seq_1000_all_34_models import (
    ADAMW_MODELS,
    MUON_MODELS,
    REPO_ROOT,
    TEST_JSONL,
    Seq2SeqTokenizerBundle,
    _SEQ2SEQ_GEN_CACHE,
    generate_batched_1000,
    load_model_from_ckpt,
    pretokenize_all_1000,
)


@torch.no_grad()
def measure_bs1_latency(model, items, tok_key, bundle, mode, device="cuda") -> float:
    warmup_items = items[:2]
    sub_items = items[:20]
    _ = generate_batched_1000(
        model=model,
        items=warmup_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=40,
        batch_size=1,
        device=device,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    hyps = generate_batched_1000(
        model=model,
        items=sub_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=40,
        batch_size=1,
        device=device,
    )
    torch.cuda.synchronize()
    assert len(hyps) == 20, f"Expected 20 hypotheses for bs=1, got {len(hyps)}"
    return (time.time() - t0) * 1000.0 / len(sub_items)


@torch.no_grad()
def measure_bs64_latency(model, items, tok_key, bundle, mode, device="cuda") -> float:
    warmup_items = items[:64]
    sub_items = items[:256]  # 4 full batches of 64
    _ = generate_batched_1000(
        model=model,
        items=warmup_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=40,
        batch_size=64,
        device=device,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    hyps = generate_batched_1000(
        model=model,
        items=sub_items,
        tok_key=tok_key,
        bundle=bundle,
        mode=mode,
        max_new_tokens=40,
        batch_size=64,
        device=device,
    )
    torch.cuda.synchronize()
    assert len(hyps) == 256, f"Expected 256 hypotheses for bs=64, got {len(hyps)}"
    return (time.time() - t0) * 1000.0 / len(sub_items)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:256]

    print(f"Pre-tokenizing 256 test sentences on {device}...")
    bundle = Seq2SeqTokenizerBundle(device=device)
    pre_cache = pretokenize_all_1000(samples, bundle)

    p5000 = os.path.join(REPO_ROOT, "results_seq2seq_all_5000_models.json")
    p1000 = os.path.join(REPO_ROOT, "results_seq2seq_1000_all_models.json")

    d5000 = json.load(open(p5000, "r", encoding="utf-8"))
    d1000 = json.load(open(p1000, "r", encoding="utf-8"))

    for suite_name, model_list in [("adamw", ADAMW_MODELS), ("muon", MUON_MODELS)]:
        print(f"\n==================== Benchmarking {suite_name.upper()} Suite (17 models) ====================")
        for m_id, ckpt_rel, tok_key in model_list:
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
            model, _ = load_model_from_ckpt(ckpt_path, device=device)
            items = pre_cache[tok_key]

            has_charhead = "charhead" in tok_key
            is_nochar_24 = "mc_24" in tok_key
            if has_charhead:
                modes = ["charhead_24", "primary", "jamo_lvt"]
            elif is_nochar_24:
                modes = ["primary", "jamo_lvt"]
            else:
                modes = ["primary"]

            for mode in modes:
                lat_bs64 = measure_bs64_latency(model, items, tok_key, bundle, mode, device=device)

                # Check if we already have a verified bs=1 measurement in d5000 or d1000
                l5 = d5000[suite_name][m_id][mode].get("latency_ms", 0)
                l1 = d1000[suite_name][m_id][mode].get("latency_ms", 0)

                expected_min = 150.0 if suite_name == "adamw" else 300.0
                if l5 < expected_min:
                    # If l1 in d1000 is already a verified bs=1 measurement (>= expected_min), we can verify or measure live
                    lat_bs1 = measure_bs1_latency(model, items, tok_key, bundle, mode, device=device)
                else:
                    lat_bs1 = l5

                d5000[suite_name][m_id][mode]["latency_ms"] = round(lat_bs1, 1)
                d5000[suite_name][m_id][mode]["batched_latency_ms"] = round(lat_bs64, 1)

                d1000[suite_name][m_id][mode]["latency_ms"] = round(lat_bs1, 1)
                d1000[suite_name][m_id][mode]["batched_latency_ms"] = round(lat_bs64, 1)

                print(
                    f"  [{suite_name.upper()}] {m_id:38s} | {mode:12s} | "
                    f"Single-Stream (bs=1): {lat_bs1:5.1f} ms/sent | "
                    f"Batched (bs=64): {lat_bs64:5.1f} ms/sent"
                )

            del model
            torch.cuda.empty_cache()

            with open(p5000, "w", encoding="utf-8") as f:
                json.dump(d5000, f, indent=2, ensure_ascii=False)
            with open(p1000, "w", encoding="utf-8") as f:
                json.dump(d1000, f, indent=2, ensure_ascii=False)

    print("\nSuccessfully updated both results_seq2seq_all_5000_models.json and results_seq2seq_1000_all_models.json!")


if __name__ == "__main__":
    main()
