#!/usr/bin/env python3
"""Evaluate all saved checkpoints (3, 5, 10, 15, 20 epochs & best_ckpt) of the newly weighted
24-lane No-Char Hangul Factorizers (Coarse POS & Full POS) on all downstream benchmarks including Seq2Seq.
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
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEVICE = "cuda:0"
MAX_EXAMPLES = 100

MODELS = [
    {
        "id": "nochar_full_pos_new_weighted",
        "name": "Hangul Factorizer No-Char (Full POS, New Winning Weights)",
        "type": "mc_24_full",
        "dir": "out_mc_nochar_full_pos_new_weighted",
    },
    {
        "id": "nochar_coarse_pos_new_weighted",
        "name": "Hangul Factorizer No-Char (Coarse POS, New Winning Weights)",
        "type": "mc_24_coarse",
        "dir": "out_mc_nochar_coarse_pos_new_weighted",
    },
]

CHECKPOINTS = [
    ("3_epoch", "3_epochs.pt"),
    ("5_epoch", "5_epochs.pt"),
    ("10_epoch", "10_epochs.pt"),
    ("15_epoch", "15_epochs.pt"),
    ("20_epoch", "20_epochs.pt"),
    ("best", "best_ckpt.pt"),
]


def evaluate_four_caps_and_hellaswag():
    results = {}
    for m in MODELS:
        m_id = m["id"]
        out_dir = m["dir"]
        results[m_id] = {}
        for ep_name, ckpt_file in CHECKPOINTS:
            ckpt_path = os.path.join(REPO_ROOT, out_dir, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path} (not found)")
                continue

            print(f"\n>>> Evaluating 4 Capabilities & Ko-HellaSwag: {m_id} @ {ep_name} ({ckpt_path})...")
            res_entry = {}

            # 1. Four Capabilities
            t4 = f"/tmp/temp_4cap_{m_id}_{ep_name}.json"
            subprocess.run(
                [
                    sys.executable,
                    "benchmarks/run_four_capability_evals.py",
                    "--ckpts",
                    f"{ep_name}:{ckpt_path}",
                    "--device",
                    DEVICE,
                    "--max_examples",
                    str(MAX_EXAMPLES),
                    "--output_json",
                    t4,
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            if os.path.exists(t4):
                with open(t4) as f:
                    res_entry["four_capabilities"] = json.load(f).get(ep_name, {})
                os.remove(t4)

            # 2. Ko-HellaSwag
            ths = f"/tmp/temp_hs_{m_id}_{ep_name}.json"
            subprocess.run(
                [
                    sys.executable,
                    "benchmarks/run_ko_hellaswag.py",
                    "--ckpt_path",
                    ckpt_path,
                    "--device",
                    DEVICE,
                    "--eval_all_norms",
                    "--max_examples",
                    str(MAX_EXAMPLES),
                    "--output_json",
                    ths,
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            if os.path.exists(ths):
                with open(ths) as f:
                    res_entry["ko_hellaswag"] = json.load(f).get("accuracies", {})
                os.remove(ths)

            results[m_id][ep_name] = res_entry
            with open("results_new_weighted_nochar_all_epochs.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def evaluate_seq2seq_and_oov():
    from run_seq2seq_translation_benchmark import (
        TranslationEvaluator,
        canonicalize,
        load_model_from_ckpt,
        TEST_JSONL,
    )
    import sacrebleu

    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    tf_samples = all_samples[:100]
    gen_samples = all_samples[:20]
    canon_refs = [canonicalize(s["ko"]) for s in gen_samples]

    seq2seq_results = {}
    for m in MODELS:
        m_id = m["id"]
        out_dir = m["dir"]
        seq2seq_results[m_id] = {}

        for ep_name, ckpt_file in CHECKPOINTS:
            ckpt_rel_path = os.path.join(out_dir, ckpt_file)
            ckpt_path = os.path.join(REPO_ROOT, ckpt_rel_path)
            if not os.path.exists(ckpt_path):
                continue
            print(f"\n>>> Evaluating Seq2Seq Translation: {m_id} @ {ep_name} ({ckpt_path})...")
            model, ckpt_dict = load_model_from_ckpt(ckpt_path, device=DEVICE)
            val_loss = float(ckpt_dict.get("best_val_loss", 0.0))
            iter_num = int(ckpt_dict.get("iter_num", 0))

            evaluator = TranslationEvaluator(m, ckpt_dict=ckpt_dict, device=DEVICE)
            t_eval = evaluator.evaluate_teacher_forced(model, tf_samples)
            hyps, gen_lat_ms = evaluator.generate_completions(model, gen_samples, max_new_tokens=40)
            canon_hyps = [canonicalize(h) for h in hyps]

            filt_hyps = [h if h.strip() else "<empty>" for h in canon_hyps]
            filt_refs = [r if r.strip() else "<empty>" for r in canon_refs]

            bleu = sacrebleu.corpus_bleu(filt_hyps, [filt_refs], smooth_method="exp", use_effective_order=True).score
            chrf = sacrebleu.corpus_chrf(filt_hyps, [filt_refs], char_order=18).score

            seq2seq_results[m_id][ep_name] = {
                "path": ckpt_rel_path,
                "iter_num": iter_num,
                "train_val_loss": val_loss,
                "bpj": t_eval["bpj"],
                "target_loss": t_eval["target_loss"],
                "target_ppl": t_eval["target_ppl"],
                "bleu": float(bleu),
                "chrf": float(chrf),
                "latency_ms": float(gen_lat_ms),
            }

            with open("results_new_weighted_nochar_seq2seq.json", "w") as f:
                json.dump(seq2seq_results, f, indent=2, ensure_ascii=False)

    return seq2seq_results


if __name__ == "__main__":
    # evaluate_four_caps_and_hellaswag() already completed and saved to results_new_weighted_nochar_all_epochs.json
    evaluate_seq2seq_and_oov()
    print("\n>>> Running OOV / Unicode Stress Test on best_ckpt.pt for new models...")
    subprocess.run(
        [
            sys.executable,
            "benchmarks/run_oov_evaluations.py",
            "--models",
            "nochar_coarse_pos_new_weighted",
            "nochar_full_pos_new_weighted",
            "--output_json",
            "results_new_weighted_nochar_oov.json",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    print("All downstream evaluations for newly weighted No-Char models completed!")
