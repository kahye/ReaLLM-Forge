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

with open("results_seq2seq_ko_to_en_all_15_best_checkpoints.json", "r") as f:
    ko_en = json.load(f)

with open("results_seq2seq_all_15_best_checkpoints_detailed.json", "r") as f:
    en_ko = json.load(f)

print("="*125)
print(f"{'Model ID':<36} | {'Family':<20} | {'KO->EN BPC':<10} | {'L0 Loss':<8} | {'BLEU':<6} | {'chrF':<6} | {'EN->KO BPJ':<10} | {'EN BLEU':<8} | {'Lat (ms)':<8}")
print("="*125)
for mid, m_data in ko_en["models"].items():
    fam = m_data["family"]
    c_ko = m_data["corpus_metrics"]
    c_en = en_ko["models"][mid]["corpus_metrics"] if mid in en_ko["models"] else {}
    bpc = c_ko["bpc"]
    l0 = c_ko["lane0_target_loss"]
    bleu_ko = c_ko["bleu"]
    chrf_ko = c_ko["chrf"]
    bpj_en = c_en.get("bpj", 0.0)
    bleu_en = c_en.get("bleu", 0.0)
    lat = c_ko["latency_ms"]
    print(f"{mid:<36} | {fam:<20} | {bpc:10.4f} | {l0:8.4f} | {bleu_ko:6.2f} | {chrf_ko:6.2f} | {bpj_en:10.4f} | {bleu_en:8.2f} | {lat:8.1f}")
print("="*125)
