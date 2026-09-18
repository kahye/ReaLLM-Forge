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
import os, sys, math, json, pickle, torch
import numpy as np
import torch.nn.functional as F
from dataclasses import fields

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from model import GPT, GPTConfig
from eval_oov_and_tail_bpc import load_model, load_tokenizers, eval_syllable_bpc, eval_syllable_exact_match, eval_syllable_minimal_pair, eval_prompt_oov_bpc, MODELS

tok = load_tokenizers()
sample_models = [MODELS[0], MODELS[1], MODELS[2], MODELS[3], MODELS[5], MODELS[9], MODELS[13]]

prompt = "훈민정음 언해본 서문에서 반시옷(ㅿ)의 음가를 설명해줘."
oov_chars = {'ㅿ'}

for name, path, label, m_type in sample_models:
    print(f"Testing {name} ({m_type})...")
    m, conf = load_model(path)
    bpc_s = eval_syllable_bpc(m, conf, m_type, tok, '갂')
    exact = eval_syllable_exact_match(m, conf, m_type, tok, '갂')
    discr = eval_syllable_minimal_pair(m, conf, m_type, tok, '갂')
    print(f"  Syllable 갂: BPC={bpc_s:.2f}, Exact={exact}, Discr={discr}")
    
    bpc_oov = eval_prompt_oov_bpc(m, conf, m_type, tok, prompt, oov_chars)
    print(f"  OOV ㅿ: BPC={bpc_oov:.2f}")

print("All tested model types passed successfully!")
