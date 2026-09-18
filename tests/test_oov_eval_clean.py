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
from sample import get_tokenizer_functions
from hangul_factorizer import S_BASE, S_COUNT, N_COUNT, T_COUNT, CHOSEONG, JUNGSEONG, JONGSEONG, HangulPosFactorizedTokenizer
from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    m_args = {k: v for k, v in m_args.items() if k in allowed}
    m_args["dropout"] = 0.0
    conf = GPTConfig(**m_args)
    model = GPT(conf)
    sd = checkpoint["model"]
    prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    return model, conf

tok_f = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
tok_h = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode="full")

stois_three_hot = []
for lp in ["korean_pos_mc/char_byte", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong"]:
    with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
        stois_three_hot.append(pickle.load(f)["stoi"])

prompt = "조선시대 문헌에서 삼룡구름용(龘)과 부엌 화로(爨) 같은 희귀 한자 표기를 발견했다."
oov_chars = {'龘', '爨'}

# Test 1: Three-hot
m_th, c_th = load_model("out_three_hot_ind_char_byte_muon/20_epochs.pt")
seq = tok_f.encode_text(prompt)
t_lists = [[] for _ in range(4)]
for item in seq:
    ch = item["char"]
    idx = item["indices"]
    byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
    t_lists[0].append(byte_id)
    for li, fi in enumerate([1, 2, 3], start=1):
        tc = tok_f.token_for(fi, idx[fi])
        t_lists[li].append(stois_three_hot[li].get(tc, 0))

vocab_sizes = c_th.vocab_sizes
inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
tgt = [torch.tensor([l[1:]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
td = {f"lane_{i}": l for i, l in enumerate(inp)}
tg = {f"lane_{i}": l for i, l in enumerate(tgt)}
with torch.no_grad():
    logits_list, _ = m_th(token_dict=td, target_dict=tg)

bpcs = []
for pos, item in enumerate(seq[1:]): # target position is pos
    ch = item["char"]
    if ch in oov_chars:
        # Lane 0 target
        target_byte = tgt[0][0, pos].item()
        lp = F.log_softmax(logits_list[0][0, pos, :], dim=-1)[target_byte].item()
        bpcs.append(-lp / math.log(2))
print(f"Three-hot prompt OOV BPC: {sum(bpcs)/len(bpcs):.2f}")

# Test 2: No-char H11
m_h11, c_h11 = load_model("out_mc_nochar_full_pos_charhead_h11_muon/20_epochs.pt")
steps = tok_h.encode_text(prompt)
inp_h = [torch.tensor([[s[i] for s in steps[:-1]]], device=DEVICE, dtype=torch.long) for i in range(24)]
tgt_h = [torch.tensor([[s[i] for s in steps[1:]]], device=DEVICE, dtype=torch.long) for i in range(24)]
td_h = {f"lane_{i}": l for i, l in enumerate(inp_h)}
tg_h = {f"lane_{i}": l for i, l in enumerate(tgt_h)}
with torch.no_grad():
    logits_list, _ = m_h11(token_dict=td_h, target_dict=tg_h)
bpcs_h = []
# For steps, check where oov chars land
for pos in range(len(steps) - 1):
    piece = tok_h.sp.id_to_piece(steps[pos + 1][0])
    if any(c in piece for c in oov_chars):
        target_bpe = tgt_h[0][0, pos].item()
        lp = F.log_softmax(logits_list[0][0, pos, :], dim=-1)[target_bpe].item()
        bpcs_h.append(-lp / math.log(2))
print(f"H11 prompt OOV BPC: {sum(bpcs_h)/len(bpcs_h):.2f}")

print("Clean OOV evaluation test passed!")
