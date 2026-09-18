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
from hangul_factorizer import S_BASE, S_COUNT, N_COUNT, T_COUNT, CHOSEONG, JUNGSEONG, JONGSEONG, HangulPosFactorizedTokenizer

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

stois_three_hot = []
for lp in ["korean_pos_mc/char_byte", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong"]:
    with open(os.path.join(REPO_ROOT, "data", lp, "meta.pkl"), "rb") as f:
        stois_three_hot.append(pickle.load(f)["stoi"])

tok_f = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
model, conf = load_model("out_three_hot_ind_char_byte_muon/20_epochs.pt")

s = '갂'
prefix = "단어: "
code = ord(s) - S_BASE
L = code // N_COUNT
V = (code % N_COUNT) // T_COUNT
T = code % T_COUNT

prompt = prefix + s
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

vocab_sizes = conf.vocab_sizes
inp = [torch.tensor([l[:-1]], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
td = {f"lane_{i}": l for i, l in enumerate(inp)}
with torch.no_grad():
    logits_list, _ = model(token_dict=td)

target_l = t_lists[1][-1]
target_v = t_lists[2][-1]
target_t = t_lists[3][-1]
lp_l = F.log_softmax(logits_list[1][0, -1, :], dim=-1)[target_l].item()
lp_v = F.log_softmax(logits_list[2][0, -1, :], dim=-1)[target_v].item()
lp_t = F.log_softmax(logits_list[3][0, -1, :], dim=-1)[target_t].item()
nll = -(lp_l + lp_v + lp_t)
bpc = nll / math.log(2)
print(f"Three-hot BPC for {s}: {bpc:.2f} (L:{target_l}, V:{target_v}, T:{target_t})")

# Exact match test
inp_pref = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(t_lists)]
# using only prefix length:
pref_seq = tok_f.encode_text(prefix)
pref_lists = [[] for _ in range(4)]
for item in pref_seq:
    ch = item["char"]
    idx = item["indices"]
    byte_id = ord(ch.encode("utf-8")[0:1]) if len(ch) > 0 else 0
    pref_lists[0].append(byte_id)
    for li, fi in enumerate([1, 2, 3], start=1):
        tc = tok_f.token_for(fi, idx[fi])
        pref_lists[li].append(stois_three_hot[li].get(tc, 0))

inp_p = [torch.tensor([l], device=DEVICE, dtype=torch.long).clamp(0, vocab_sizes[i] - 1) for i, l in enumerate(pref_lists)]
td_p = {f"lane_{i}": l for i, l in enumerate(inp_p)}
with torch.no_grad():
    logits_p, _ = model(token_dict=td_p)

pred_l = logits_p[1][0, -1, :].argmax().item()
pred_v = logits_p[2][0, -1, :].argmax().item()
pred_t = logits_p[3][0, -1, :].argmax().item()
match = (pred_l == target_l and pred_v == target_v and pred_t == target_t)
print(f"Pred L={pred_l}, V={pred_v}, T={pred_t} | Target L={target_l}, V={target_v}, T={target_t} | Match={match}")
