# gemma3_continual_model.py
"""Gemma 3 270M Continual Pretraining Model for ReaLLM-Forge.

Wraps Hugging Face's official Gemma3TextModel (18 layers, d=640, GQA, dual-frequency RoPE,
sliding-window attention, Q/K RMSNorm, and GeGLU) with compact tokenizer input embeddings
and Fused Head Projections for deployment and continual training.
"""
from __future__ import annotations
import os, os as _os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
from gpt_conf import GPTConfig

DEFAULT_BACKBONE_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "gemma3_270m_backbone.pt"
)


class Gemma3ContinualModel(nn.Module):
    """Gemma 3 270M backbone with custom tokenizer embeddings and Fused LM Heads."""

    def __init__(
        self,
        config: GPTConfig,
        backbone_ckpt_path: Optional[str] = DEFAULT_BACKBONE_CKPT,
    ):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.is_multicontext = bool(config.multicontext or config.multidataset_wte)
        self.output_only_lanes = set(getattr(config, "output_only_lanes", []) or [])
        self.vocab_sizes = list(getattr(config, "vocab_sizes", []) or [])
        self.is_fused_head = bool(getattr(config, "fused_lm_head", False))

        # Instantiate Gemma3TextModel architecture
        hf_config = AutoConfig.from_pretrained("google/gemma-3-270m")
        text_cfg = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        self.gemma = Gemma3TextModel(text_cfg)

        # Remove the massive 262k native embedding table
        if hasattr(self.gemma, "embed_tokens"):
            del self.gemma.embed_tokens

        # Load pretrained 18 transformer layers and final RMSNorm from checkpoint
        if backbone_ckpt_path and os.path.exists(backbone_ckpt_path):
            state = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
            clean_state = {}
            for k, v in state.items():
                if k.startswith("model."):
                    clean_state[k[len("model."):]] = v
                else:
                    clean_state[k] = v
            self.gemma.load_state_dict(clean_state, strict=False)

        # Build custom tokenizer embeddings and heads
        d_model = getattr(text_cfg, "hidden_size", 640)
        self.config.n_embd = d_model
        self.transformer = nn.ModuleDict()

        if self.is_multicontext:
            for i, v_size in enumerate(self.vocab_sizes):
                if i not in self.output_only_lanes:
                    self.transformer[f"wte_{i}"] = nn.Embedding(v_size, d_model)
                self.transformer[f"lm_head_{i}"] = nn.Linear(d_model, v_size, bias=False)

            # Weight tying for input lanes
            if getattr(config, "wte_weight_tying", True):
                for i in range(len(self.vocab_sizes)):
                    if i not in self.output_only_lanes:
                        self.transformer[f"lm_head_{i}"].weight = self.transformer[f"wte_{i}"].weight
        else:
            self.transformer["wte"] = nn.Embedding(config.vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)
            if getattr(config, "wte_weight_tying", True):
                self.lm_head.weight = self.transformer["wte"].weight

        # Reference ln_f and h for hooks / inspection in training
        self.transformer["ln_f"] = self.gemma.norm
        self.transformer["h"] = self.gemma.layers

    def crop_block_size(self, block_size: int):
        """Adjust block size (Gemma 3 uses RoPE so no embedding table cropping is required)."""
        self.block_size = block_size
        self.config.block_size = block_size

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.is_multicontext:
                for i in range(len(self.vocab_sizes)):
                    if i not in self.output_only_lanes and f"wte_{i}" in self.transformer:
                        n_params -= self.transformer[f"wte_{i}"].weight.numel()
            else:
                if "wte" in self.transformer:
                    n_params -= self.transformer["wte"].weight.numel()
        return n_params

    def set_lsv_index(self, idx: int):
        pass

    def freeze_non_lsv_parameters(self):
        pass

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPs utilization (MFU) on A100 GPU (312 TFLOPS bfloat16 peak)."""
        N = self.get_num_params()
        L, H, Q, T = 18, 4, 256, getattr(self, "block_size", 256)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / max(dt, 1e-6))
        flops_promised = 312e12
        return float(flops_achieved / flops_promised)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive generation for single stream."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def fuse_lm_heads(self) -> Gemma3ContinualModel:
        """Fuse all multicontext LM heads into a single projection for deployment/inference."""
        if not self.is_multicontext or not self.vocab_sizes:
            return self

        total_vocab = sum(self.vocab_sizes)
        d_model = self.config.n_embd
        ref_weight = self.transformer["lm_head_0"].weight
        device = ref_weight.device
        dtype = ref_weight.dtype

        fused_head = nn.Linear(d_model, total_vocab, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            fused_weight = torch.cat(
                [self.transformer[f"lm_head_{i}"].weight for i in range(len(self.vocab_sizes))],
                dim=0,
            )
            fused_head.weight.copy_(fused_weight)

        self.transformer["lm_head_fused"] = fused_head
        self.is_fused_head = True
        return self

    def forward(
        self,
        idx: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        targets: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        token_dict: Optional[Dict[str, torch.Tensor]] = None,
        target_dict: Optional[Dict[str, torch.Tensor]] = None,
        token_list: Optional[List[torch.Tensor]] = None,
        target_list: Optional[List[torch.Tensor]] = None,
        loss_weights: Optional[List[float]] = None,
        loss_fn: Optional[Any] = None,
        iter_num: Optional[int] = None,
        **kwargs,
    ):
        if self.is_multicontext or token_dict is not None or isinstance(idx, (list, tuple)):
            return self._forward_multicontext(
                idx=idx,
                targets=targets,
                token_dict=token_dict,
                target_dict=target_dict,
                token_list=token_list,
                target_list=target_list,
                loss_weights=loss_weights,
            )
        else:
            return self._forward_single(idx=idx, targets=targets)

    def _forward_single(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()
        x = self.transformer["wte"](idx)
        hidden = self.gemma(inputs_embeds=x).last_hidden_state

        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference optimization: slice last position
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None

        return logits, loss

    def _forward_multicontext(
        self,
        idx: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        targets: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        token_dict: Optional[Dict[str, torch.Tensor]] = None,
        target_dict: Optional[Dict[str, torch.Tensor]] = None,
        token_list: Optional[List[torch.Tensor]] = None,
        target_list: Optional[List[torch.Tensor]] = None,
        loss_weights: Optional[List[float]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        if token_list is None:
            if token_dict is not None:
                if all(f"lane_{i}" in token_dict for i in range(len(token_dict))):
                    token_list = [token_dict[f"lane_{i}"] for i in range(len(token_dict))]
                else:
                    token_list = list(token_dict.values())
            elif isinstance(idx, (list, tuple)):
                token_list = list(idx)

        if target_list is None:
            if target_dict is not None:
                if all(f"lane_{i}" in target_dict for i in range(len(target_dict))):
                    target_list = [target_dict[f"lane_{i}"] for i in range(len(target_dict))]
                else:
                    target_list = list(target_dict.values())
            elif isinstance(targets, (list, tuple)):
                target_list = list(targets)

        # Additive embedding across active input lanes
        x = None
        for i, v_size in enumerate(self.vocab_sizes):
            if i not in self.output_only_lanes:
                h = self.transformer[f"wte_{i}"](token_list[i])
                x = h if x is None else x + h

        hidden = self.gemma(inputs_embeds=x).last_hidden_state

        # Fused Head Projection
        if self.is_fused_head and "lm_head_fused" in self.transformer:
            all_logits = self.transformer["lm_head_fused"](hidden)
        else:
            W_fused = torch.cat(
                [self.transformer[f"lm_head_{i}"].weight for i in range(len(self.vocab_sizes))],
                dim=0,
            )
            all_logits = F.linear(hidden, W_fused)

        logits = list(torch.split(all_logits, self.vocab_sizes, dim=-1))

        if target_list is not None:
            losses = []
            for i, tgt in enumerate(target_list):
                if tgt.numel() == 0:
                    continue
                loss_i = F.cross_entropy(
                    logits[i].view(-1, self.vocab_sizes[i]),
                    tgt.view(-1),
                    ignore_index=-1,
                )
                losses.append(loss_i)
            return logits, losses
        else:
            logits = [pred[:, [-1], :] for pred in logits]
            return logits, None

    @classmethod
    def from_checkpoint_dict(cls, checkpoint: Dict[str, Any], device: str = "cpu") -> Gemma3ContinualModel:
        model_args = dict(checkpoint.get("model_args", {}))
        gptconf = GPTConfig(**model_args)
        model = cls(gptconf, backbone_ckpt_path=None)

        state_dict = checkpoint["model"]
        prefix = "_orig_mod."
        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                clean_sd[k[len(prefix):]] = v
            else:
                clean_sd[k] = v

        if "transformer.lm_head_fused.weight" in clean_sd:
            model.fuse_lm_heads()

        model.load_state_dict(clean_sd, strict=False)
        model.to(device)
        return model
