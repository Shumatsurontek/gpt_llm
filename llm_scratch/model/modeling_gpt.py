# GPT-like autoregressive language model with local attention
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .blocks import TransformerBlock, RMSNorm


@dataclass
class GPTConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 512
    local_window: int | None = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_mult: int = 4
    norm: str = "layernorm"  # or "rmsnorm"
    use_rel_bias: bool = False
    use_rope: bool = True
    tie_weights: bool = True


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.n_heads, local_window=cfg.local_window if cfg.local_window is not None else cfg.max_seq_len, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout, ffn_mult=cfg.ffn_mult, norm=cfg.norm, use_rel_bias=cfg.use_rel_bias, use_rope=cfg.use_rope)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = nn.LayerNorm(cfg.dim) if cfg.norm == "layernorm" else RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        x = self.token_emb(idx)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

