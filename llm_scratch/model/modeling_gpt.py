# GPT-like autoregressive language model with local attention
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .blocks import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_mult: int = 4
    norm: str = "layernorm"  # or "rmsnorm"
    use_rel_bias: bool = True
    tie_weights: bool = True


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.n_heads, local_window=cfg.max_seq_len, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout, ffn_mult=cfg.ffn_mult, norm=cfg.norm, use_rel_bias=cfg.use_rel_bias)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = nn.LayerNorm(cfg.dim) if cfg.norm == "layernorm" else nn.Identity()
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=True)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)[None, :]
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

