# Core model components: attention with relative bias (local causal), GLU-FFN, blocks
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros((2 * max_distance + 1, num_heads)))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        # returns (num_heads, q_len, k_len)
        # distances i - j clipped to [-max_distance, max_distance]
        ctx = torch.arange(q_len, device=device)[:, None]
        keys = torch.arange(k_len, device=device)[None, :]
        rel = ctx - keys
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        out = self.bias[rel]  # (q_len, k_len, H)
        return out.permute(2, 0, 1)


class CausalLocalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, local_window: int = 512, dropout: float = 0.1, attn_dropout: float = 0.1, use_rel_bias: bool = True, max_rel_distance: int = 128):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.local_window = local_window
        self.use_rel_bias = use_rel_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativePositionBias(n_heads, max_rel_distance) if use_rel_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)
        # scaled dot-product attention with causal + local mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # causal mask
        mask = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
        # local window: only attend within +/- local_window
        if self.local_window is not None:
            i = torch.arange(T, device=x.device)
            j = torch.arange(T, device=x.device)
            local = (i[:, None] - j[None, :]).abs() <= self.local_window
            mask = mask & local
        attn = attn.masked_fill(~mask, float("-inf"))

        if self.use_rel_bias and self.rel_bias is not None:
            attn = attn + self.rel_bias(T, T, x.device)[None, :, :, :]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class GLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(dim, hidden)
        self.proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GLU with SiLU gate
        a = F.silu(self.fc1(x))
        g = self.fc2(x)
        y = a * g
        y = self.proj(y)
        return self.dropout(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, local_window: int, dropout: float = 0.1, attn_dropout: float = 0.1, ffn_mult: int = 4, norm="layernorm", use_rel_bias=True):
        super().__init__()
        self.norm1 = RMSNorm(dim) if norm == "rmsnorm" else nn.LayerNorm(dim)
        self.attn = CausalLocalSelfAttention(dim, n_heads, local_window, dropout, attn_dropout, use_rel_bias=use_rel_bias)
        self.norm2 = RMSNorm(dim) if norm == "rmsnorm" else nn.LayerNorm(dim)
        self.ffn = GLUFFN(dim, hidden_mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

