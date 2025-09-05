# Core model components: attention with relative bias (local causal), GLU-FFN, blocks
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Tuple as Tup

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., d) where d even
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_rope_cache(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> Tup[torch.Tensor, torch.Tensor]:
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, half)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (T, dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


class CausalLocalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, local_window: int = 512, dropout: float = 0.1, attn_dropout: float = 0.1, use_rel_bias: bool = False, use_rope: bool = True, rope_base: float = 10000.0, max_rel_distance: int = 128):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.local_window = local_window
        self.use_rel_bias = use_rel_bias
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout_p = attn_dropout
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativePositionBias(n_heads, max_rel_distance) if use_rel_bias else None

        self._mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        if self.use_rope:
            # apply RoPE to q and k
            cos, sin = _build_rope_cache(T, self.head_dim, x.device, x.dtype, base=self.rope_base)
            cos = cos[None, None, :, :]  # (1,1,T,D)
            sin = sin[None, None, :, :]
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)

        # attention mask cache key (T, local_window, device)
        key = (T, int(self.local_window) if self.local_window is not None else -1, str(x.device))
        mask = self._mask_cache.get(key)
        if mask is None:
            base = torch.full((T, T), 0.0, device=x.device, dtype=x.dtype)
            # causal lower-triangular allowed; disallow others with -inf
            causal = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            if self.local_window is not None:
                i = torch.arange(T, device=x.device)
                j = torch.arange(T, device=x.device)
                local = (i[:, None] - j[None, :]).abs() <= self.local_window
                allow = causal & local
            else:
                allow = causal
            base = base.masked_fill(~allow, float("-inf"))
            self._mask_cache[key] = base
            mask = base

        # scaled dot-product attention (PyTorch optimized)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class GLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.w12 = nn.Linear(dim, 2 * hidden, bias=True)
        self.proj = nn.Linear(hidden, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w12(x)
        val, gate = torch.chunk(h, 2, dim=-1)
        y = F.silu(gate) * val
        y = self.proj(y)
        return self.dropout(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, local_window: int, dropout: float = 0.1, attn_dropout: float = 0.1, ffn_mult: int = 4, norm="layernorm", use_rel_bias=False, use_rope=True):
        super().__init__()
        self.norm1 = RMSNorm(dim) if norm == "rmsnorm" else nn.LayerNorm(dim)
        self.attn = CausalLocalSelfAttention(dim, n_heads, local_window, dropout, attn_dropout, use_rel_bias=use_rel_bias, use_rope=use_rope)
        self.norm2 = RMSNorm(dim) if norm == "rmsnorm" else nn.LayerNorm(dim)
        self.ffn = GLUFFN(dim, hidden_mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

