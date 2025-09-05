# Simple nucleus/top-k sampling utilities
from __future__ import annotations

import torch


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    # logits: (V,)
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float('inf')
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumprobs > top_p
        # shift to keep at least one token
        mask[..., 0] = False
        indices_to_remove = sorted_indices[mask]
        logits[indices_to_remove] = -float('inf')
    return logits

