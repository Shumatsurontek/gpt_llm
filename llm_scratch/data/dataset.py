from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    data_path: str
    seq_len: int = 256
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    pad_id: Optional[int] = None


class CausalLMDataset(Dataset):
    """
    Produces (input_ids, target_ids) pairs from a token id stream stored as a .pt tensor
    Expect a tensor of shape (N,) with dtype torch.long, containing token ids.
    """

    def __init__(self, ids_path: str | Path, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.ids = torch.load(str(ids_path))  # tensor long shape (N,)
        assert self.ids.dim() == 1
        self.N = self.ids.numel()
        self.seq_len = cfg.seq_len

    def __len__(self):
        # number of windows we can extract
        return max(0, (self.N - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        i = idx * self.seq_len
        j = i + self.seq_len + 1
        x = self.ids[i:j-1]
        y = self.ids[i+1:j]
        return x.clone(), y.clone()

