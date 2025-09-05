from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class TrainConfig:
    # data
    train_ids: str = "data/train_ids.pt"
    val_ids: str = "data/val_ids.pt"
    seq_len: int = 256

    # tokenizer
    tokenizer_path: str = "artifacts/tokenizer.json"

    # model
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_mult: int = 4
    max_seq_len: int = 256
    norm: str = "layernorm"
    use_rel_bias: bool = False
    use_rope: bool = True
    local_window: int | None = 256
    tie_weights: bool = True

    # optim
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 2000
    grad_clip: float = 1.0
    batch_size: int = 32

    # training
    seed: int = 42
    device: str | None = None  # auto if None; prefer "mps" on Mac
    ckpt_dir: str = "artifacts/checkpoints"

    # logging
    wandb_project: str | None = None  # e.g., "llm-scratch"
    wandb_run_name: str | None = None
    wandb_enabled: bool = False


def load_train_config(path: str | Path) -> TrainConfig:
    cfg = load_yaml(path)
    return TrainConfig(**cfg)

