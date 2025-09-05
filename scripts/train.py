# Train a GPT-like model on a tokenized dataset
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import logging
import wandb
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_scratch.config import TrainConfig, load_train_config
from llm_scratch.data.dataset import CausalLMDataset, DataConfig
from llm_scratch.model.modeling_gpt import GPTModel, GPTConfig
from llm_scratch.tokenizer.bpe import BPETokenizer


def auto_device(prefer: str | None = None) -> torch.device:
    if prefer in {"cuda", "cpu", "mps"}:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg: TrainConfig = load_train_config(args.config)
    device = auto_device(cfg.device)
    torch.manual_seed(cfg.seed)
    logger.info("Loaded config from %s", args.config)
    logger.info("Device: %s", device)

    # tokenizer
    tok = BPETokenizer.load(cfg.tokenizer_path)
    logger.info("Tokenizer: vocab_size=%d from %s", len(tok.vocab), cfg.tokenizer_path)

    # datasets
    train_ds = CausalLMDataset(cfg.train_ids, DataConfig(cfg.train_ids, seq_len=cfg.seq_len))
    val_ds = CausalLMDataset(cfg.val_ids, DataConfig(cfg.val_ids, seq_len=cfg.seq_len))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    logger.info(
        "Datasets: train_tokens=%d, val_tokens=%d, seq_len=%d, batch_size=%d",
        train_ds.N, val_ds.N, cfg.seq_len, cfg.batch_size,
    )
    logger.info(
        "Steps per epoch approx: train=%d, val=%d",
        max(1, len(train_ds) // cfg.batch_size), max(1, len(val_ds) // cfg.batch_size)
    )

    # wandb
    wandb_run = None
    if cfg.wandb_enabled and cfg.wandb_project:
        try:
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config={
                "config_path": str(args.config),
                **{k: getattr(cfg, k) for k in cfg.__dict__.keys() if not k.startswith("__")}
            })
            logger.info("wandb: initialized project=%s run=%s", cfg.wandb_project, wandb_run.name if wandb_run else "-")
        except Exception as e:
            logger.warning("wandb initialization failed: %s", e)

    # model
    model = GPTModel(GPTConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        ffn_mult=cfg.ffn_mult,
        norm=cfg.norm,
        use_rel_bias=cfg.use_rel_bias,
        tie_weights=cfg.tie_weights,
    )).to(device)
    logger.info("Model params: %d", _count_parameters(model))

    # optim
    optim = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=cfg.max_steps, eta_min=1e-6)
    logger.info("Optimizer: AdamW lr=%.2e betas=%s wd=%.1e", cfg.lr, cfg.betas, cfg.weight_decay)
    logger.info("Scheduler: CosineAnnealingLR T_max=%d", cfg.max_steps)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    logger.info("Checkpoints dir: %s", cfg.ckpt_dir)

    train_iter = cycle(train_dl)
    pbar = tqdm(range(cfg.max_steps), desc="training")
    for step in pbar:
        model.train()
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        if (step + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("step=%d train_loss=%.4f lr=%.3e", step + 1, loss.item(), lr)
            if wandb_run is not None:
                try:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "lr": lr, "step": step + 1})
                except Exception as e:
                    logger.warning("wandb.log failed: %s", e)

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = next(iter(val_dl))
                vx = vx.to(device)
                vy = vy.to(device)
                v_logits = model(vx)
                v_loss = nn.functional.cross_entropy(v_logits.reshape(-1, cfg.vocab_size), vy.reshape(-1))
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "val": f"{v_loss.item():.3f}"})
            logger.info("val: step=%d val_loss=%.4f", step + 1, v_loss.item())
            if wandb_run is not None:
                try:
                    import wandb
                    wandb.log({"val/loss": v_loss.item(), "step": step + 1})
                except Exception as e:
                    logger.warning("wandb.log failed: %s", e)

        if (step + 1) % 200 == 0:
            ckpt_path = Path(cfg.ckpt_dir) / f"step_{step+1}.pt"
            torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, ckpt_path)
            pbar.write(f"Saved checkpoint to {ckpt_path}")
            logger.info("checkpoint: %s", ckpt_path)
            if wandb_run is not None:
                try:
                    import wandb
                    wandb.save(str(ckpt_path))
                except Exception as e:
                    logger.warning("wandb.save failed: %s", e)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as e:
            logger.warning("wandb finish failed: %s", e)


if __name__ == "__main__":
    main()

