# Generate text from a trained checkpoint
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_scratch.model.modeling_gpt import GPTModel, GPTConfig
from llm_scratch.tokenizer.bpe import BPETokenizer
from llm_scratch.utils.sampling import top_k_top_p_filtering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"]) if "cfg" in ckpt else None
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tok = BPETokenizer.load(args.tokenizer)
    ids = torch.tensor(tok.encode(args.prompt, add_bos=True), dtype=torch.long, device=device)[None, :]

    for _ in range(args.max_new_tokens):
        if ids.size(1) >= cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len:]
        with torch.no_grad():
            logits = model(ids)[:, -1, :].squeeze(0)
            logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id[None, :]], dim=1)

    out = tok.decode(ids.squeeze(0).tolist())
    print(out)


if __name__ == "__main__":
    main()

