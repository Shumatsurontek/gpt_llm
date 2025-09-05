# Generate text from a trained checkpoint
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop_on_eos", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"]) if "cfg" in ckpt else None
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tok = BPETokenizer.load(args.tokenizer)
    ids = torch.tensor(tok.encode(args.prompt, add_bos=True), dtype=torch.long, device=device)[None, :]

    eos_id = None
    if args.stop_on_eos:
        eos_id = tok.vocab.get("<eos>")

    for _ in range(args.max_new_tokens):
        if ids.size(1) >= cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len:]
        with torch.no_grad():
            logits = model(ids)[:, -1, :].squeeze(0)
            if args.temperature and args.temperature != 1.0:
                logits = logits / args.temperature
            logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id[None, :]], dim=1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    out = tok.decode(ids.squeeze(0).tolist())
    print(out)


if __name__ == "__main__":
    main()

