# LLM from scratch - Training scripts
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import logging

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_scratch.tokenizer.bpe import BPETokenizer, BPEConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Text files to train tokenizer")
    parser.add_argument("--out", default="artifacts/tokenizer.json", help="Output tokenizer path")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_pair_count", type=int, default=2)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Starting tokenizer training: files=%s, out=%s, vocab_size=%d, min_pair_count=%d",
        ",".join(map(str, args.input)),
        args.out,
        args.vocab_size,
        args.min_pair_count,
    )
    tok = BPETokenizer.train(args.input, BPEConfig(vocab_size=args.vocab_size, min_pair_count=args.min_pair_count))
    tok.save(args.out)
    logging.info("Saved tokenizer to %s (vocab=%d)", args.out, len(tok.vocab))


if __name__ == "__main__":
    main()

