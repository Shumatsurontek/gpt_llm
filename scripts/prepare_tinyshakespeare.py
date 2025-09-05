from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional
import urllib.request


def _download_url(url: str, out_file: Path) -> None:
    logging.info("Downloading %s -> %s", url, out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_file)
    logging.info("Done: %s (%.2f MB)", out_file, out_file.stat().st_size / (1024 * 1024))


def _save_lines(lines: Iterable[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for line in lines:
            if line is None:
                continue
            s = str(line).rstrip("\n")
            if s:
                f.write(s)
                f.write("\n")
    logging.info("Saved corpus to %s (%.2f MB)", out_file, out_file.stat().st_size / (1024 * 1024))


def _prepare_from_hf(dataset_path: str, name: Optional[str], split: str, text_field: Optional[str], out_file: Path) -> None:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'datasets' package is required. Install with: pip install datasets"
        ) from e

    logging.info("Loading HF dataset: %s name=%s split=%s", dataset_path, name or "-", split)
    ds = load_dataset(dataset_path, name=name, split=split, streaming=False)
    # infer text field if not provided
    if text_field is None:
        candidates = ["text", "content", "article", "document", "body"]
        text_field = next((c for c in candidates if c in ds.features), None)
        if text_field is None:
            raise ValueError(
                f"Cannot infer text field. Available: {list(ds.features.keys())}. Use --text_field."
            )
    logging.info("Using text field: %s", text_field)
    _save_lines((ex[text_field] for ex in ds), out_file)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Prepare a text corpus: Tiny Shakespeare (default), URL, or Hugging Face dataset")
    parser.add_argument("--out_dir", default="data", help="Output directory")
    parser.add_argument("--out_file", default=None, help="Output file name (defaults per source)")
    # Hugging Face options
    parser.add_argument("--hf_path", default=None, help="HF dataset path, e.g. wikitext")
    parser.add_argument("--hf_name", default=None, help="HF dataset name/config, e.g. wikitext-2-raw-v1")
    parser.add_argument("--hf_split", default="train", help="HF split (train/validation/test)")
    parser.add_argument("--text_field", default=None, help="Text field name (default: try to infer)")
    # URL option
    parser.add_argument("--url", default=None, help="Direct URL to a .txt file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_path:
        # default out file name for HF
        default_name = f"hf_{args.hf_path.replace('/', '_')}_{args.hf_split}.txt"
        out_file = out_dir / (args.out_file or default_name)
        _prepare_from_hf(args.hf_path, args.hf_name, args.hf_split, args.text_field, out_file)
        return

    if args.url:
        # default: keep remote basename
        name = args.url.rstrip("/").split("/")[-1] or "corpus.txt"
        out_file = out_dir / (args.out_file or name)
        _download_url(args.url, out_file)
        return

    # Default: Tiny Shakespeare (backward compatible)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out_file = out_dir / (args.out_file or "tiny_shakespeare.txt")
    _download_url(url, out_file)


if __name__ == "__main__":
    main()