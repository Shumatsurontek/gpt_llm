# Download Tiny Shakespeare dataset
from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out_path = Path(args.out_dir) / "tiny_shakespeare.txt"
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print("Done.")


if __name__ == "__main__":
    main()

