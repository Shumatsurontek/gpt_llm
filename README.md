# LLM-Scratch — A Minimal, Reproducible RoPE LLM (CPU/MPS First)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Educational, open-source LLM built for clarity and CPU/MPS-first training. Features:
- RoPE positional encoding (no absolute pos-emb)
- Local causal attention via PyTorch SDPA (memory-friendly)
- SwiGLU MLP, RMSNorm, pre-norm transformer blocks
- BPE tokenizer (8k vocab) with simple training pipeline
- Structured logging + optional W&B (never hard-fails)

This repo aims for readable, strongly-typed Python with small functions and clean configs. Perfect for learning, tinkering, and extending.

## Quickstart
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Prepare demo data (Tiny Shakespeare):
```bash
python scripts/prepare_tinyshakespeare.py --out_dir data
```

Train tokenizer (8k vocab):
```bash
python scripts/train_tokenizer.py --input data/tiny_shakespeare.txt \
  --out artifacts/tokenizer.json --vocab_size 8000
```

Build token IDs (train/val):
```bash
python - <<'PY'
from pathlib import Path
import torch
from llm_scratch.tokenizer.bpe import BPETokenizer
inp = Path('data/tiny_shakespeare.txt').read_text(encoding='utf-8')
tok = BPETokenizer.load('artifacts/tokenizer.json')
ids = tok.encode(inp, add_bos=True)
t = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(t))
Path('data').mkdir(exist_ok=True)
torch.save(t[:n], 'data/train_ids.pt')
torch.save(t[n:], 'data/val_ids.pt')
print('✓ wrote data/train_ids.pt and data/val_ids.pt')
PY
```

Train (default config):
```bash
python scripts/train.py --config configs/base.yaml
```

Generate:
```bash
python scripts/generate.py \
  --ckpt artifacts/checkpoints/step_1000.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "To be, or not to be" --max_new_tokens 100
```

## Generic Corpus (Hugging Face / folders / globs)
Tokenizer training with dirs/globs:
```bash
python scripts/train_tokenizer.py \
  --input /path/corpus_dir "/path/books/**/*.txt" /path/one.txt \
  --out artifacts/tokenizer.json --vocab_size 8000
```
IDs from arbitrary text:
```bash
python scripts/prepare_corpus_ids.py \
  --inputs /path/corpus_dir "/path/books/**/*.txt" \
  --tokenizer artifacts/tokenizer.json \
  --out_dir data --val_ratio 0.1 --add_bos
```

## Config Presets
- CPU light: `configs/base.cpu.yaml` (smaller model, CPU-friendly)
- Medium ~160M: `configs/base.160m.yaml` (ctx 2k)
- Large ~400M: `configs/base.400m.yaml` (ctx 2k, local_window 1k, RMSNorm)

Switch config:
```bash
python scripts/train.py --config configs/base.400m.yaml
```

## Weights & Biases (optional)
Enable in YAML or via env vars:
```yaml
wandb_enabled: true
wandb_project: llm-scratch
wandb_run_name: "run-local"
```
```bash
wandb login
python scripts/train.py --config configs/base.yaml
```

## Project Structure
```
llm_scratch/
  data/           # dataset utilities
  model/          # blocks + GPT model
  tokenizer/      # BPE
  utils/          # sampling, helpers
scripts/           # train, generate, tokenizer, data prep
configs/           # training presets
docs/              # paper.md / paper.tex
```

## Paper / Docs
- Obsidian: `docs/paper.md`
- LaTeX: `docs/paper.tex` (build with `tectonic -o docs docs/paper.tex`)

## Contributing
PRs welcome! Please:
- follow Conventional Commits (e.g., `feat(tokenizer): ...`),
- keep functions small with full type hints,
- add structured logs where it helps diagnosis.

## License
MIT — see `LICENSE` (feel free to use/modify; contributions are welcome).

## Acknowledgments
Thanks to the open-source community for making CPU/MPS-friendly LLM research practical.
