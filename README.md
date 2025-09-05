# LLM from scratch

Projet éducatif pour entraîner un petit LLM en Python 3.11, basé sur:
- Tokenizer BPE minimal (option “from scratch”)
- Transformer causal avec attention locale + biais positionnel relatif
- GLU-FFN avec SiLU, pré-norm
- Entraînement sur Mac (MPS) ou CPU/GPU selon disponibilité

Attention: ce repo vise la clarté; ce n’est pas une implémentation ultra-optimisée.

## Prérequis
- Python 3.11 (préféré)
- macOS Apple Silicon (MPS) recommandé (M4 Pro 48GB OK)

## Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch détectera automatiquement MPS si disponible.

## Données de démo (Tiny Shakespeare)
```bash
python scripts/prepare_tinyshakespeare.py --out_dir data
```

## Tokenizer (BPE minimal)
Entraînement et sauvegarde du tokenizer:
```bash
python scripts/train_tokenizer.py --input data/tiny_shakespeare.txt --out artifacts/tokenizer.json --vocab_size 8000
```

## Préparation des IDs (tokenisation)
```bash
python - <<'PY'
from pathlib import Path
import torch
from llm_scratch.tokenizer.bpe import BPETokenizer

inp = Path('data/tiny_shakespeare.txt').read_text(encoding='utf-8')
tok = BPETokenizer.load('artifacts/tokenizer.json')
ids = tok.encode(inp, add_bos=True)
# split 90/10
t = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(t))
train, val = t[:n], t[n:]
Path('data').mkdir(exist_ok=True)
torch.save(train, 'data/train_ids.pt')
torch.save(val, 'data/val_ids.pt')
print('Saved token ids to data/train_ids.pt and data/val_ids.pt')
PY
```

## Entraînement
Adapter configs/base.yaml au besoin puis:
```bash
python scripts/train.py --config configs/base.yaml
```

### Suivi Weights & Biases (optionnel)
- Dépendance ajoutée: `wandb`
- Activer dans `configs/base.yaml`:
```yaml
wandb_enabled: true
wandb_project: llm-scratch
wandb_run_name: "run-local"
```
- Connectez-vous une fois:
```bash
wandb login
```
- Lancement:
```bash
python scripts/train.py --config configs/base.yaml
```
Les métriques `train/loss`, `val/loss`, `lr` et les checkpoints sont loggés.

## Génération
```bash
python scripts/generate.py --ckpt artifacts/checkpoints/step_1000.pt --tokenizer artifacts/tokenizer.json --prompt "To be, or not to be" --max_new_tokens 100
```

## Notes
- L’attention locale réduit le coût mémoire/temps vs dense. Pour de très longues séquences, augmentez `max_seq_len` avec prudence.
- Le tokenizer BPE ici est volontairement simple. Pour la production, envisagez `sentencepiece` (Unigram) ou `tokenizers`.
- Activez RMSNorm via `norm: rmsnorm` dans la config si souhaité.

