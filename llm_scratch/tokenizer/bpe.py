# Byte-pair encoding tokenizer (minimal), training + encode/decode
# Python 3.11
# Note: This is a simple educational BPE, not byte-level GPT-2 encoder.
# It supports special tokens and persistence to JSON.

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _whitespace_tokenize(text: str) -> List[str]:
    # Simple whitespace + keep punctuation attached
    # For better robustness, you can switch to a unicode-aware char-level pre-tokenizer
    return text.strip().split()


def _get_stats(tokens: List[List[str]]) -> Counter[Tuple[str, str]]:
    stats: Counter[Tuple[str, str]] = Counter()
    for word in tokens:
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += 1
    return stats


def _merge_vocab(tokens: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    first, second = pair
    pattern = re.compile(rf"(?<!\S){re.escape(first)}\s+{re.escape(second)}(?!\S)")
    merged: List[List[str]] = []
    for word in tokens:
        if len(word) < 2:
            merged.append(word)
            continue
        s = " ".join(word)
        s = pattern.sub(f"{first}{second}", s)
        merged.append(s.split(" "))
    return merged


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BPEConfig:
    vocab_size: int = 32000
    min_pair_count: int = 2


class BPETokenizer:
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]], unk_token: str = "<unk>"):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.unk_token = unk_token

    @classmethod
    def train(cls, corpus_paths: List[str | Path], config: BPEConfig = BPEConfig()) -> "BPETokenizer":
        logger.info(
            "Training BPE: target_vocab_size=%d, min_pair_count=%d, files=%d",
            config.vocab_size,
            config.min_pair_count,
            len(corpus_paths),
        )
        # Build initial vocab from characters and words
        words: List[List[str]] = []
        token_counts: Counter[str] = Counter()
        for p in corpus_paths:
            path = Path(p)
            logger.info("Reading corpus: %s (%.2f MB)", path, path.stat().st_size / (1024 * 1024))
            text = path.read_text(encoding="utf-8")
            for tok in _whitespace_tokenize(text):
                chars = list(tok)
                # add end-of-word marker to help merges respect word boundaries
                if chars:
                    chars[-1] = chars[-1] + "</w>"
                else:
                    chars = ["</w>"]
                words.append(chars)
                for ch in chars:
                    token_counts[ch] += 1
        # initialize vocab with specials and characters
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch, _ in token_counts.most_common():
            if ch not in vocab:
                vocab[ch] = len(vocab)
        merges: List[Tuple[str, str]] = []

        def current_vocab_size():
            return len(vocab)

        logger.info(
            "Initialized vocab: specials=%d, unique_chars=%d, total=%d",
            len(SPECIAL_TOKENS),
            len(vocab) - len(SPECIAL_TOKENS),
            len(vocab),
        )

        # BPE merges until vocab_size
        merges_done = 0
        while current_vocab_size() < config.vocab_size:
            stats = _get_stats(words)
            if not stats:
                logger.info("No more pairs to merge; stopping early at vocab=%d", current_vocab_size())
                break
            (best_pair, best_count) = stats.most_common(1)[0]
            if best_count < config.min_pair_count:
                logger.info(
                    "Stopping: best pair '%s%s' occurs %d < min_pair_count=%d",
                    best_pair[0],
                    best_pair[1],
                    best_count,
                    config.min_pair_count,
                )
                break
            words = _merge_vocab(words, best_pair)
            new_token = "".join(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)
                merges.append(best_pair)
                merges_done += 1

            if merges_done % 100 == 0:
                logger.info(
                    "Progress: merges=%d, vocab=%d, last_pair='%s%s' (count=%d)",
                    merges_done,
                    current_vocab_size(),
                    best_pair[0],
                    best_pair[1],
                    best_count,
                )
        return cls(vocab=vocab, merges=merges)

    def save(self, path: str | Path):
        obj = {
            "vocab": self.vocab,
            "merges": self.merges,
            "unk_token": self.unk_token,
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Tokenizer saved: %s (vocab=%d, merges=%d)", path, len(self.vocab), len(self.merges))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in obj["merges"]]
        tok = cls(vocab=obj["vocab"], merges=merges, unk_token=obj.get("unk_token", "<unk>"))
        logger.info("Tokenizer loaded: %s (vocab=%d, merges=%d)", path, len(tok.vocab), len(tok.merges))
        return tok

    def _apply_merges(self, word: List[str]) -> List[str]:
        # Greedy apply merges learned at training
        pairs = {(a, b) for a, b in zip(word, word[1:])}
        merges_set = set(self.merges)
        while True:
            candidates = pairs & merges_set
            if not candidates:
                break
            # pick first found merge (order of merges was learned; we could keep priority)
            a, b = next(iter(candidates))
            new_token = a + b
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = {(x, y) for x, y in zip(word, word[1:])}
        return word

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos and "<bos>" in self.vocab:
            ids.append(self.vocab["<bos>"])
        for tok in _whitespace_tokenize(text):
            chars = list(tok)
            if chars:
                chars[-1] = chars[-1] + "</w>"
            else:
                chars = ["</w>"]
            merged = self._apply_merges(chars)
            for m in merged:
                ids.append(self.vocab.get(m, self.vocab[self.unk_token]))
        if add_eos and "<eos>" in self.vocab:
            ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        words: List[str] = []
        buff: List[str] = []
        for t in tokens:
            if t in SPECIAL_TOKENS:
                continue
            if t.endswith("</w>"):
                buff.append(t[:-4])
                words.append("".join(buff))
                buff = []
            else:
                buff.append(t)
        if buff:
            words.append("".join(buff))
        return " ".join(words)

