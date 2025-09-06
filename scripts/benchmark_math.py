# Benchmark GSM8K and MATH (subset) with zero-shot prompting
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import Tuple, Optional

import torch

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_scratch.model.modeling_gpt import GPTModel, GPTConfig
from llm_scratch.tokenizer.bpe import BPETokenizer
from llm_scratch.utils.sampling import top_k_top_p_filtering

from datasets import load_dataset  # lazy import


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_answer_gsm8k(gt: str) -> str:
    # Ground-truth in GSM8K often has '#### <answer>' at the end
    m = re.search(r"####\s*(.+)$", gt.strip())
    return normalize_ans(m.group(1) if m else gt)


def extract_answer_math(gt: str) -> str:
    # MATH solutions often contain \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", gt)
    if m:
        return normalize_ans(m.group(1))
    return normalize_ans(gt)


def normalize_ans(s: str) -> str:
    s = s.strip()
    # Remove trailing punctuation and spaces
    s = s.replace(",", "").replace(" ", "")
    # Common formats like fractions or decimals preserved
    return s


def extract_from_model(text: str) -> str:
    # Try to find final number or expression at the end
    # Prefer patterns like '#### 123' or \boxed{123}
    m = re.search(r"####\s*([^\n]+)$", text.strip())
    if m:
        return normalize_ans(m.group(1))
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return normalize_ans(m.group(1))
    # Fallback: last number (int/float or simple fraction) in the text
    m = re.findall(r"(?:-?\d+\.\d+|-?\d+\/\d+|-?\d+)", text)
    if m:
        return normalize_ans(m[-1])
    return normalize_ans(text.splitlines()[-1] if text.strip() else "")


def generate(model: GPTModel, tok: BPETokenizer, prompt: str, device: torch.device, max_new_tokens: int, top_k: int, top_p: float, temperature: float, eos_id: int | None) -> str:
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(tok.encode(prompt, add_bos=True), dtype=torch.long, device=device)[None, :]
        for _ in range(max_new_tokens):
            if ids.size(1) >= model.cfg.max_seq_len:
                ids = ids[:, -model.cfg.max_seq_len:]
            logits = model(ids)[:, -1, :].squeeze(0)
            if temperature and temperature != 1.0:
                logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id[None, :]], dim=1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break
        text = tok.decode(ids.squeeze(0).tolist())
    # Remove the prompt prefix from decoded text if present
    if text.startswith(prompt):
        return text[len(prompt):]
    return text


def build_prompt(dataset: str, question: str) -> str:
    if dataset == "gsm8k":
        return (
            "You are a helpful math tutor. Solve the problem step by step.\n"
            "Provide only the final numeric answer after '####'.\n\n"
            f"Problem: {question}\n\nSolution: "
        )
    # math
    return (
        "Solve the following math problem. Show reasoning briefly.\n"
        "Return the final answer inside \\boxed{...}.\n\n"
        f"Problem: {question}\n\nSolution: "
    )


def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + " …"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1, help="Log every N examples")
    parser.add_argument("--log_chars", type=int, default=320, help="Max characters to show for prompt/gen")
    parser.add_argument("--show_prompt", action="store_true", help="Log the full prompt per example")
    parser.add_argument("--show_text", action="store_true", help="Log the raw generated text per example")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("bench")

    torch.manual_seed(args.seed)
    device = auto_device()
    logger.info("Device: %s", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"]) if "cfg" in ckpt else None
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    tok = BPETokenizer.load(args.tokenizer)
    eos_id = tok.vocab.get("<eos>")

    # Load dataset
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=args.split)
        get_q = lambda ex: ex["question"]
        get_a = lambda ex: extract_answer_gsm8k(ex["answer"])  # type: ignore[index]
    else:
        ds = load_dataset("hendrycks/math", split=args.split)
        get_q = lambda ex: ex["problem"]
        get_a = lambda ex: extract_answer_math(ex["solution"])  # type: ignore[index]

    total = 0
    correct = 0
    for idx, ex in enumerate(ds.select(range(min(args.limit, len(ds))))):
        question = get_q(ex)
        gt = get_a(ex)
        prompt = build_prompt(args.dataset, question)
        out = generate(model, tok, prompt, device, args.max_new_tokens, args.top_k, args.top_p, args.temperature, eos_id)
        pred = extract_from_model(out)
        is_ok = (pred == gt)
        correct += int(is_ok)
        total += 1
        if (total % args.log_every) == 0:
            logger.info(
                "[%s][%d] correct=%s | pred=%s | gt=%s",
                args.dataset, idx, is_ok, pred, gt,
            )
            if args.show_prompt:
                logger.info("prompt: %s", _truncate(prompt.replace("\n", " "), args.log_chars))
            if args.show_text:
                logger.info("generation: %s", _truncate(out.replace("\n", " "), args.log_chars))

    acc = 100.0 * correct / max(1, total)
    logger.info("Final %s accuracy on %d samples: %.2f%%", args.dataset, total, acc)
    print(f"{args.dataset} accuracy: {correct}/{total} = {acc:.2f}%")


if __name__ == "__main__":
    main()


