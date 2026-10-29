"""Prepare a small MATH algebra training set for the Slime tutorial."""
import argparse
import json
from pathlib import Path

from math_quickstart import parse_answer


def write_dataset(rows, tokenizer, output, *, count=64):
    if count < 1:
        raise ValueError("Choose at least one problem.")
    selected = []
    for row in rows:
        messages = [
            {"role": "system", "content": r"Solve the problem step by step. Put your final answer in \boxed{}."},
            {"role": "user", "content": row["problem"]},
        ]
        prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(prefix) > 4096 or not parse_answer(row["solution"]):
            continue
        selected.append({"problem": row["problem"], "solution": row["solution"]})
        if len(selected) == count:
            break

    if len(selected) < count:
        raise ValueError(f"Requested {count} usable problems, but found {len(selected)}.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    print(f"Wrote {len(selected)} problems to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model directory or Hugging Face model ID.")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file.")
    parser.add_argument("--count", type=int, default=64, help="Number of problems (default: 64).")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    write_dataset(rows, tokenizer, args.output, count=args.count)


if __name__ == "__main__":
    main()
