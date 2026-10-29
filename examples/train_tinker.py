"""Train a Tinker actor with one-step asynchronous FOCUS MATH collection."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from focus.integrations import JsonlDirectory, Problem
from focus.integrations.tinker import train
from math_training import collector_for


def read_problems(path):
    """Read MATH records from one JSON object, a JSON array, or JSONL."""
    path = Path(path)
    text = path.read_text()
    rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl" else json.loads(text))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) or
        not all(isinstance(row.get(key), str) and row[key].strip() for key in ("problem", "solution"))
        for row in rows
    ):
        raise ValueError("Provide MATH records with nonempty problem and solution strings")
    return [Problem(row["problem"], row["solution"], row.get("id", i)) for i, row in enumerate(rows)]


async def main(args):
    import tinker
    problems = read_problems(args.problem_file)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    service = tinker.ServiceClient()
    status = "errored"
    try:
        training = await service.create_lora_training_client_async(base_model=args.model, rank=32)
        await train(
            training, model=args.model, problems=problems, collector_for=collector_for,
            steps=args.steps, async_depth=args.async_depth, learning_rate=args.learning_rate,
            # The model is used without tools; Segments reads each reply as a token chunk.
            profile="token_chunk",
            critic_sink=JsonlDirectory(args.critic_output),
            on_step=lambda report: print(json.dumps(asdict(report))),
        )
        checkpoint = await training.save_state_async(name=f"after-{args.steps}-updates")
        saved = await checkpoint.result_async()
        print(f"Training checkpoint: {saved.path}")
        print(f"Critic targets: {args.critic_output}")
        status = "success"
    except (asyncio.CancelledError, KeyboardInterrupt):
        status = "interrupted"
        raise
    finally:
        await service.close(status=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--problem-file", required=True, help="MATH JSON object, JSON array, or JSONL")
    parser.add_argument("--steps", type=int, default=3, help="Actor updates; cycle through the input records")
    parser.add_argument("--async-depth", type=int, choices=(0, 1), default=1,
                        help="1 overlaps next collection with training; 0 collects synchronously")
    parser.add_argument("--critic-output", default="critic-targets",
                        help="Directory for the critic records, one JSONL file per batch")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    asyncio.run(main(parser.parse_args()))
