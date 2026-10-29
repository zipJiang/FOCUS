"""MATH collection with 256-token edges and exact, uncompacted prefixes.

Run with --offline for a scripted check, or --model and --base-url for vLLM.
Requires FOCUS and math-verify[antlr4_13_2]==0.9.0.
Live runs connect to a vLLM server.
"""
import argparse
import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from math_verify import LatexExtractionConfig, parse, verify

from focus import Agent, Env, Segments, StepResult
from focus.backends import (
    ChatCodec, GenerateResult, Policy, PolicyFormat, SamplingParams,
    TokenChunk, TokenChunkParser, VLLMPolicy,
)

CHUNK_TOKENS = 256
MAX_CHUNKS = 16


def boxed_expression(text):
    """Extract the last boxed answer, including nested LaTeX braces."""
    start = text.rfind(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:i]
    return ""


def parse_answer(text):
    expression = boxed_expression(text)
    if not expression:
        return []
    return parse(
        f"${expression}$", extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
    )


@dataclass(frozen=True)
class MathState:
    tokens: tuple[int, ...] = ()
    chunks: int = 0


class MathEnv(Env[MathState, TokenChunk]):
    def __init__(self, solution: str, decode_tokens: Callable[[Sequence[int]], str]):
        self.gold = parse_answer(solution)
        if not self.gold:
            raise ValueError("The reference solution needs a parseable \\boxed{answer}")
        self.decode_tokens = decode_tokens

    async def reset(self) -> MathState:
        return MathState()

    async def step(
        self, state: MathState, action: TokenChunk, *, workspace=None,
    ) -> StepResult[MathState]:
        next_state = MathState(state.tokens + action.tokens, state.chunks + 1)
        correct = False
        if action.finished:
            # Decode the complete token sequence; a character can span chunks.
            answer = parse_answer(self.decode_tokens(next_state.tokens))
            correct = bool(answer and verify(self.gold, answer))
        return StepResult(
            next_state=next_state, done=action.finished,
            reward_outcome=float(correct),
        )


class CharacterTokenizer:
    """Offline fixture only: one character per token."""
    chat_template = "offline"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, conversation, *, add_generation_prompt, **kwargs):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in conversation)
        return self.encode(text + ("assistant: " if add_generation_prompt else ""))


class ScriptedCompletion:
    """Two segments per attempt; alternate correct and incorrect final answers."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0
        self.endings = 0

    async def __call__(self, prefix, params):
        self.calls += 1
        assert params.max_tokens == CHUNK_TOKENS
        text = self.tokenizer.decode(prefix)
        if text.endswith("assistant: "):
            output = "Subtract seven, then divide by three. ".ljust(CHUNK_TOKENS)
            reason = "length"
        else:
            self.endings += 1
            output = r"\boxed{5}" if self.endings % 2 else r"\boxed{4}"
            reason = "stop"
        tokens = tuple(self.tokenizer.encode(output))
        return GenerateResult(
            prefix_tokens=tuple(prefix), tokens=tokens, text=output,
            stop_reason=reason, logprobs=[-0.5] * len(tokens),
        )


def make_agent(policy, solution, decode_tokens, *, execution=None):
    return Agent(
        policy=policy, env=MathEnv(solution, decode_tokens),
        system_prompt=r"Solve the problem step by step. Put your final answer in \boxed{}.",
        context=Segments(CHUNK_TOKENS),
        sampling_params=SamplingParams(temperature=0.7, logprobs=0),
        max_steps=MAX_CHUNKS, execution=execution,
    )


async def rollout(policy, tokenizer, problem, solution):
    agent = make_agent(policy, solution, tokenizer.decode)
    async with agent:
        root = await agent.start(problem)
        checkpoint = await agent.advance(root)
        print(f"First segment: {len(checkpoint.state.tokens)} tokens")
        if checkpoint.done:
            endpoints = [checkpoint]
        else:
            # Both branches inherit the same context and remaining horizon.
            left = await agent.run(checkpoint.fork())
            right = await agent.run(checkpoint.fork())
            endpoints = [left, right]
        for i, endpoint in enumerate(endpoints, 1):
            print(f"Branch {i}: chunks={endpoint.state.chunks}, "
                  f"reward={endpoint.reward_outcome:g}, truncated={endpoint.truncated}")
            print(tokenizer.decode(endpoint.state.tokens, skip_special_tokens=True))
    return checkpoint, endpoints


async def main(args):
    problem = "Solve for x: 3x + 7 = 22."
    solution = r"Subtract 7 and divide by 3 to get \boxed{5}."
    if args.problem_file:
        row = json.loads(Path(args.problem_file).read_text())
        problem, solution = row["problem"], row["solution"]

    if args.offline:
        tokenizer = CharacterTokenizer()
        format = PolicyFormat(
            "math-chunks", ChatCodec(tokenizer), TokenChunkParser(),
        )
        policy = Policy(
            format=format, model="scripted", complete=ScriptedCompletion(tokenizer),
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        format = PolicyFormat(
            "math-chunks", ChatCodec(tokenizer), TokenChunkParser(),
        )
        policy = VLLMPolicy(
            model=args.model, format=format,
            base_url=args.base_url, api_key=args.api_key,
        )

    try:
        await rollout(policy, tokenizer, problem, solution)
    finally:
        await policy.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--problem-file", help="One MATH JSON record with problem and solution")
    parser.add_argument("--model", help="Actor model/tokenizer path or Hugging Face ID")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()
    if not args.offline and not args.model:
        parser.error("Supply --model, or use --offline for the scripted example")
    if args.offline and args.problem_file:
        parser.error("The offline fixture solves only the built-in algebra problem")
    asyncio.run(main(args))
