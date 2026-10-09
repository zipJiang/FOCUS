"""Run with python -m examples.protocols --protocol clm|memex|rlm|rlm-memex.

A deterministic token backend makes execution and masking inspectable without a GPU.
Replace the Policy with your backend while retaining the same runtime/protocol wiring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from focus import (
    CLM,
    RLM,
    AdmissionPool,
    ExecutionServices,
    LocalInferenceExecutor,
    Memex,
    ResourceBudget,
    Runner,
)
from focus.codec import ChatCodec, Message
from focus.generation import GenerateResult, Policy, PolicyFormat, SamplingParams
from focus.generation.parsing import HermesToolCallParser
from focus.preparation.records import spans_of
from focus.protocols.rlm import InvocationEnv


class CharacterTokenizer:
    """Small reversible demonstration codec; use the model's tokenizer in real jobs."""

    chat_template = "demo"

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(map(ord, text))

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return "".join(map(chr, token_ids))

    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        *,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> list[int]:
        text = "".join(str(m["role"])[0] + str(m["content"]) for m in conversation)
        return self.encode(text + ("a" if add_generation_prompt else ""))


def call(name: str, **arguments: Any) -> str:
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


async def main(choice: str) -> None:
    finish = call("rlm_return", answer="Demo complete")
    if choice == "clm":
        protocol = CLM()
        outputs = [
            call(
                "context_edit",
                code=(
                    "import json\nfrom pathlib import Path\n"
                    "messages = [{'role': 'user', 'content': 'retained evidence'}]\n"
                    "Path('context.json').write_text(json.dumps(messages))"
                ),
            ),
            finish,
        ]
    elif choice == "memex":
        protocol = Memex()
        outputs = [
            call("memory_compress", indices=[1], summary="evidence archived"),
            call("memory_retrieve", index=0),
            finish,
        ]
    else:
        protocol = RLM(context=Memex()) if choice == "rlm-memex" else RLM()
        outputs = (
            [call("memory_compress", indices=[1], summary="input is external")]
            if choice == "rlm-memex"
            else []
        )
        outputs += [
            call(
                "rlm_python",
                code=(
                    "print(context[:40])\n"
                    "print(llm_query_batched(['left part', 'right part']))"
                ),
            ),
            "left result",
            "right result",
            finish,
        ]
    responses = iter(outputs)

    async def complete(
        tokens: Sequence[int], params: SamplingParams | None
    ) -> GenerateResult:
        text = next(responses)
        await asyncio.sleep(0.01)
        return GenerateResult(
            text=text, tokens=tuple(map(ord, text)), logprobs=[-0.1] * len(text)
        )

    policy = Policy(
        format=PolicyFormat(
            "demo", ChatCodec(CharacterTokenizer()), HermesToolCallParser()
        ),
        complete=complete,
    )
    async with ExecutionServices(
        inference=LocalInferenceExecutor(
            AdmissionPool(capacity=2),
            own_clients=True,
        )
    ) as services:
        runner = Runner(
            policy=policy,
            env=InvocationEnv(),
            system_prompt="Solve the task.",
            protocol=protocol,
            services=services,
            resource_budget=ResourceBudget(model_calls=16, process_cells=8),
        )
        result = await runner.run_from("External evidence for this demonstration.")
        spans = spans_of(result.turns)
        print(
            json.dumps(
                {
                    "answer": result.state,
                    "generations": len(result.turns),
                    "actor_tokens": sum(sum(s.trainable) for s in spans),
                    "usage": asdict(services.usage[result.episode_id]),
                    "remaining_budget": asdict(result.budget),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["clm", "memex", "rlm", "rlm-memex"], required=True
    )
    asyncio.run(main(parser.parse_args().protocol))
