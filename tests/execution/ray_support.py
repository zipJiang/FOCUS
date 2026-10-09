"""Serializable backend factory used in real process tests."""

import asyncio

from focus.generation import GenerateResult, Policy
from tests.helpers import test_format


def make_policy():
    async def complete(tokens, params):
        await asyncio.sleep(0.1 if params is None else params.temperature)
        return GenerateResult(text="ok", tokens=(111, 107), logprobs=[-0.1, -0.1])

    return Policy(format=test_format(), complete=complete)


def worker(name, namespace, count):
    from focus.execution import InferenceRequest
    from focus.execution.ray import RayInferenceExecutor

    async def run():
        executor = RayInferenceExecutor(name, namespace)
        try:
            return await asyncio.gather(
                *[
                    executor.submit(
                        InferenceRequest("custom:policy:test", "test", (1,))
                    )
                    for _ in range(count)
                ]
            )
        finally:
            await executor.aclose()

    return asyncio.run(run())
