"""The critic interface, its function adapters, and the vLLM pooling backend."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from focus import (
    Critic,
    VLLMCritic,
)
from tests.helpers import rollout


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    """A duck-typed httpx.AsyncClient: records posts, returns a canned response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.calls.append((url, json))
        return FakeResponse(self._payload)

    async def aclose(self) -> None:
        self.closed = True


# classify/LAST -> a scalar list; ALL/per-token -> nested per-token vectors.
POOLING = {
    "data": [
        {"index": 0, "data": [1.5]},
        {"index": 1, "data": [[0.1, 2.0], [0.3, 4.0]]},
    ]
}


def _critic(payload: dict[str, Any], **kw: Any) -> tuple[VLLMCritic, FakeClient]:
    client = FakeClient(payload)
    critic = VLLMCritic(base_url="http://h:8000", model="m", client=client, **kw)
    return critic, client


async def test_a_constant_critic_predicts_its_value_everywhere() -> None:
    critic = Critic.constant(0.5, timeout=3.0)
    assert await critic.predict(["a", "b", "c"]) == [0.5, 0.5, 0.5]
    assert critic.timeout == 3.0


async def test_from_fn_accepts_sync_and_async_functions_and_a_context() -> None:
    def lengths(contexts):
        return [float(len(c)) for c in contexts]

    async def halves(contexts):
        return [len(c) / 2 for c in contexts]

    assert await Critic.from_fn(lengths).predict(["ab"]) == [2.0]
    assert await Critic.from_fn(halves).predict(["ab"]) == [1.0]
    state = replace(rollout(), messages=({"role": "user", "content": "hi"},))
    assert "hi" in Critic.constant(0.0).context(state)
    custom = Critic.from_fn(lengths, context=lambda rs: "fixed")
    assert custom.context(state) == "fixed"


async def test_vllm_predictions_reduce_and_keep_request_order() -> None:
    critic, client = _critic(POOLING)
    # a flat vector is reduced by "last"; a per-token nesting by "last" of "last"
    assert await critic.predict(["a", "b"]) == [1.5, 4.0]
    # both contexts were scored in ONE /pooling request, input order preserved
    assert client.calls == [("/pooling", {"model": "m", "input": ["a", "b"]})]


async def test_vllm_response_is_reordered_by_index() -> None:
    payload = {"data": [{"index": 1, "data": [9.0]}, {"index": 0, "data": [1.0]}]}
    critic, _ = _critic(payload)
    assert await critic.predict(["first", "second"]) == [1.0, 9.0]


async def test_vllm_reduce_strategies() -> None:
    payload = {"data": [{"index": 0, "data": [1.0, 2.0, 3.0]}]}
    cases = (("last", 3.0), ("first", 1.0), ("sum", 6.0), ("mean", 2.0))
    for reduce, expected in cases:
        critic, _ = _critic(payload, reduce=reduce)
        assert await critic.predict(["x"]) == [expected]


async def test_vllm_closes_its_client() -> None:
    critic, client = _critic(POOLING)
    await critic.aclose()
    assert client.closed


def test_bad_reduce_rejected() -> None:
    with pytest.raises(ValueError, match="reduce"):
        VLLMCritic(base_url="http://h", model="m", reduce="median")
