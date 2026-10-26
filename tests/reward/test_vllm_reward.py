from __future__ import annotations

from typing import Any

import pytest

from focus.reward import RewardResult, VLLMRewardModel


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

    async def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.calls.append((url, json))
        return FakeResponse(self._payload)


# classify/LAST -> a scalar list; ALL/per-token -> nested per-token vectors.
POOLING = {
    "data": [
        {"index": 0, "data": [1.5]},
        {"index": 1, "data": [[0.1, 2.0], [0.3, 4.0]]},
    ]
}


def _model(payload: dict[str, Any], **kw: Any) -> tuple[VLLMRewardModel, FakeClient]:
    client = FakeClient(payload)
    rm = VLLMRewardModel(base_url="http://h:8000", model="m", client=client, **kw)
    return rm, client


@pytest.mark.asyncio
async def test_ascore_batch_parses_reduces_and_preserves_order() -> None:
    rm, client = _model(POOLING)
    results = await rm.ascore_batch(["a", "b"])

    assert [r.score for r in results] == [1.5, 4.0]  # reduce="last"
    assert results[0].token_scores is None  # a flat vector -> no per-token scores
    assert results[1].token_scores == (2.0, 4.0)  # per-token, each reduced by "last"
    # both contexts were scored in ONE /pooling request, input order preserved
    assert client.calls == [("/pooling", {"model": "m", "input": ["a", "b"]})]


@pytest.mark.asyncio
async def test_ascore_single() -> None:
    rm, _ = _model({"data": [{"index": 0, "data": [3.25]}]})
    result = await rm.ascore("ctx")
    assert isinstance(result, RewardResult)
    assert result.score == 3.25


@pytest.mark.asyncio
async def test_response_reordered_by_index() -> None:
    # a backend may return items out of order; results follow request order via `index`
    payload = {"data": [{"index": 1, "data": [9.0]}, {"index": 0, "data": [1.0]}]}
    rm, _ = _model(payload)
    results = await rm.ascore_batch(["first", "second"])
    assert [r.score for r in results] == [1.0, 9.0]


@pytest.mark.asyncio
async def test_reduce_strategies() -> None:
    payload = {"data": [{"index": 0, "data": [1.0, 2.0, 3.0]}]}
    cases = (("last", 3.0), ("first", 1.0), ("sum", 6.0), ("mean", 2.0))
    for reduce, expected in cases:
        rm, _ = _model(payload, reduce=reduce)
        assert (await rm.ascore("x")).score == expected


def test_sync_bridge_matches_async() -> None:
    rm, _ = _model({"data": [{"index": 0, "data": [2.0]}]})
    assert rm.score("x").score == 2.0  # sync `score` bridges to `ascore` via run_async


def test_bad_reduce_rejected() -> None:
    with pytest.raises(ValueError, match="reduce"):
        VLLMRewardModel(base_url="http://h", model="m", reduce="median")
