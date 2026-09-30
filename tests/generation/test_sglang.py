"""Standalone SGLang native generation backend."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from focus.generation import Policy, SamplingParams, SGLangPolicy
from focus.registry import build
from tests.helpers import test_format


def _output() -> dict[str, Any]:
    return {
        "text": " hi",
        "meta_info": {
            "finish_reason": {"type": "stop", "matched": 151645},
            "output_token_logprobs": [[-0.1, 11], [-0.4, 22]],
        },
    }


class Response:
    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output = _output() if output is None else output
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> Any:
        return self.output


class SyncClient:
    instances: list[SyncClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        type(self).instances.append(self)

    def post(self, url: str, *, json: dict[str, Any]) -> Response:
        self.calls.append((url, json))
        return Response()

    def close(self) -> None:
        self.closed = True


class AsyncClient:
    instances: list[AsyncClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        type(self).instances.append(self)

    async def post(self, url: str, *, json: dict[str, Any]) -> Response:
        self.calls.append((url, json))
        return Response()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    SyncClient.instances.clear()
    AsyncClient.instances.clear()
    module = SimpleNamespace(Client=SyncClient, AsyncClient=AsyncClient)
    monkeypatch.setattr("focus.generation.sglang._import_httpx", lambda: module)


def make_policy(**kwargs: Any) -> SGLangPolicy:
    return SGLangPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://router:30000",
        **kwargs,
    )


def test_sync_generation_uses_native_generate_and_exact_ids() -> None:
    policy = make_policy(
        api_key="secret",
        default_params=SamplingParams(
            max_tokens=16, temperature=0.8, top_p=0.9, top_k=40, seed=1729
        ),
    )
    result = policy.generate_tokens(
        (5, 6), SamplingParams(max_tokens=4, temperature=None, top_p=None)
    )

    assert result.tokens == (11, 22)
    assert result.logprobs == [-0.1, -0.4]
    assert result.prefix_tokens == (5, 6)
    assert result.text == " hi"
    assert result.matched_stop == 151645
    client = SyncClient.instances[0]
    assert client.kwargs == {
        "headers": {"Authorization": "Bearer secret"},
        "timeout": None,
    }
    url, payload = client.calls[0]
    assert url == "http://router:30000/generate"
    assert payload == {
        "input_ids": [5, 6],
        "sampling_params": {
            "max_new_tokens": 4,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "sampling_seed": 1729,
        },
        "return_logprob": True,
    }


def test_async_generation_and_lifecycle() -> None:
    async def exercise() -> None:
        policy = make_policy(headers={"X-Test": "yes"}, timeout=123.0)
        result = await policy.agenerate_tokens((1, 2))
        assert result.tokens == (11, 22)
        client = AsyncClient.instances[0]
        assert client.kwargs == {"headers": {"X-Test": "yes"}, "timeout": 123.0}
        assert client.calls[0][0] == "http://router:30000/generate"
        await policy.aclose()
        assert client.closed
        assert SyncClient.instances[0].closed

    asyncio.run(exercise())


def test_prompt_and_top_logprobs_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _output()
    output["meta_info"].update(
        {
            "input_token_logprobs": [[None, 5], [-0.3, 6]],
            "input_top_logprobs": [
                None,
                [[-0.3, 6], [-1.2, 8]],
            ],
            "output_top_logprobs": [
                [[-0.1, 11], [-2.0, 13]],
                [[-0.4, 22]],
            ],
        }
    )

    class Client(SyncClient):
        def post(self, url: str, *, json: dict[str, Any]) -> Response:
            self.calls.append((url, json))
            return Response(output)

    monkeypatch.setattr(
        "focus.generation.sglang._import_httpx",
        lambda: SimpleNamespace(Client=Client, AsyncClient=AsyncClient),
    )
    policy = make_policy()
    result = policy.generate_tokens(
        (5, 6), SamplingParams(logprobs=2, prompt_logprobs=2)
    )

    assert result.prompt_logprobs == [None, -0.3]
    assert result.prompt_top_logprobs == [
        None,
        {"token_id:6": -0.3, "token_id:8": -1.2},
    ]
    assert result.top_logprobs == [
        {"token_id:11": -0.1, "token_id:13": -2.0},
        {"token_id:22": -0.4},
    ]
    payload = Client.instances[-1].calls[0][1]
    assert payload["logprob_start_len"] == 0
    assert payload["top_logprobs_num"] == 2


def test_complete_generate_url_is_not_duplicated() -> None:
    policy = SGLangPolicy(
        model="m",
        format=test_format(),
        base_url="http://router/generate/",
    )
    assert policy.url == "http://router/generate"


def test_request_failure_has_actionable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client(SyncClient):
        def post(self, url: str, *, json: dict[str, Any]) -> Response:
            del url, json
            raise TimeoutError("slow")

    monkeypatch.setattr(
        "focus.generation.sglang._import_httpx",
        lambda: SimpleNamespace(Client=Client, AsyncClient=AsyncClient),
    )
    policy = make_policy()
    with pytest.raises(
        RuntimeError,
        match=r"SGLang /generate request failed.*prompt_tokens=3, max_tokens=9",
    ):
        policy.generate_tokens((1, 2, 3), SamplingParams(max_tokens=9))


def test_registered_under_sglang_name() -> None:
    policy = build(
        Policy,
        {
            "name": "sglang",
            "model": "m",
            "format": test_format(),
            "base_url": "http://router:30000",
        },
    )
    assert isinstance(policy, SGLangPolicy)
