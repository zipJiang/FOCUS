"""Reward model backed by a vLLM pooling model's ``POST /pooling`` endpoint.

vLLM serves reward models as pooling models. Unlike the generation backends (which use
the OpenAI SDK's ``/v1/completions``), ``/pooling`` is not an OpenAI-typed endpoint, so
this backend speaks raw HTTP over ``httpx``. That also makes the base URL honest: vLLM
mounts ``/pooling`` at the server **root**, not under ``/v1``.

Each input yields a ``data`` payload: a scalar reward vector (classify / LAST pooling),
or a per-token nesting (ALL pooling). :meth:`_reduce_to_result` folds it to the sequence
scalar (``reduce`` picks how) and surfaces per-token values when the payload is nested.

    # a served reward model: vllm serve <model> --runner pooling   (on host:8000)
    rm = VLLMRewardModel(base_url="http://host:8000", model="<model>")
    rm.startup_check()
    result = rm.score("<a formatted prompt+response>")   # result.score is a float

The standard installation includes the HTTP client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from focus.reward.interfaces import AsyncRewardModel
from focus.reward.types import RewardResult

#: How to fold a list of floats into one scalar.
_REDUCERS: dict[str, Callable[[list[float]], float]] = {
    "last": lambda xs: xs[-1],
    "first": lambda xs: xs[0],
    "sum": sum,
    "mean": lambda xs: sum(xs) / len(xs),
}


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("VLLMRewardModel needs httpx; install httpx") from exc
    return httpx


def _flatten(raw: Any) -> list[float]:
    """Flatten an arbitrarily nested pooling payload to floats, left-to-right."""

    if isinstance(raw, (int, float)):
        return [float(raw)]
    out: list[float] = []
    for x in raw:
        out.extend(_flatten(x))
    return out


class VLLMRewardModel(AsyncRewardModel):
    """Score serialized contexts against a vLLM reward model over ``/pooling``."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        reduce: str = "last",
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        if reduce not in _REDUCERS:
            raise ValueError(
                f"reduce must be one of {sorted(_REDUCERS)}; got {reduce!r}"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._reduce = reduce
        self._timeout = timeout
        self._client = client  # injectable (duck-typed httpx.AsyncClient) for tests

    def _headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    def _http(self) -> Any:
        if self._client is None:
            httpx = _import_httpx()
            self._client = httpx.AsyncClient(
                base_url=self._base_url, headers=self._headers(), timeout=self._timeout
            )
        return self._client

    async def ascore_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        resp = await self._http().post(
            "/pooling", json={"model": self._model, "input": list(contexts)}
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [self._reduce_to_result(item["data"]) for item in data]

    async def ascore(self, context: str) -> RewardResult:
        return (await self.ascore_batch([context]))[0]

    def _reduce_to_result(self, raw: Any) -> RewardResult:
        """Fold one input's ``data`` payload into a :class:`RewardResult`."""

        reduce = _REDUCERS[self._reduce]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            # per-token (ALL pooling): reduce each token's vector, keep as token_scores
            per_token = [reduce(_flatten(tok)) for tok in raw]
            return RewardResult(score=reduce(per_token), token_scores=tuple(per_token))
        flat = _flatten(raw)
        return RewardResult(score=reduce(flat) if flat else 0.0)

    def startup_check(self) -> None:
        httpx = _import_httpx()
        try:
            with httpx.Client(
                base_url=self._base_url, headers=self._headers(), timeout=self._timeout
            ) as client:
                resp = client.post(
                    "/pooling", json={"model": self._model, "input": "ping"}
                )
                resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"reward model not reachable at {self._base_url}/pooling: {exc}"
            ) from exc

    def close(self) -> None:
        # httpx.AsyncClient exposes ``aclose`` (async) not ``close``; a sync teardown
        # only closes a sync client, so an async one is left for GC (like generation).
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = ["VLLMRewardModel"]
