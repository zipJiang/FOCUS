"""The critic :math:`V_\\phi`: predicted remaining return at a checkpoint.

A :class:`Critic` turns checkpoint contexts into value predictions. Search scores every
checkpoint it creates; selective allocation compares those predictions with observed
outcomes, and preparation exports each scored context with its branch-mean target.

Subclass :class:`Critic` and implement :meth:`~Critic.predict`, or wrap a function with
:meth:`Critic.from_fn`. :meth:`Critic.constant` supplies a uniform prior for a first
collection round. :class:`VLLMCritic` serves a value model over vLLM's ``/pooling``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from focus.execution.pool import Owner
    from focus.harness.rollout import AnyRollout

type PredictFn = Callable[[Sequence[str]], Sequence[float] | Awaitable[Sequence[float]]]


class Critic(ABC):
    """Predict the remaining return of checkpoint contexts, one float per context."""

    #: Seconds to wait for one prediction batch; ``None`` waits as long as it takes.
    #: A timeout counts as a scoring failure: the rollouts are kept, the estimate is
    #: missing.
    timeout: float | None = None
    #: Renders a checkpoint as the critic's input; ``None`` uses the transcript.
    render: Callable[[AnyRollout], str] | None = None

    @abstractmethod
    async def predict(self, contexts: Sequence[str]) -> Sequence[float]:
        """Predict the remaining return for each context, in order."""

    def context(self, rollout: AnyRollout) -> str:
        """Render a checkpoint as the critic's input.

        Uses :attr:`render` when set, otherwise the ``role: content`` transcript.
        Critic training records carry the same text.
        """
        if self.render is not None:
            return self.render(rollout)
        from focus.harness.transcript import format_transcript

        return format_transcript(rollout.messages)

    async def predict_owned(
        self, contexts: Sequence[str], owner: Owner
    ) -> Sequence[float]:
        """Predict on behalf of ``owner``; remote critics account the request to it.

        The default ignores ``owner`` and calls :meth:`predict`.
        """
        return await self.predict(contexts)

    async def aclose(self) -> None:
        """Release clients; the default holds none."""
        return None

    @staticmethod
    def constant(
        value: float,
        *,
        context: Callable[[AnyRollout], str] | None = None,
        timeout: float | None = None,
    ) -> Critic:
        """Predict ``value`` everywhere -- the prior of a critic not yet trained.

        ``context`` renders the checkpoint contexts exported as critic inputs.
        """
        return _FunctionCritic(
            lambda contexts: [value] * len(contexts), timeout, context
        )

    @staticmethod
    def from_fn(
        fn: PredictFn,
        *,
        context: Callable[[AnyRollout], str] | None = None,
        timeout: float | None = None,
    ) -> Critic:
        """Wrap a batch function, synchronous or asynchronous, as a critic."""
        return _FunctionCritic(fn, timeout, context)


class _FunctionCritic(Critic):
    def __init__(
        self,
        fn: PredictFn,
        timeout: float | None,
        context: Callable[[AnyRollout], str] | None = None,
    ) -> None:
        self._fn = fn
        self.timeout = timeout
        self.render = context

    async def predict(self, contexts: Sequence[str]) -> Sequence[float]:
        result = self._fn(contexts)
        if inspect.isawaitable(result):
            result = await result
        return result


#: How to fold a pooling output into one scalar.
_REDUCERS: dict[str, Callable[[list[float]], float]] = {
    "last": lambda xs: xs[-1],
    "first": lambda xs: xs[0],
    "sum": sum,
    "mean": lambda xs: sum(xs) / len(xs),
}


def _flatten(raw: Any) -> list[float]:
    """Flatten an arbitrarily nested pooling payload to floats, left to right."""
    if isinstance(raw, (int, float)):
        return [float(raw)]
    out: list[float] = []
    for x in raw:
        out.extend(_flatten(x))
    return out


class VLLMCritic(Critic):
    """A value model served by vLLM as a pooling model, over ``POST /pooling``.

    vLLM mounts ``/pooling`` at the server root, not under ``/v1``. Each input yields a
    scalar vector (LAST pooling) or a per-token nesting (ALL pooling); ``reduce`` folds
    it to one prediction.

        critic = VLLMCritic(base_url="http://host:8000", model="<value model>")

    ``context`` renders each checkpoint as the model's input text.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        reduce: str = "last",
        http_timeout: float = 60.0,
        timeout: float | None = None,
        context: Callable[[AnyRollout], str] | None = None,
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
        self._http_timeout = http_timeout
        self.timeout = timeout
        self.render = context
        self._client = client  # injectable (duck-typed httpx.AsyncClient) for tests

    def _headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers(),
                timeout=self._http_timeout,
            )
        return self._client

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        resp = await self._http().post(
            "/pooling", json={"model": self._model, "input": list(contexts)}
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [self._reduce_one(item["data"]) for item in data]

    def _reduce_one(self, raw: Any) -> float:
        reduce = _REDUCERS[self._reduce]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return reduce([reduce(_flatten(token)) for token in raw])
        flat = _flatten(raw)
        return reduce(flat) if flat else 0.0

    async def aclose(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "aclose", None)
            if close is not None:
                await close()


__all__ = ["Critic", "VLLMCritic"]
