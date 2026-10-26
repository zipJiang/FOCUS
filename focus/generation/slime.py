"""Generator backed by an injected HTTP ``post`` (e.g. slime's SGLang router).

Unlike :class:`VLLMPolicy` / :class:`TinkerPolicy`, this generator
owns **no client**. The host (a slime rollout actor) hands in its own gated async
``post`` callable plus the ``/generate`` route, so the generator is a thin,
cheap-to-build wrapper with nothing live or non-picklable inside. That matters for
plug-and-play with frameworks that expose the inference engine only at rollout time.

The SGLang ``/generate`` endpoint returns, in ``meta_info.output_token_logprobs``, one
``[logprob, token_id, ...]`` pair per sampled token -- so the completion tokens and
their logprobs are aligned by construction (no re-encode / desync).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from focus.generation.policy import Policy, PolicyFormat, register_policy
from focus.generation.sglang import (
    _parse_sglang_output,
    _sglang_payload,
)
from focus.generation.types import GenerateResult, SamplingParams, TokenId

# An async HTTP POST: ``await post(url, payload[, headers=...]) -> parsed-JSON dict``.
PostFn = Callable[..., Awaitable[Mapping[str, Any]]]


@register_policy("slime")
class SlimePolicy(Policy[Any]):
    """Token generation over an injected ``post`` against an SGLang ``/generate`` route.

    :meth:`agenerate` posts against the route; sync :meth:`generate` is the
    :class:`~focus.generation.Policy` bridge over it (``asyncio.run``,
    so a worker thread with no running loop).
    """

    def __init__(
        self,
        post: PostFn,
        url: str,
        *,
        model: str,
        format: PolicyFormat[Any],
        version: str = "policy",
        default_params: SamplingParams | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._post = post
        self._url = url
        super().__init__(
            model=model, format=format, version=version, default_params=default_params
        )
        # Passed through as ``**``: a host whose ``post`` takes no ``headers`` keyword
        # (the common case) must not be called with one.
        self._post_kwargs = {"headers": dict(headers)} if headers is not None else {}

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        prefix_tuple, params = self._prepare_generation(prefix_tokens, sampling_params)
        payload = _sglang_payload(prefix_tuple, params)
        output = await self._post(self._url, payload, **self._post_kwargs)
        return self._validate_generation(
            _parse_sglang_output(output, prefix_tuple, params), params
        )


__all__ = [
    "PostFn",
    "SlimePolicy",
]
