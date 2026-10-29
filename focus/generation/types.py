"""Shared token generation value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TokenId = int

#: What the three always-set knobs fall back to. Named rather than written inline
#: because they are read twice: as this class's field defaults, and by
#: :func:`~focus.generation._common.merge_sampling_params` filling in a
#: partial overlay that deliberately left one unset.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 1.0


@dataclass
class SamplingParams:
    """Per-call generation knobs."""

    # `None` on these three means *not overridden*, which only an overlay says: a params
    # object handed to a backend never carries one, because `merge_sampling_params`
    # fills it from the defaults it was merged onto (or from the constants above). The
    # distinction exists because every non-`None` field of an overlay wins, so an
    # overlay built to say one thing -- a phase's temperature, a fold's
    # `max_tokens` -- would otherwise also pin the sampling law to this class's defaults
    # and quietly undo whatever the backend was configured with.
    temperature: float | None = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    top_p: float | None = DEFAULT_TOP_P
    # ``None`` leaves top-k to the backend default; ``-1`` disables it (verl/vLLM/SGLang
    # convention). Backends that don't support top-k ignore it.
    top_k: int | None = None
    stop: list[str] | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    repetition_penalty: float | None = None
    # A request-local draw seed. ``None`` leaves sampling to the backend. Keeping it
    # with the other sampling controls keeps a rollout's sampling settings together
    # without subclassing a transport merely to add one wire field.
    seed: int | None = None


@dataclass(frozen=True)
class NativeToolCall:
    """One call a backend returned *as structure* rather than as text in the completion.

    The generation-layer twin of
    :class:`~focus.generation.parsing.ToolCall`, and deliberately not that
    class:
    this layer knows nothing about the harness, and a backend must be able to report
    what the API gave it without importing an env's vocabulary.
    ``arguments`` is the JSON object *text* verbatim, the same shape a
    ``<tool_call>`` body decodes to, so both roads reach ``verify_args`` unchanged.
    ``call_id`` is the API's own id for the call, kept because a re-rendered history
    has to quote it back.
    """

    name: str
    arguments: str
    call_id: str = ""


@dataclass
class GenerateResult:
    """One generator call."""

    tokens: tuple[TokenId, ...] = field(default_factory=tuple)
    prefix_tokens: tuple[TokenId, ...] = field(default_factory=tuple)
    text: str = ""
    stop_reason: str = "stop"
    # What ended a ``stop`` finish, when the backend reports it: the eos/stop token id
    # (int) or the matched stop string (str). ``None`` when unreported. Lets an env
    # distinguish a clean eos finish from a stop-string hit (e.g. verl/MemSearcher
    # treats a ``<|im_start|>`` match as an illegal stop and kills the trajectory).
    matched_stop: int | str | None = None
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[dict[str, float]] | None = None
    prompt_logprobs: list[float | None] = field(default_factory=list)
    prompt_top_logprobs: list[dict[str, float] | None] | None = None
    entropy: float = 0.0
    # Whether ``prefix_tokens`` + ``tokens`` are the model's exact generation, i.e. the
    # true token ids the LM conditioned on and emitted. Token-native backends (vLLM,
    # tinker, slime) are exact. A backend that reconstructs tokens from text it cannot
    # perfectly invert -- e.g. the OpenAI chat backend, whose server-side prompt
    # template is unknowable -- sets this ``False`` so training skips non-on-policy.
    exact_generation: bool = True
    #: The calls the backend returned through a *structured* channel, three-state:
    #: ``None`` -- this **backend** has no such channel (every token-native backend),
    #: so a parser reading it is misconfigured rather than looking at a turn that
    #: called nothing; ``()`` -- a channel-capable backend, and this turn made no call:
    #: an ordinary prose turn (the env's stray/nudge path), whether schemas were sent
    #: and declined or never sent at all (a toolless fold's reply is exactly this);
    #: non-empty -- the calls themselves. Which state a result gets is a property of
    #: the backend, never of one call -- that is what lets a fold's toolless
    #: generations flow through the same native parser as the rollout's tool turns.
    native_tool_calls: tuple[NativeToolCall, ...] | None = None

    #: Provider output items needed to replay stateless native history (e.g. reasoning).
    native_output: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        self.tokens = tuple(self.tokens)
        self.prefix_tokens = tuple(self.prefix_tokens)
        if self.native_tool_calls is not None:
            self.native_tool_calls = tuple(self.native_tool_calls)
        _validate_logprob_alignment(
            "completion",
            token_count=len(self.tokens),
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        _validate_logprob_alignment(
            "prefix",
            token_count=len(self.prefix_tokens),
            logprobs=self.prompt_logprobs,
            top_logprobs=self.prompt_top_logprobs,
        )


def _validate_logprob_alignment(
    label: str,
    *,
    token_count: int,
    logprobs: list[float] | list[float | None],
    top_logprobs: list[dict[str, float]] | list[dict[str, float] | None] | None,
) -> None:
    if logprobs and len(logprobs) != token_count:
        raise ValueError(
            f"{label} logprobs length {len(logprobs)} does not match "
            + f"{label} token count {token_count}"
        )
    if top_logprobs is not None and len(top_logprobs) != token_count:
        raise ValueError(
            f"{label} top_logprobs length {len(top_logprobs)} does not match "
            + f"{label} token count {token_count}"
        )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "GenerateResult",
    "NativeToolCall",
    "SamplingParams",
    "TokenId",
]
