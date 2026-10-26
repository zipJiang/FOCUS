"""A model's format, sampling, and action semantics as one operational API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from focus.codec import ChatCodec, HFTokenizer, Message
from focus.generation.interfaces import prepare_generation, run_async
from focus.generation.types import GenerateResult, SamplingParams, TokenId
from focus.registry import registrable, registrar

if TYPE_CHECKING:
    from focus.generation.parsing.base import ActionParser


@dataclass(frozen=True)
class PolicyFormat[A]:
    """Explicit, immutable tokenizer/template/parser binding for one model artifact.

    Set ``continuation=True`` to resume one assistant response from its original
    token IDs. Use it with exact token-input backends and ``AppendOnly``. The codec
    renders the initial prompt; the parser independently defines the action type.
    Known model families resolve through :meth:`resolve`.
    """

    name: str
    codec: ChatCodec
    parser: ActionParser[A]
    continuation: bool = False

    def _boundary(self, messages: Sequence[Message]) -> int:
        if not self.continuation:
            return len(messages)
        boundary = next(
            (
                i
                for i, message in enumerate(messages)
                if "focus_continuation_tokens" in message
            ),
            len(messages),
        )
        for message in messages[boundary:]:
            if (
                message.get("role") != "assistant"
                or "focus_continuation_tokens" not in message
            ):
                raise ValueError(
                    "Token continuation requires uninterrupted assistant output"
                )
            tokens = message["focus_continuation_tokens"]
            if not isinstance(tokens, (tuple, list)) or any(
                type(token) is not int for token in tokens
            ):
                raise ValueError("Continuation metadata must contain integer token IDs")
            if message.get("content") != self.codec.decode(
                tokens, skip_special_tokens=True
            ):
                raise ValueError(
                    "Continuation text was edited; its token IDs are unchanged"
                )
        return boundary

    def prepare(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> PreparedPrompt:
        """Render a chat prompt, or append exact segments to its initial tokens."""
        history = deepcopy(tuple(messages))
        schemas = deepcopy(tuple(tools))
        boundary = self._boundary(history)
        tokens = self.codec.render(history[:boundary], tools=schemas)
        if self.continuation:
            tokens += tuple(
                token
                for message in history[boundary:]
                for token in message["focus_continuation_tokens"]
            )
        return PreparedPrompt(tokens, self, history, schemas)

    def validate_context(self, messages: Sequence[Message]) -> None:
        """Validate continuation metadata and uninterrupted assistant output."""
        self._boundary(messages)

    def record_reply(self, result: GenerateResult, messages: list[Message]) -> None:
        """Append an assistant reply, retaining exact IDs for continuation."""
        if self.continuation and (
            not result.exact_generation
            or result.native_tool_calls
            or result.native_output
        ):
            raise ValueError("Token continuation requires an exact text completion")
        if result.native_tool_calls is None:
            result.text = self.codec.decode(result.tokens, skip_special_tokens=True)
        message: Message = {"role": "assistant", "content": result.text}
        if result.native_tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in result.native_tool_calls
            ]
        if result.native_output:
            message["native_output"] = deepcopy(result.native_output)
        if self.continuation:
            message["focus_continuation_tokens"] = tuple(result.tokens)
        messages.append(message)

    @classmethod
    def resolve(
        cls,
        model: str,
        *,
        tokenizer: HFTokenizer | None = None,
        profile: str | None = None,
    ) -> PolicyFormat[Any]:
        from focus.generation.parsing.base import TextActionParser
        from focus.generation.parsing.hermes import HermesToolCallParser
        from focus.generation.parsing.native import NativeToolCallParser
        from focus.generation.parsing.qwen_xml import QwenXMLToolCallParser

        if profile is None:
            family = model.lower().split("/")[-1]
            if family.startswith("qwen3.5-"):
                profile = "qwen_xml"
            elif family.startswith("qwen2.5-"):
                profile = "hermes"
            else:
                raise ValueError(
                    f"Unknown model format for {model!r}; supply a profile"
                )
        parsers: dict[str, ActionParser[Any]] = {
            "qwen_xml": QwenXMLToolCallParser(),
            "hermes": HermesToolCallParser(),
            "native": NativeToolCallParser(),
            "text": TextActionParser(),
        }
        if profile not in parsers:
            raise ValueError(f"Unknown format profile {profile!r}")
        if tokenizer is None:
            if profile == "native":
                from focus.tiktoken_chat import TiktokenChatTokenizer

                tokenizer = TiktokenChatTokenizer()
            else:
                from transformers import AutoTokenizer

                tokenizer = cast(HFTokenizer, AutoTokenizer.from_pretrained(model))
        assert tokenizer is not None
        return cls(f"{model}:{profile}", ChatCodec(tokenizer), parsers[profile])


@dataclass(frozen=True)
class PreparedPrompt:
    """A branch-local prompt snapshot. Tokens are estimates for native policies."""

    tokens: tuple[TokenId, ...]
    format: PolicyFormat[Any]
    _messages: tuple[Message, ...]
    _tools: tuple[dict[str, Any], ...]

    @property
    def messages(self) -> list[Message]:
        return deepcopy(list(self._messages))

    @property
    def tools(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._tools))


Completion = Callable[
    [Sequence[TokenId], SamplingParams | None], Awaitable[GenerateResult]
]


@registrable(slot="policy")
class Policy[A]:
    """Own the model identity and every operation a rollout asks of that model.

    Backends implement ``agenerate_tokens`` or, for corporate APIs, ``agenerate``.
    A custom token policy may instead supply a completion callable. No codec or parser
    is independently selected by the runner. Remote vLLM is still token-native.
    """

    provides_exact_tokens = True
    provides_native_tool_calls = False

    def __init__(
        self,
        *,
        format: PolicyFormat[A],
        model: str = "custom",
        served_model: str | None = None,
        version: str = "policy",
        default_params: SamplingParams | None = None,
        complete: Completion | None = None,
        exact: bool = True,
        native: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if format.parser.requires_native_channel and not native:
            raise ValueError("This format requires a structured tool-call channel")
        if format.continuation and (not exact or native):
            raise ValueError("Token continuation requires an exact token-input policy")
        self._format = format
        self._model = model
        self._served_model = served_model or model
        self._version = version
        self._complete = complete
        self.default_params = deepcopy(default_params or SamplingParams())
        self.provides_exact_tokens = exact
        self.provides_native_tool_calls = native

    @property
    def format(self) -> PolicyFormat[A]:
        return self._format

    @property
    def model(self) -> str:
        return self._model

    @property
    def served_model(self) -> str:
        return self._served_model

    @property
    def version(self) -> str:
        return self._version

    @property
    def trainable(self) -> bool:
        return self.provides_exact_tokens and self.format.parser.trainable

    def prepare(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> PreparedPrompt:
        return self.format.prepare(messages, tools)

    def _check_prompt(self, prompt: PreparedPrompt) -> None:
        if prompt.format is not self.format:
            raise ValueError("Prepared prompt belongs to a different policy format")

    def validate_context(self, messages: Sequence[Message]) -> None:
        """Reject unusable edited history before commit, without making a request.

        The format checks exact continuation history. Native policies also enforce
        their tool-call/response contract.
        """
        self.format.validate_context(messages)

    async def agenerate(
        self,
        prompt: PreparedPrompt,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self._check_prompt(prompt)
        return await self.agenerate_tokens(prompt.tokens, sampling_params)

    def generate(
        self,
        prompt: PreparedPrompt,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        return run_async(self.agenerate(prompt, sampling_params))

    async def agenerate_tokens(
        self,
        tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """Explicit token-input path for exact rollouts, replay and scoring."""
        if self._complete is None:
            raise NotImplementedError("This policy does not support token input")
        prefix, params = self._prepare_generation(tokens, sampling_params)
        result = await self._complete(prefix, params)
        result.exact_generation &= self.provides_exact_tokens
        return self._validate_generation(result, params)

    def generate_tokens(
        self,
        tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        return run_async(self.agenerate_tokens(tokens, sampling_params))

    def decode(
        self, tokens: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self.format.codec.decode(tokens, skip_special_tokens=skip_special_tokens)

    def parse(self, result: GenerateResult) -> A:
        return self.format.parser.parse(result)

    def is_exact(self, result: GenerateResult) -> bool:
        """Per-call provenance. Replay policies can mix inexact targets and live folds.

        ``trainable`` is the conservative capability used before a search starts;
        this method records what actually happened on a particular call.
        """
        return result.exact_generation and self.format.parser.trainable

    def record_reply(self, result: GenerateResult, messages: list[Message]) -> None:
        self.format.record_reply(result, messages)

    def _prepare_generation(
        self,
        tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None,
    ) -> tuple[tuple[TokenId, ...], SamplingParams]:
        prefix, params = prepare_generation(
            self.default_params, tokens, sampling_params
        )
        if self.format.continuation:
            limit = params.max_tokens
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("max_tokens must be a positive integer")
        return prefix, params

    def _validate_generation(
        self,
        result: GenerateResult,
        params: SamplingParams,
    ) -> GenerateResult:
        if self.format.continuation:
            if not result.exact_generation:
                raise ValueError(
                    "Token continuation requires exact generated token IDs"
                )
            limit = params.max_tokens
            assert limit is not None
            if len(result.tokens) > limit:
                raise ValueError(
                    f"Generation exceeded the requested limit of {limit} tokens"
                )
            if result.stop_reason == "length" and len(result.tokens) != limit:
                raise ValueError("Generation stopped before the requested token limit")
        return result

    def startup_check(self) -> None:
        """Optional readiness check."""

    def close(self) -> None:
        """Release policy-owned resources."""

    async def aclose(self) -> None:
        """Async lifecycle hook for policies that own an async client."""
        self.close()


register_policy = registrar(Policy)
