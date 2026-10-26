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

    The codec renders messages as model tokens. The parser independently defines
    the environment action. Context protocols choose which prompt to prepare.
    Known model families resolve through :meth:`resolve`.
    """

    name: str
    codec: ChatCodec
    parser: ActionParser[A]

    def prepare(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> PreparedPrompt:
        """Render the current conversation into a prompt snapshot."""
        history = deepcopy(tuple(messages))
        schemas = deepcopy(tuple(tools))
        return PreparedPrompt(
            self.codec.render(history, tools=schemas), self, history, schemas
        )

    def record_reply(self, result: GenerateResult, messages: list[Message]) -> None:
        """Append an assistant reply, preserving native tool-call fields."""
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
    exact_continuation: bool = False

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

        Native policies enforce their tool-call/response contract. Context protocols
        validate their own prefix-preservation requirements.
        """
        return None

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
        prefix, params = prepare_generation(
            self.default_params, tokens, sampling_params
        )
        result = await self._complete(prefix, params)
        result.exact_generation &= self.provides_exact_tokens
        return result

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

    def startup_check(self) -> None:
        """Optional readiness check."""

    def close(self) -> None:
        """Release policy-owned resources."""

    async def aclose(self) -> None:
        """Async lifecycle hook for policies that own an async client."""
        self.close()


register_policy = registrar(Policy)
