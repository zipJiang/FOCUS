"""Corporate Responses policy with stateless, structured history.

Every request sends the full branch history with store=False. Function calls and their
outputs use Responses input items. Provider output items, including encrypted reasoning,
are retained for continuation; local token IDs are estimates and never on-policy data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from focus.codec import Message, encode_text
from focus.generation._openai_shared import (
    NativePolicy,
    _request_error,
)
from focus.generation.interfaces import (
    merge_sampling_params,
    normalize_stop_reason,
)
from focus.generation.policy import (
    PolicyFormat,
    PreparedPrompt,
)
from focus.generation.types import (
    GenerateResult,
    NativeToolCall,
    SamplingParams,
    TokenId,
)


def _flatten_tool(schema: Mapping[str, Any]) -> dict[str, Any]:
    """One function schema in the Responses API's flat shape.

    Chat completions nests the function under ``{"type": "function", "function":
    {...}}``; the Responses API hoists those keys to the top level. A schema already
    written flat (no ``"function"`` mapping) is passed through unchanged, so a caller
    who wrote the endpoint's own shape is not double-converted.
    """
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return dict(schema)
    # ``**function`` rather than three named keys: ``strict`` and whatever else the API
    # grows live at the same level, and dropping them silently would be worse than
    # forwarding one the server rejects out loud.
    return {"type": "function", **dict(function)}


class OpenAIResponsesPolicy(NativePolicy):
    """Token completion backed by the OpenAI Responses API.

    :meth:`agenerate` ``await``s ``AsyncOpenAI``; sync :meth:`generate` is the
    :class:`~focus.generation.Policy` bridge over it.
    """

    #: The Responses output list always has room for ``function_call`` items: the
    #: channel is the backend's, so every result carries a tuple (see `_parse_output`),
    #: never `None`.
    provides_native_tool_calls = True

    def __init__(
        self,
        *,
        model: str,
        format: PolicyFormat[Any] | None = None,
        version: str = "policy",
        default_params: SamplingParams | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        locked_sampling: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            format=format,
            version=version,
            default_params=default_params,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            locked_sampling=locked_sampling,
        )
        #: ``None`` leaves the effort to the model's own default, which is also what a
        #: non-reasoning model accepts; a string ("minimal", "low", "medium", "high")
        #: is sent as ``reasoning={"effort": ...}``. Not validated here -- the set is
        #: the API's to grow, and a name it does not know is a server error naming
        #: itself, which beats a client-side list that ages.
        self._reasoning_effort = reasoning_effort

    async def agenerate(
        self,
        prompt: PreparedPrompt,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self._check_prompt(prompt)
        prefix_tuple = prompt.tokens
        params = merge_sampling_params(self.default_params, sampling_params)
        model = self.served_model
        messages = _response_input(self._to_messages(prompt.messages), prompt.messages)

        try:
            response = await self._async_client().responses.create(
                **self._response_kwargs(model, messages, params, prompt.tools)
            )
        except Exception as exc:
            raise _request_error(
                exc,
                base_url=self.base_url,
                model=model,
                prefix_tokens=prefix_tuple,
                params=params,
                label="responses",
            ) from exc

        return self._parse_output(response, prefix_tuple)

    def _response_kwargs(
        self,
        model: str,
        messages: list[Message],
        params: SamplingParams,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": params.max_tokens,
            # Always, and never `previous_response_id`: server-side conversation state
            # is incompatible with a harness whose folds rewrite the prefix and whose
            # tree forks share it. Stateless full-input-per-call is the only mode in
            # which "the prompt" means what this library says it means.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if not self._locked_sampling:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
        if tools:
            # The model's own tool protocol, in *this* endpoint's spelling of it.
            # `tool_choice` is left at the API's default -- the harness asks for tools
            # to be available, never for one to be called.
            kwargs["tools"] = [_flatten_tool(schema) for schema in tools]
        if self._reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        return kwargs

    def _parse_output(
        self,
        response: Any,
        prefix_tuple: tuple[TokenId, ...],
    ) -> GenerateResult:
        """The output *list* read by item type: text, calls, and nothing else.

        ``reasoning`` items are skipped deliberately. Their summaries are a rendering
        of thinking the API did not return in full, not the completion -- folding them
        into ``text`` would put words in the assistant turn that the model never said
        out loud, and the harness re-renders that turn into the next prompt.
        """
        texts: list[str] = []
        calls: list[NativeToolCall] = []
        for item in getattr(response, "output", None) or ():
            kind = str(getattr(item, "type", "") or "")
            if kind == "function_call":
                calls.append(
                    NativeToolCall(
                        name=str(getattr(item, "name", "") or ""),
                        # the wire's JSON text verbatim, as `NativeToolCall` documents
                        arguments=str(getattr(item, "arguments", "") or ""),
                        call_id=str(getattr(item, "call_id", "") or ""),
                    )
                )
            elif kind == "message":
                texts.append(_message_text(item))
        text = "".join(texts)
        return GenerateResult(
            tokens=tuple(encode_text(self._codec, text)),
            prefix_tokens=prefix_tuple,
            text=text,
            stop_reason=_stop_reason(response),
            logprobs=[],
            exact_generation=False,
            native_tool_calls=tuple(calls),
            native_output=tuple(
                _output_item(item) for item in getattr(response, "output", ()) or ()
            ),
        )


def _message_text(item: Any) -> str:
    """The text of one ``message`` output item -- its ``output_text`` parts, joined."""
    parts = getattr(item, "content", None) or ()
    return "".join(
        str(getattr(part, "text", "") or "")
        for part in parts
        if str(getattr(part, "type", "") or "") == "output_text"
    )


def _stop_reason(response: Any) -> str:
    """The response's status as a finish reason in this library's vocabulary.

    A Responses call reports *why* it is short on ``incomplete_details.reason`` rather
    than on a per-choice ``finish_reason``; the one reading that matters to a harness is
    ``max_output_tokens``, which is the same event every other backend calls ``length``.
    Anything else finished, so it stopped.
    """
    if str(getattr(response, "status", "") or "") != "incomplete":
        return "stop"
    details = getattr(response, "incomplete_details", None)
    reason = (
        details.get("reason")
        if isinstance(details, Mapping)
        else getattr(details, "reason", None)
    )
    if str(reason) == "max_output_tokens":
        return "length"
    return normalize_stop_reason(reason)


__all__ = [
    "OpenAIResponsesPolicy",
]


def _output_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return dict(item.model_dump(mode="json", exclude_none=True))
    if isinstance(item, dict):
        return dict(item)
    return {key: _output_value(value) for key, value in vars(item).items()}


def _output_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_output_value(part) for part in value]
    if hasattr(value, "__dict__"):
        return _output_item(value)
    return value


def _response_input(messages: list[Message], history: list[Message]) -> list[Message]:
    # Validation ran on canonical messages above. Match metadata by assistant position;
    # an optional policy system prompt adds no assistant and cannot shift that index.
    native = iter(m for m in history if m["role"] == "assistant")
    items: list[Message] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        elif role == "assistant":
            saved = next(native)
            if saved.get("native_output"):
                items.extend(saved["native_output"])
                continue
            if message.get("content"):
                items.append({"role": role, "content": message["content"]})
            items.extend(
                {"type": "function_call", "call_id": call["id"], **call["function"]}
                for call in message.get("tool_calls", ())
            )
        else:
            items.append(message)
    return items
