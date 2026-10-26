"""Model backends, formats, and parsers.

A policy backend serves the actor; :class:`PolicyFormat` binds its tokenizer and chat
template to an action parser. Backends resolve the format from the model name unless
one is given. :class:`VLLMCritic` serves a value model.
"""

from focus.codec import ChatCodec, Message, TextCodec
from focus.critic import VLLMCritic
from focus.generation import (
    GenerateResult,
    OpenAIChatPolicy,
    OpenAIResponsesPolicy,
    Policy,
    PolicyFormat,
    PreparedPrompt,
    SamplingParams,
    SGLangPolicy,
    SlimePolicy,
    TinkerPolicy,
    TokenChunk,
    TokenChunkParser,
    TokenId,
    VLLMPolicy,
)
from focus.generation.parsing import (
    HermesToolCallParser,
    NativeToolCallParser,
    QwenXMLToolCallParser,
    TextActionParser,
    ToolTurn,
)
from focus.tiktoken_chat import TiktokenChatTokenizer

__all__ = [
    # Policies
    "Policy",
    "VLLMPolicy",
    "SGLangPolicy",
    "TinkerPolicy",
    "SlimePolicy",
    "OpenAIChatPolicy",
    "OpenAIResponsesPolicy",
    "VLLMCritic",
    # Formats and parsing
    "PolicyFormat",
    "ChatCodec",
    "TextCodec",
    "TiktokenChatTokenizer",
    "HermesToolCallParser",
    "QwenXMLToolCallParser",
    "NativeToolCallParser",
    "TextActionParser",
    "TokenChunkParser",
    # Generation records
    "SamplingParams",
    "GenerateResult",
    "PreparedPrompt",
    "Message",
    "TokenChunk",
    "TokenId",
    "ToolTurn",
]
