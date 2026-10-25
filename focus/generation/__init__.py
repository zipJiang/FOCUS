"""Model policies: preparation, generation, decoding and action parsing.

Token-native policies retain exact input and output IDs. Corporate API policies send
structured messages and report estimated IDs as inexact, suitable for evaluation.
"""

from focus.generation.continuation import (
    TokenChunk,
    TokenChunkParser,
    TokenContinuationPolicy,
)
from focus.generation.openai_chat import OpenAIChatPolicy
from focus.generation.openai_responses import OpenAIResponsesPolicy
from focus.generation.policy import (
    Policy,
    PolicyFormat,
    PreparedPrompt,
    register_policy,
)
from focus.generation.sglang import SGLangPolicy
from focus.generation.slime import SlimePolicy
from focus.generation.tinker import TinkerPolicy
from focus.generation.types import (
    GenerateResult,
    NativeToolCall,
    SamplingParams,
    TokenId,
)
from focus.generation.vllm import VLLMPolicy

__all__ = [
    "OpenAIChatPolicy",
    "OpenAIResponsesPolicy",
    "SGLangPolicy",
    "SlimePolicy",
    "TinkerPolicy",
    "Policy",
    "PolicyFormat",
    "PreparedPrompt",
    "VLLMPolicy",
    "GenerateResult",
    "NativeToolCall",
    "SamplingParams",
    "TokenChunk",
    "TokenChunkParser",
    "TokenContinuationPolicy",
    "TokenId",
    "register_policy",
]
