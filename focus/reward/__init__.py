"""Reward: sequence-level reward for a serialized state context.

A :class:`RewardModel` interface (sync + async). The first
backend, :class:`VLLMRewardModel`, scores against a vLLM pooling/reward model over
``POST /pooling``.
"""

from focus.reward.config import RewardConfig
from focus.reward.interfaces import (
    AsyncRewardModel,
    RewardModel,
    run_async,
)
from focus.reward.types import RewardResult
from focus.reward.vllm import VLLMRewardModel

__all__ = [
    "AsyncRewardModel",
    "RewardConfig",
    "RewardModel",
    "RewardResult",
    "VLLMRewardModel",
    "run_async",
]
