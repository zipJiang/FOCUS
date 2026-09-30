"""Reward: sequence-level reward for a serialized state context.

Mirrors :mod:`focus.generation`: a registry-pluggable :class:`RewardModel`
interface (sync + async) with backends registered via ``register_reward``. The first
backend, :class:`VLLMRewardModel`, scores against a vLLM pooling/reward model over
``POST /pooling``.
"""

from focus.reward.config import RewardConfig
from focus.reward.interfaces import (
    AsyncRewardModel,
    RewardModel,
    register_reward,
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
    "register_reward",
    "run_async",
]
