"""FOCUS: active critic learning with selective rollouts for language-model agents.

An :class:`Agent` runs a policy in an environment. A :class:`Collector` searches from a
prompt -- continuing the checkpoints its :class:`Search` ranks highest, scored by a
:class:`Critic` -- and prepares actor advantages and critic targets as a
:class:`Batch`. Extension interfaces live in :mod:`focus.extend`; model backends in
:mod:`focus.backends`.
"""

from focus.collector import Collector, Tree
from focus.critic import Critic
from focus.harness import (
    Agent,
    AppendOnly,
    Env,
    Execution,
    Segments,
    StepResult,
    Summarize,
    ToolEnv,
    ToolError,
    tool,
)
from focus.preparation import ActorSample, Batch, CriticSample, DirectBranchTd, Grpo
from focus.search import ErrorReduction, Fifo, Phase, Search, Selective, TokenEntropy
from focus.values import RewardConfig

# Grouped in the order the documentation introduces them.
__all__ = [
    # Collecting a training batch
    "Agent",
    "Collector",
    "Critic",
    "Batch",
    "ActorSample",
    "CriticSample",
    "Tree",
    "Execution",
    # Where to spend the rollout budget
    "Search",
    "Selective",
    "Phase",
    "Fifo",
    "ErrorReduction",
    "TokenEntropy",
    # Actor recipes and reward weighting
    "DirectBranchTd",
    "Grpo",
    "RewardConfig",
    # The task
    "Env",
    "ToolEnv",
    "StepResult",
    "tool",
    "ToolError",
    # What the policy sees
    "AppendOnly",
    "Segments",
    "Summarize",
]
