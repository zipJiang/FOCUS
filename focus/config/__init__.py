"""Search configuration records.

:class:`~focus.config.rollout.RolloutConfig` holds the search passes and reward
settings of a collection, and :class:`RolloutSpec` is that record plus the two inputs
an actor's factory needs (``env``, ``system_prompt``).
"""

from focus.config.rollout import (
    RolloutConfig,
    RolloutSpec,
    SearchPass,
)

__all__ = [
    "SearchPass",
    "RolloutConfig",
    "RolloutSpec",
]
