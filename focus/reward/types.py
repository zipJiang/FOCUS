"""Reward value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardResult:
    """A reward model's score for one serialized context.

    ``score`` is the sequence-level scalar reward. ``token_scores`` carries per-token
    values when the pooler returns them (an ALL / process reward), else ``None``.
    """

    score: float
    token_scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if self.token_scores is not None:
            self.token_scores = tuple(float(x) for x in self.token_scores)


__all__ = ["RewardResult"]
