"""Remaining inference resources travel with each branch checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from focus.execution.pool import CapacityError


@dataclass(frozen=True)
class ResourceBudget:
    model_calls: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("resource limits must be nonnegative integers or None")

    def charge(self, **costs: int) -> ResourceBudget:
        updated = {}
        for name, cost in costs.items():
            remaining = getattr(self, name)
            if cost < 0:
                raise ValueError("resource charges cannot be negative")
            if remaining is not None:
                if cost > remaining:
                    raise CapacityError(f"rollout {name} budget exhausted")
                updated[name] = remaining - cost
        return replace(self, **updated)


__all__ = ["ResourceBudget"]
