"""Remaining resources travel with checkpoints; descendants share a live scope."""

from __future__ import annotations

from dataclasses import dataclass, replace

from focus.execution.pool import CapacityError


@dataclass(frozen=True)
class ResourceBudget:
    model_calls: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    process_cells: int | None = None

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
