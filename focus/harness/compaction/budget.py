"""``budget``: the agentic strategy under a token-or-counter trigger.

Not a class. Folding "on the token budget *or* once the round's tool budget is spent" is
a statement about *when*, and when is a :class:`~...compaction.triggers.Trigger` now --
so this is the composition, registered under a name, rather than a subclass that welds a
state counter onto the agentic strategy by inheritance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from focus.generation import SamplingParams
from focus.harness.compaction.agentic import AgenticCompactor
from focus.harness.compaction.base import Compactor
from focus.harness.compaction.triggers import (
    AnyOf,
    PromptTokens,
    StateCounter,
)
from focus.registry import build, register


def _sampling(value: SamplingParams | Mapping[str, Any]) -> SamplingParams:
    """Coerce the one forwarded knob that is not a scalar.

    Everything else in ``**agentic_kwargs`` is a plain value, but a nested mapping is
    only coerced off a parameter's *annotation* -- and ``**kwargs`` has none. Without
    this, ``{"name": "budget", "sampling_params": {...}}`` would reach the compactor as
    a dict and fail at the first fold, a generate call away from the config that caused
    it.
    """
    return build(SamplingParams, value)


@register(Compactor, "budget", nested_builders={"sampling_params": _sampling})
def build_budget(
    *,
    remaining: str = "calls_remaining",
    maximum: str = "calls_max",
    max_prompt_tokens: int = 8192,
    **agentic_kwargs: object,
) -> AgenticCompactor[Any]:
    """An :class:`AgenticCompactor` that folds on prompt length *or* a spent counter.

    The registry accepts a factory as readily as a class (it reflects on whatever it is
    handed), so the common composition gets a name without a type to go with it.
    ``**agentic_kwargs`` forwards the strategy knobs -- ``max_interactions``,
    ``sampling_params``, ``instruction``.

    The default field names are
    :class:`~focus.harness.tools.budget.RoundBudget`'s,
    which is what makes ``{"name": "budget"}`` work with no further wiring: mix
    ``RoundBudget`` into the task state, set ``calls_max`` at ``reset``, and the
    counter half of the trigger is charged and refilled by the library. Before that
    class existed the defaults named fields nothing defined, so the counter half of
    this composition was dead unless the task happened to spell it the same way.
    """
    return AgenticCompactor(
        trigger=AnyOf(
            PromptTokens(max_prompt_tokens),
            StateCounter(remaining, maximum),
        ),
        **agentic_kwargs,  # type: ignore[arg-type]
    )


__all__ = ["build_budget"]
