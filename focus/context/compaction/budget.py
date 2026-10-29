"""``budget``: the agentic strategy under a token-or-counter trigger.

Not a class. Folding "on the token budget *or* once the round's tool budget is spent" is
a statement about *when*, and when is a :class:`~...compaction.triggers.Trigger` now --
so this is the composition rather than a subclass that welds a
state counter onto the agentic strategy by inheritance.
"""

from __future__ import annotations

from typing import Any

from focus.context.compaction.agentic import Summarize
from focus.context.triggers import AnyOf, PromptTokens, StateCounter


def build_budget(
    *,
    remaining: str = "calls_remaining",
    maximum: str = "calls_max",
    max_prompt_tokens: int = 8192,
    **agentic_kwargs: object,
) -> Summarize[Any]:
    """An :class:`Summarize` that folds on prompt length *or* a spent counter.

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
    return Summarize(
        trigger=AnyOf(
            PromptTokens(max_prompt_tokens),
            StateCounter(remaining, maximum),
        ),
        **agentic_kwargs,  # type: ignore[arg-type]
    )


__all__ = ["build_budget"]
