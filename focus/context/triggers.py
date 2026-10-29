"""*When* to fold -- composable, and separate from *how*.

A :class:`Trigger` answers one question about the current checkpoint ("is it time?") and
owns the matching answer to "what has to change so it stops being time?". Those two
belong together and nowhere else: the agent front-loads the check, so a trigger that
fires but is never relieved re-fires on the very next turn and the rollout folds
forever. Keeping :meth:`Trigger.relieve` beside :meth:`Trigger.fires` is what makes that
pairing checkable in one place rather than split across a compactor's two overrides.

Splitting them out of the compactor is what lets *when* compose. Folding on a token
budget **or** a spent per-round call counter would otherwise mean a subclass welding a
state counter onto the agentic strategy by inheritance -- so an env author would have to
know which compactor they were getting, and a third condition would mean a third
subclass. :class:`AnyOf` composes instead, and the strategy stops caring.

The state a trigger reads is the *task* state, passed through untouched. That is why
:class:`StateCounter` names its fields as strings: the harness has no opinion about what
a task calls its budget, only that resetting it is what relieves the trigger.

The *prompt* it reads is the rendered token ids, as a bare sequence. The agent has that
list in hand -- it is what it is about to generate from -- and the only question asked
of it here is how long it is, which a wrapper answered no better than ``len`` does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from focus.generation import TokenId


class Trigger[S](ABC):
    """Whether to fold now, and what it takes to stop wanting to."""

    @abstractmethod
    def fires(self, prompt: Sequence[TokenId], state: S) -> bool:
        """Whether the current prompt / task state calls for a fold."""

    def check(self, state: S) -> None:
        """Refuse a state this trigger cannot read, once, before a rollout is spent.

        Called by the agent with the state :meth:`~...Env.reset` returned, which is the
        first moment a trigger's contract is checkable at all -- it names its fields as
        strings and the harness is generic over the state, so nothing earlier can see
        the shape. Every other place the mismatch surfaces is worse: :meth:`fires` runs
        inside an expansion, so a missing field is a bare ``AttributeError`` the
        scheduler charges as a dead rollout, and :meth:`relieve` runs later still, at
        the first fold, after the turns leading to it have been paid for. Default no-op:
        a trigger that reads only the prompt has no contract to state.
        """
        del state

    def relieve(self, state: S) -> S:
        """The state to continue from after folding. Default: unchanged.

        A trigger keyed on prompt length relieves itself -- the folded context is
        shorter, and nothing about the task state has to move. One keyed on a *counter*
        does not, and must reset here, or :meth:`fires` stays true forever.
        """
        return state


class PromptTokens(Trigger[Any]):
    """Fold once the rendered prompt exceeds ``max_prompt_tokens``.

    Self-relieving: folding replaces the transcript with a summary, so the next render
    is short again. This is the default trigger, and the only one that needs no
    cooperation from the task state.
    """

    def __init__(self, max_prompt_tokens: int = 8192) -> None:
        self._max_prompt_tokens = max_prompt_tokens

    def fires(self, prompt: Sequence[TokenId], state: Any) -> bool:
        del state
        return len(prompt) > self._max_prompt_tokens


class StateCounter(Trigger[Any]):
    """Fold once ``state.<remaining>`` is spent, then reset it from ``state.<maximum>``.

    The common task pattern (search / browse / deontic): a per-round tool budget on the
    task state, refilled at each fold.

    The default field names are the ones
    :class:`~focus.harness.tools.budget.RoundBudget` supplies, so a state that
    mixes
    it in is charged by
    :meth:`~focus.harness.tools.environment.ToolEnv.on_tool_calls` and
    refilled here
    with nothing further to write. Any other pair of names still works; the task then
    owns the decrement, as it always did.
    """

    def __init__(
        self,
        remaining: str = "calls_remaining",
        maximum: str = "calls_max",
    ) -> None:
        self._remaining = remaining
        self._maximum = maximum

    def fires(self, prompt: Sequence[TokenId], state: Any) -> bool:
        del prompt
        # ``getattr`` off a duck-typed state is ``Any``; the criterion is a bool.
        return bool(getattr(state, self._remaining) <= 0)

    def check(self, state: Any) -> None:
        """Demand both counter fields, and a refill that actually refills.

        Neither failure shows up until it has cost something otherwise: a state with no
        counter raises ``AttributeError`` out of :meth:`fires` on the first turn, and a
        ``maximum`` of zero raises out of :meth:`relieve` at the first fold -- both
        inside an expansion, which the scheduler charges as a dead rollout. Asked here
        they name the mixin that supplies the fields and the value that has to be set.
        """
        missing = [n for n in (self._remaining, self._maximum) if not hasattr(state, n)]
        if missing:
            raise TypeError(
                f"{type(state).__name__} has no {', '.join(missing)}: a state_counter "
                + f"trigger folds on {self._remaining} and refills it from "
                + f"{self._maximum}, so the task state must carry both -- mix in "
                + "focus.harness.RoundBudget, or point this trigger at the "
                + "pair of fields your state does define."
            )
        self._refuse_spent_refill(state, getattr(state, self._maximum))

    def _refuse_spent_refill(self, state: Any, refill: Any) -> None:
        """One wording for a refill that leaves the counter spent, for both callers.

        A refill that leaves the counter spent is not a small mistake: `fires` is
        checked before every turn and a fold takes no env step, so the rollout folds
        forever without advancing -- a hang, not a failure. The likeliest cause is a
        `RoundBudget` left at its unconfigured default, so say so.
        """
        if refill <= 0:
            raise ValueError(
                f"{type(state).__name__}.{self._maximum} is {refill}, so relieving "
                + f"{self._remaining} would leave it spent and this trigger would fire "
                + "forever. Set the round budget on the state `reset` returns."
            )

    def relieve(self, state: Any) -> Any:
        refill = getattr(state, self._maximum)
        # Last line of defence: unreachable once `check` passed at the root
        # checkpoint, but `relieve` is public and a task may hand back a smaller budget.
        self._refuse_spent_refill(state, refill)
        updates: dict[str, object] = {self._remaining: refill}
        return replace(state, **updates)


class AnyOf(Trigger[Any]):
    """Fold when *any* member fires; relieve through every member, in order.

    Relieving all of them rather than just the one that fired is deliberate: the point
    of relief is that ``fires`` is false afterwards, and with several members that is a
    property of the composite, not of whichever one happened to trip first.
    """

    def __init__(self, *triggers: Trigger[Any]) -> None:
        self._triggers = triggers

    @classmethod
    def of(cls, triggers: Sequence[Trigger[Any]]) -> AnyOf:
        """Build from a sequence of triggers."""
        return cls(*triggers)

    def fires(self, prompt: Sequence[TokenId], state: Any) -> bool:
        return any(t.fires(prompt, state) for t in self._triggers)

    def check(self, state: Any) -> None:
        """Every member's contract, not just the first -- any of them may fire."""
        for trigger in self._triggers:
            trigger.check(state)

    def relieve(self, state: Any) -> Any:
        for trigger in self._triggers:
            state = trigger.relieve(state)
        return state


__all__ = ["AnyOf", "PromptTokens", "StateCounter", "Trigger"]
