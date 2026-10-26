"""Task state, action effects, observations, and rewards.

An Env receives actions that the policy has already parsed. It applies the task
rules and returns a StepResult; model generation belongs to the runner. Return new
states from step() so continuations can share their starting state safely. Tools
receive the active branch's workspace on each call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from focus.codec import Message
from focus.harness.workspace import Workspace

if TYPE_CHECKING:
    from focus.harness.rollout import AnyRollout


# ``S_co`` is covariant: a ``StepResult`` is frozen and only ever *yields* its state, so
# a result over a narrower state is usable wherever a wider one is. ``Env`` itself stays
# invariant -- it takes a state as well as returning one.
@dataclass(frozen=True)
class StepResult[S_co]:
    """What one applied action produced: the transition and the world's reply turns."""

    next_state: S_co
    #: The messages the world produced in reply -- tool results (``role: "tool"``), a
    #: user turn, whatever the env emits. The rollout loop appends them to the
    #: conversation. Empty when the step yields nothing to show.
    messages: tuple[Message, ...] = field(default_factory=tuple)
    #: Task success, scored once at the end: nonzero only on the episode-ending
    #: transition. Kept apart from :attr:`reward_step` because the two are weighted
    #: against each other by a *frozen* coefficient at training time -- an env that
    #: pre-mixed them would bake in a weight nothing downstream could revisit. See
    #: :class:`~focus.values.RewardConfig`.
    reward_outcome: float = 0.0
    #: Per-transition shaping: progress, tool success, format compliance. Dense, and
    #: only meaningful relative to a chosen ``beta_step``.
    reward_step: float = 0.0
    #: Natural episode termination. Step-budget *truncation* is the rollout loop's
    #: concern, not the env's, so it is not carried here.
    done: bool = False

    @property
    def reward(self) -> float:
        """The raw sum, outcome + step -- what tree ranking and backup read.

        Deliberately *not* a stored field: the two components have to reach a training
        statistic separately, so anything weighting them goes through a
        :class:`~focus.values.RewardConfig` and records which weight it
        used. This is the unweighted convenience view, where only ordering matters.
        """
        return self.reward_outcome + self.reward_step


class Env[S, A](ABC):
    """Define the task's initial state and response to each parsed action.

    Implement reset() and step(). Both are asynchronous so environments can await
    tool I/O. Override validate_workspace() for task-specific storage requirements.
    """

    #: The action kind :meth:`step` expects: ``"text"`` (the reply's text) or
    #: ``"tool"`` (a :class:`~focus.generation.parsing.ToolTurn`). An agent whose
    #: policy format was resolved from the model name picks the matching parser.
    action: Literal["text", "tool"] = "text"

    @abstractmethod
    async def reset(self) -> S:
        """Return the initial state for a fresh episode."""

    @abstractmethod
    async def step(
        self, state: S, action: A, *, workspace: Workspace | None = None
    ) -> StepResult[S]:
        """Apply ``action`` to ``state`` and return the transition.

        ``action`` is already decoded (an ``ActionParser`` produced it from the model
        output) -- ``step`` never calls a model. Do not mutate ``state`` in place; the
        next state in the :class:`StepResult` keeps branches independent. ``workspace``
        is the rollout's durable scratch space, passed per call so tools write to *this*
        branch's store; envs that need no workspace ignore it.
        """

    def validate_workspace(self, workspace: Workspace) -> None:
        """Check task-specific workspace requirements before an episode starts.

        The default accepts any workspace. Override this method to reject an
        incompatible store before the runner requests a model generation.
        """
        return None

    async def finish(self, state: S, action: A) -> StepResult[S]:
        """End the episode *on* ``action`` -- the rollout is out of steps.

        Not a :meth:`step`: the budget is spent, so nothing in ``action`` is executed
        (no tools run, no world effects) and the transition is terminal whatever it
        contains. The default records the state unchanged; override to read the final
        answer out of the action or to score it, exactly as the natural ending does.
        See :meth:`~focus.harness.agent.Agent.finish`, which drives it.
        """
        return StepResult(state, done=True)

    def should_finish(self, rollout: AnyRollout) -> bool:
        """Whether to stop working and take the final-answer turn now.

        Called before each decision of a live rollout. The default never stops early;
        override for task limits such as repeated identical requests.
        """
        return False

    def finish_tools(self, tools: Sequence[dict[str, Any]]) -> Sequence[dict[str, Any]]:
        """Tool schemas offered on the final-answer turn; by default, all of them."""
        return tools

    def on_compaction_failure(self, state: S, reasons: tuple[str, ...]) -> S:
        """The terminal state of an episode ended by a rejected compaction.

        Used when the agent is built with ``on_compaction_failure="end"``. The episode
        ends with zero outcome reward; override to clear a partial answer or record
        the reasons.
        """
        return state


__all__ = ["Env", "StepResult"]
