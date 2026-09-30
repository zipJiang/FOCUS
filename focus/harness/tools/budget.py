"""Task-state mixins: the fields the harness itself reads off a task state.

A task state is the author's own dataclass and the harness has no opinion about what a
task tracks. But a few pieces of machinery keep bookkeeping *on* that state -- they have
nowhere else to put it, because a state is threaded through forks and folds and an env
that stashed the same counter on itself would leak it across branches. Every such field
is declared as a mixin, so composing them is how an author says which conventions the
state opts into::

    @dataclass(frozen=True)
    class MyState(SubmitFields, RoundBudget):
        question: str = ""

:class:`~focus.harness.tools.submission.SubmitFields` supplies optional
answer storage. :class:`RoundBudget` supplies the tool budget spent by dispatch and
refilled by :class:`~...compaction.triggers.StateCounter`.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoundBudget:
    """A per-round tool-call budget: spent by dispatch, refilled by a fold.

    The pattern behind ``StateCounter`` (search / browse / deontic): an agent gets so
    many tool calls per round of context, and running out is what makes it stop and
    compact. Before this existed both halves were the task author's to write -- the
    library named ``calls_remaining`` / ``calls_max`` as the trigger's defaults but
    defined neither, so the registered ``budget`` compactor folded on a field nothing
    decremented and only a script that had written its own countdown ever saw it fire.

    Mixing this in supplies the fields and, with them, the decrement:
    :meth:`~focus.harness.tools.environment.ToolEnv.on_tool_calls` charges
    ``calls_remaining`` one per dispatched call when the state carries it, and
    :meth:`~focus.harness.compaction.triggers.StateCounter.relieve` refills it
    from ``calls_max`` at each fold. A task that wants different field names keeps
    naming them to ``StateCounter`` and does its own charging, exactly as before.

    Both default to ``0``, which means *unconfigured*, not *unlimited*: ``reset`` is
    expected to hand back a state whose ``calls_max`` is the real budget. A state left
    at ``0`` is refused by :meth:`...StateCounter.relieve` rather than folding forever
    on a counter that refills to spent.
    """

    #: Calls left in this round. Reaching zero is what fires a ``StateCounter``.
    calls_remaining: int = 0
    #: What a fold refills ``calls_remaining`` to. Set it at ``reset``.
    calls_max: int = 0


__all__ = ["RoundBudget"]
