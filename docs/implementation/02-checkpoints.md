# 02 — Checkpoints, context and generation identity

Schema 2 records explicit checkpoint positions, context versions, effects, invocation
identity, remaining steps and execution data. A fold tag labels generated work; it no
longer grants permission to branch. SparseCheckpoints and EveryDecision express the
selection strategy independently.

A snapshot now copies mutable message dictionaries, task state and execution data in
addition to the workspace. This closes an aliasing hole in the previous assumption
that tuples of message dictionaries were immutable. Exact generation records remain
shared: editing live context cannot rewrite the history used for training.

Packing also separates invocation and policy identities even if token prefixes match.
Forks preserve remaining horizon; an explicit horizon override is accepted at a fresh
root, not used to replenish an already-spent continuation budget.

Validation: full suite, strict typing, lint and formatting. New checks exercise mutable
message/global isolation and identical prefixes belonging to distinct invocations.

Exercise: construct two contexts with identical visible text but different external
memory. Explain why they must remain distinct checkpoints for value estimation.

Complete snapshot storage is now a separate opt-in runner service. See
[13 — Complete snapshot persistence and recovery](13-checkpointer.md) for the
checkpointer API, atomic commits and retry semantics. Search selection does not
control which recoverable states are persisted.
