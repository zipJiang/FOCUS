# 01 — Behavioral baseline

The runtime refactor starts from a clean FOCUS main at a9750ed. Search allocation,
execution scheduling, context management, and training packing are different concerns.
The refactor will make their boundaries explicit while preserving task outcomes.

The baseline suite already exercises workspace isolation, compaction, exact prefix
packing, discounted returns, Slime runtime reuse, and concurrent VinePPO probes.
New characterization checks ensure changing live context never edits historical
training decisions, and that branching copies memory rather than sharing mutations.

## Review checklist

- Snapshot messages, memory, process globals and files without aliasing or replay.
- Keep search alternatives separate from recursive invocation relationships.
- Release inference and active process permits before waiting for descendants.
- Route roots, summaries, probes and scorers through shared admission.
- Never train retrieved text or helper output as freshly generated actor tokens.
- Reject missing or non-finite actor likelihoods; do not invent certainty.
- Account for failed/cancelled work and preserve task-time discount semantics.
- Confirm CLI examples, optional Ray integration and lifecycle cleanup.

## Migration direction

Checkpoint schema 2 introduces explicit eligibility and execution effects. Old
checkpoints cannot resume. Callers will inject ExecutionServices and select a
ContextProtocol and CheckpointPolicy. A single trainable policy can occupy several
invocations; fixed helper policies remain outside its loss.

## Exercise

Explain why deleting an assistant message from live context must not delete the
model's earlier training sample. Identify where the current Turn record preserves it.
