# 05 — Context behavior is an extension point

Runner now receives ContextProtocol. AppendOnly delegates actions to the model-free
task environment; Summary supplies the existing trigger/compactor behavior. All three
maintained environment factories now construct protocols. Runner's old compactor
constructor argument is removed; migrate to protocol=Summary(compactor).

The generation is recorded before its effects execute. Protocol effects carry that
origin index, so recursive work can append records without confusing which decision
caused a memory write. Task transitions remain distinct from context-only operations.
Checkpoint selection consumes effects rather than protocol names. EveryDecision can
expose an earlier choice; SparseCheckpoints exposes settled context operations.

ToolProtocol serializes context operations in a turn. A batch mixing task operations
and context operations is rejected before either executes; feedback tells the actor
how to retry. This avoids racing a rewrite against tools that read the old document.

Before: Runner knew how to trigger compaction and directly stepped the environment.
After: protocol.before prepares context; protocol.apply interprets the action; the
runner records exact generation and asks CheckpointPolicy whether it may branch.

See examples/custom_protocol.py for a registered retention policy. No scheduler
modification is needed. Protocols are stateless; mutable per-branch state belongs in
RolloutState.execution and is copied by snapshots.

Validation: retained task/compaction baselines, external custom protocol, context-only
credit, explicit effect origins, and mixed-batch atomic rejection. Full non-Ray suite:
1178 passed, 4 skipped (2 Ray tests deselected); mypy and Ruff pass.

Exercise: add a protocol operation that retains the initial system message plus the
last n messages. Compare SparseCheckpoints with EveryDecision on the same episode.
