# 07 — Indexed experience memory

Memex adds memory_status, memory_compress and memory_retrieve. Status exposes stable
history record numbers and archive indices. Compression stores selected history
references plus a model-written summary; it removes their live occurrences and appends
the summary. Retrieval serializes the exact referenced original messages. An archive
does not regenerate evidence from its summary.

Before: replacing a conversation could permanently discard its evidence.
After: live context -> summary; immutable history <- archive references -> retrieval.
Archive indices are local to a branch; existing indices survive forks and siblings
can add independent entries. Validation precedes the immutable commit of archive,
index and context version. Invalid indices do not partially write memory.

Context references accompany the live document. A CLM rewrite deliberately clears
these references because edited text is no longer the original history record. The
original records remain independently retrievable. Summaries and returned evidence
are observations; only the decisions that generated them enter actor token spans.

RewardConfig now weights named protocol reward components independently of task
transitions. Memex.operation_cost defaults to zero. A memory penalty does not advance
task discount time, and changing its weight requires no scheduler changes.

Interfaces: Archive(index, references, summary), Memex(operation_cost, max_archives),
Turn.reward_components and RewardConfig.component_weights. This release implements
memory execution and training traces, not a paper-scale MemexRL reproduction.

Validation: exact retrieved evidence, stable indices, sibling isolation, invalid-index
atomicity, and configurable penalties with no task-time advance. Full non-Ray suite,
strict typing, lint and formatting pass.

Exercise: assign memory a zero weight during one preparation pass and a positive
weight during another. Keep the same trace and observe why the two reward configs
must be tracked separately in training provenance.
