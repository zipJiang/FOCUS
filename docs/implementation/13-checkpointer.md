# 13 — Complete snapshot persistence and recovery

## Plan and implementation

1. Separate storage from search selection. Keep `CheckpointPolicy` and the
   `iter_run()` search interface; add `Checkpointer` underneath the runner.
2. Commit one complete `RolloutState` at a time. Include messages, original history,
   exact generation trace, workspace, task state, protocol execution state, remaining
   resources and horizon. For RLM this includes suspended Monty ancestors and all
   active children together. Do not add pending writes or per-child result journals.
3. Provide in-memory and durable SQLite implementations with shared serialization,
   integrity checks, format versioning, parent IDs and scoped history lookup.
4. Retry eligible execution errors from the last committed snapshot. Keep recovery
   on the same branch; retain `fork()` for creating new search branches.
5. Verify isolation, failed commits, corruption, fresh-runner restoration, every
   protocol, recursive stacks, parallel failure and training-trace consistency.

Implemented in `focus/harness/persistence.py` and `focus/harness/runner.py`.

## Public API

```python
from focus.harness import Runner, SQLiteCheckpointer

store = SQLiteCheckpointer("rollouts.sqlite")
runner = Runner(
    policy=policy,
    env=env,
    system_prompt="Solve the task.",
    protocol=protocol,  # AppendOnly, Summary, CLM, Memex, RLM or RLM(Memex())
    checkpointer=store,
    checkpoint_retries=2,  # additional attempts for each complete operation
    retry_on=(ConnectionError, TimeoutError),
)
root = await runner.start(prompt)
# Retain these identifiers outside the process to find the branch after a crash.
episode_id, branch_id = root.episode_id, root.branch_id
result = await runner.run(root)

# After restarting, construct a runner with compatible policy, protocol, env and
# runtime versions, pointing at the same SQLite file. No live services are saved.
restored = runner.restore_latest(episode_id, branch_id)
result = await runner.run(restored)

# Exact checkpoint lookup and history (newest first):
metadata = store.list(episode_id, branch_id)
restored = runner.restore(metadata[0].id)
# At a search-eligible checkpoint, branch rather than recover the same execution:
# alternate = await runner.run(restored.fork())
```

`MemoryCheckpointer()` has the same API and serialization isolation, but lasts only
as long as its process. Checkpointing is opt-in: `checkpointer=None` preserves the
existing storage-free execution path. The default retry policy applies only when a
checkpointer is configured. Set `checkpoint_retries=0` to persist without automatic
retries; explicit recovery remains available. Add backend-specific transient exception
classes to `retry_on` if the inference client uses them.

`put(rollout)` returns a state carrying its new `checkpoint_id`. The previous ID is
recorded as its parent. A fork initially inherits its source ID, then commits a new
branch root before executing. `get(id)` loads an independent object graph. `latest`
searches only the specified episode, branch and invocation, skipping corrupt payloads;
explicit `get` fails on corruption. Unsupported storage or rollout versions fail rather
than being upgraded silently. Parent references cannot cross episodes or invocations.

## Commit and retry boundaries

The runner commits after initialization, each successful `advance`, explicit
`compact`, and `finish`. Effective per-run horizon changes are also committed. This
happens whether or not sparse search selects the state for branching. Complete
terminal/truncated states are saved too. A protocol reporting `snapshot_available=False`
leaves its last intact checkpoint in storage; such endpoints remain analysis-only.

Runner subclasses customize `_advance`, `_compact` and `_finish`, leaving public
methods to own persistence and retry. BrowserComp-Plus, LocBench and Deontic use these
hooks so their stopping rules, failure outcomes and penalty adjustments are included
in committed state. Automatic Summary compaction executes inside the enclosing runner
operation, avoiding nested commits/retry loops.

Each attempt receives isolated mutable state. On an eligible exception the runner
loads the operation's last complete checkpoint and tries again, up to the configured
limit. An RLM `ExceptionGroup` is retryable only if all its leaves match `retry_on`.
Cancellation and unmatched errors propagate. Storage failures propagate without
rerunning model/tool work. SQLite inserts payload and metadata in one transaction, so
an aborted insert never replaces the last successful record.

For parallel RLM work:

```text
save C (waiting parent + both child continuations)
    A succeeds; B fails
    discard this round's uncommitted state
restore C
    rerun A and B
save D (complete new parent + child state)
```

Children completed in an earlier successful, committed round stay completed. Children
completed only in a failed round run again. Derived summary/helper runners do not
persist independently: only the enclosing complete continuation is a recovery point.
This avoids recording a child without a matching ancestor state.

The saved logical resource allowance rolls back with the state. In-process
`ExecutionServices.usage` continues counting physical model calls/tokens, including
failed attempts. It is telemetry rather than a persisted global spend limit. A fresh
process starts fresh telemetry. Exact training traces contain only the accepted
continuation; discarded attempts do not become generations on the recovered edge.
Retries reuse the caller's sampling controls, but can produce different model outputs.
On manual recovery, supply the same desired runner configuration and sampling controls.

## Boundaries of this version

- Full snapshots, no deltas, deduplication, pending writes, automatic pruning, or
  distributed coordination. Storage grows with saved history; each local commit is
  synchronous before execution continues.
- One active writer per branch. Stop the old execution before restoring it. Fork to
  explore independent continuations concurrently.
- Cloudpickle records are trusted local artifacts. Loading them requires a compatible
  Python/runtime environment and can execute Python; do not load untrusted databases.
- Policies, clients, environments and protocol configuration are supplied by the
  caller. Any external environment state must already be represented in task/workspace
  snapshots or be independently replay-safe. Checkpointing does not undo remote side
  effects or guarantee exactly-once tool execution.
- To manually edit a saved state, call `runner.save_checkpoint(edited)` before
  continuing; retries restore the saved representation.
- This persists rollout continuations, not the scheduler's entire search tree or its
  outstanding allocation work. Training export and critic projections are unchanged.

## Validation

`tests/harness/test_persistence.py` exercises both stores, immutable stored data,
ancestry and recovery-vs-fork identity, sparse selection, atomic failed writes,
corruption fallback, bounded retries, cancellation, resource restoration, forced
finalization, summary retries, shorter continuation horizons, and fresh runner/service
recovery for plain, Summary, CLM, Memex, RLM and RLM+Memex execution. A parallel failure
regression proves both siblings are rerun and discarded attempts never enter the trace.
A nested invocation regression restores suspended ancestors without replaying their
already committed generations.

Validation result: **1,269 passed, 4 skipped**, including real Ray integration and
managed CPython/Monty execution. Ruff lint and formatting pass; strict mypy passes
for all 191 checked source files.
