# Checkpoints and recovery

A checkpoint captures an agent partway through a task: its conversation, task state,
workspace, and generation history. You can **fork** that state to try another
continuation, or **restore** a saved state to recover an interrupted run.

For example, an agent has gathered evidence about two models and is about to choose
one. Forking lets it explore two possible conclusions from the same evidence.
Saving the checkpoint to disk lets it resume that work after a worker restarts.

## Choose where search can branch

The checkpoint policy defines the boundaries between rollout segments:

| Policy | Branching boundaries | Example use |
| --- | --- | --- |
| `EveryDecision()` | Each decision boundary, including ordinary task steps. | Compare continuations after each tool interaction. |
| `SparseCheckpoints()` | The root and committed compaction or context-change boundaries. | Compare continuations after the agent summarizes a round of work. |

In the paper, a tree node represents a resumable state and an edge contains the
segment leading to the next node. Compaction provides natural boundaries in the
experiments; `CheckpointPolicy` lets an application choose its own boundaries.

With an existing `policy` and `env`, construct a runner and fork its initial state:

```python
from focus import EveryDecision, Runner

runner = Runner(
    policy=policy,
    env=env,
    system_prompt="Compare the evidence and answer the question.",
    checkpoints=EveryDecision(),
)
async with runner.services:
    root = await runner.start("Which model generalizes better?")
    first = await runner.run(root.fork())
    second = await runner.run(root.fork())
```

Both continuations begin with the same task state and conversation. Each fork gets
its own branch identity and mutable workspace. Changes made by `first` stay on that
branch. `runner.iter_run(root)` yields selected intermediate checkpoints and the
endpoint when you want to inspect the segment boundaries directly.

## Save a rollout for recovery

A **checkpointer** stores complete rollout states. Search selection controls where
to branch; the checkpointer saves after initialization and each completed runner
operation, including states between sparse branching boundaries.

| Store | Lifetime | Use |
| --- | --- | --- |
| `MemoryCheckpointer()` | Current process. | Local retries and recovery tests. |
| `SQLiteCheckpointer(path)` | A local database file. | Resume after a process restart. |

```python
from focus.harness import SQLiteCheckpointer

store = SQLiteCheckpointer("rollouts.sqlite")
runner = Runner(
    policy=policy,
    env=env,
    system_prompt="Solve the task.",
    checkpointer=store,
    checkpoint_retries=2,
    retry_on=(ConnectionError, TimeoutError),
)
async with runner.services:
    root = await runner.start("Which model generalizes better?")
    episode_id, branch_id = root.episode_id, root.branch_id
    result = await runner.run(root)
```

Keep `episode_id` and `branch_id` in your job record. The database stores conversation,
task state, workspace, trace, and remaining resource allowance. On restart, construct
a runner with the same environment, policy format, protocol, and compatible runtime,
then point its checkpointer at the same file.

After the old execution has stopped, recover the latest complete state:

```python
restored = runner.restore_latest(episode_id, branch_id)
async with runner.services:
    result = await runner.run(restored)
```

`restore_latest` continues the same branch. `restored.fork()` starts an independent
branch when the restored state is eligible for search. For an exact saved state,
use `runner.restore(checkpoint_id)`; `store.list(episode_id, branch_id)` lists its
history newest first. A completed terminal state remains terminal after restoration.

## Understand retries and costs

Suppose a model call times out after three completed task steps. The runner reloads
the last committed state and retries the interrupted operation. With
`checkpoint_retries=2`, it allows two additional attempts. Set this to zero for
manual recovery, or extend `retry_on` with the transient errors raised by your client.
Retries use the same sampling controls and can produce a different response.

Accepted generations enter the recovered rollout's training trace. Physical usage
in `ExecutionServices.usage` also includes failed attempts. For example, if the
saved branch allowance is 10 calls and a failed attempt spends one, restoring the
checkpoint restores the allowance of 10; service telemetry still counts that call.
Record telemetry externally when costs must survive process restarts.

SQLite saves each complete state and its metadata in one transaction. Automatic
compaction commits with its enclosing runner operation. Derived compaction runners
contribute their trace and workspace changes to that same snapshot. Custom runners
extend `_advance`, `_compact`, and `_finish` to keep their behavior within this boundary.

## Recovery requirements

Use one active execution per branch and compatible Python dependencies when restoring
trusted checkpoint files. Cloudpickle reconstructs Python objects, so load databases
from a trusted source. Represent external task state in the checkpoint or make tool
operations safe to repeat: restoring Python state cannot undo a remote write.

Snapshots save complete rollouts; storage grows with their history. The caller owns
retention and pruning. Search-tree scheduling and outstanding allocation requests
belong to the search process and must be reconstructed separately after a restart.
A state with `snapshot_available=False` keeps the previous recoverable checkpoint.
After manually editing a saved state, call `runner.save_checkpoint(edited)` before
continuing so retries restore that edit.

See [execution services](execution.md) for shared capacity and resource accounting.
