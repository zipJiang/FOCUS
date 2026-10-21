# 12 — Monty continuations and recursive checkpoints

## Plan

1. Add a bounded Monty executor. Persist idle sessions and suspended external calls
   as bytes; release workers before starting children. Keep the existing process
   executor for task tools that require full CPython.
2. Make protocol operations resumable. Store pending tool batches, including native
   call IDs and replies, in rollout state rather than Python coroutine frames.
3. Replace RLM's blocking callbacks with persisted child invocation batches. Advance
   children one decision at a time, retaining parallel batch execution and ordered
   results. Expose complete enclosing checkpoints at those decisions.
4. Teach Runner to resume pending work before generating another parent decision.
   Pending work must drain even when the parent used its final ordinary step.
5. Validate nested forks, serialized checkpoints, independent branch budgets,
   append-only training traces, native tool histories, limits, and worker cleanup.

## Invariants

- Every exposed checkpoint contains the entire waiting ancestor chain. No live
  interpreter, coroutine, policy, or runner is stored in the checkpoint.
- A fork inherits remaining resources, then charges its own branch scope. All active
  descendants are rebound to that scope before continuation.
- Child transitions never terminate or reward the root task. Each child generation
  joins the enclosing trace exactly once; saved trace prefixes never change.
- Python calls use `llm_query`, `rlm_query`, and their batched forms. The explicit
  batched APIs define concurrency; results retain input order.
- Monty supports a Python subset. Unsupported language features and unavailable host
  operations produce model-visible errors. Full CPython snapshots cannot be loaded
  as Monty continuations; historical traces remain readable.

## Validation target

A root calls a child, the child calls a grandchild, and search forks before the
grandchild's next decision. Both alternatives resume through the saved child and
root Python code to different final task outcomes, without replaying prefix model
calls or Python statements.

## Implemented behavior

`ExecutionServices.monty` owns a bounded `MontyExecutor`. Each execution segment
checks out a worker, restores an idle or suspended snapshot, runs until completion
or a query, saves bytes, and releases the worker. The dependency is pinned to the
tested Monty 1.1.0 release. CPython remains available to CLM and ordinary task tools.

`ToolProtocol` saves `PendingOperations` before returning control to Runner. It
contains the current operation, remaining tool batch, native exchange, and original
effect index. RLM additionally saves a `PendingCell` with ordered child rollouts
and accumulated stdout; the suspended Monty session carries the cell's remaining
Python execution time.
Repeated `Runner.advance` calls advance unfinished children once per round. A round
may generate one decision in each independent sibling; each nested invocation follows
the same rule. Completed siblings are reused when the enclosing checkpoint forks.

`Runner.iter_run` exposes these continuations through its existing checkpoint API.
Every pending-child round is eligible for branching, including with the default sparse
policy. Only new parent decisions consume the parent's step horizon; child horizons
and the shared resource budget still bound child work. Waiting work drains before
the parent's final answer attempt. Explicit proposal sampling controls follow actor
children, while fixed helpers keep their own policy settings.

Python errors produce tool feedback and preserve the latest valid interpreter state.
Unavailable snapshots and interpreter memory/time failures stop continuation.
Host filesystem operations are unavailable in RLM; `llm_query_batched` and
`rlm_query_batched` are the supported concurrency interfaces. A cell's timeout
covers only its own Python segments: worker startup, admission queueing, child
generation, and time parked in a saved checkpoint do not consume it. Child work is
bounded by child horizons and the shared resource budget instead. A cell that
exhausts its timeout returns a model-visible `TimeoutError` and stops continuation,
like any other interpreter time failure; it is never a retryable infrastructure error.

The ordinary `RolloutState.fork()` and pickle round-trip are sufficient: no live
runtime objects are in the checkpoint. Root task transitions alone supply outcome
reward. Child generation records enter the trace once, and local effect indices are
mapped into the interleaved enclosing trace. Existing packing, search expansion, and
Slime export consume that trace without an RLM-specific scheduler.

Tests cover a grandchild fork restored in fresh execution services, different final
task rewards, independent budgets, closures and nested Python frames, completed
sibling reuse, native Chat/Responses call IDs across a multi-operation batch,
capacity-one nesting, concurrent siblings, step exhaustion, cancellation, and
ordinary search expansion through the new checkpoints.

Validation: 1,243 tests passed and 4 skipped, including the installed Ray integration
tests. Ruff lint/format and strict mypy passed. The deterministic `rlm-memex` example
also completed with five model generations and one charged Python cell.

Outcome-based probes resume the entire computation directly. A learned critic's
default `rollout_text` still renders the parent conversation; applications training a
critic over recursive execution should provide a `NodeScorer.serialize` function
that also includes the pending child contexts and query inputs. Interpreter bytes
are execution state, not a model-readable value representation.
