# Context runtime migration (checkpoint schema 2)

FOCUS now separates search allocation, context protocols, execution services and
checkpoint selection. This is a breaking API and checkpoint change. Regenerate legacy
schema-0/1 rollouts for continuation; existing artifacts can be read for analysis but
cannot safely restore the new runtime state. CLM snapshots use trusted, version-
specific cloudpickle data. RLM now uses Monty interpreter snapshots. Restore with
the same dependency environment; earlier CPython RLM sessions are analysis-only.

## Construct a runtime

Replace `Runner(compactor=compactor)` with `Runner(protocol=Summary(compactor))`.
Use `AppendOnly()` or omit `protocol` for append-only behavior. Protocols are available
from `focus` and `focus.protocols`: `CLM()`, `Memex()`, `RLM(context=Memex())`, etc.
The three maintained environment factories have been migrated.

```python
from focus import AdmissionPool, ExecutionServices, LocalInferenceExecutor
from focus import Runner, RLM, Memex, ResourceBudget, EveryDecision

async with ExecutionServices(
    inference=LocalInferenceExecutor(AdmissionPool(8, max_pending=64), own_clients=True)
) as services:
    runner = Runner(
        policy=actor,
        env=task,
        system_prompt=instructions,
        protocol=RLM(context=Memex()),
        services=services,
        resource_budget=ResourceBudget(model_calls=128, process_cells=32),
        checkpoints=EveryDecision(),
    )
    result = await runner.run_from(task_input)
```

Use one injected service object for all runtimes sharing local capacity. Derived and
recursive runners inherit it. Runtime shares it with TokenScorer and value scoring.
Local values and generation use one admission pool by default. An injected executor
may use another transport; register it under InferenceExecutor or use `_target_`.
Only service owners close resources. `own_clients=False` leaves injected policy clients
with their caller; choose True when the executor owns them. `release_episode(id)` drops
settled service accounting; checkpoint budgets remain available for later forks.

## Ray and Slime

Install `focus[ray]`. The job owner calls `create_ray_service(RayServiceConfig(...))`
with explicit `name`, `namespace` and importable `_target_` backend factory configs.
Factories build policies inside the coordinator; never serialize live clients or
PolicyFormat instances. Policy keys are `model:version:format.name`. Configure reward
factories in `RayServiceConfig.rewards` with stable keys.

Workers use `RayInferenceExecutor(name, namespace)`. Slime's `make_generate` accepts
`execution_service=(name, namespace)` and, when a value model is configured,
`value_service='registered-reward-key'`. Both types of requests consume the same
coordinator capacity. `await generate.aclose()` closes worker-side resources; the job
owner must call the coordinator's `shutdown.remote()` and then `ray.kill(handle)`.
Client shutdown does not stop the shared service. Do not automatically restart/replay
an unknown-outcome request. Heartbeats reclaim work after a worker disappears.

Ray optional tests run actual processes: `uv run --extra ray pytest -m ray -q`.
The coordinator queue and active requests are bounded; Ray transport mailboxes are
subject to Ray's own admission behavior. Avoid submitting unbounded tasks upstream;
the service rejects excess admitted pending requests with CapacityError.

## Context and checkpoint contracts

`ContextProtocol.initialize/before/apply` manage context and effects. Task environments
remain model-free. `ToolProtocol.operation` implements model-visible operations;
`ProtocolError` means invalid actor input, while implementation/transport errors must
propagate. Mixed task/protocol tool batches are rejected; context operations execute
serially. See `examples/custom_protocol.py` for registration without scheduler edits.

For native tool policies, context operations edit the settled history. The current
assistant tool call and its batch of replies are kept together after the edited
context, outside the editable document. Edits that break older call/response pairs
are rejected before context or Python session state commits. Policy adapters can
implement `validate_context(messages)` to enforce additional history constraints.

A checkpoint stores live messages, immutable-history references, branch memory,
execution snapshots, remaining decision/resource budgets and a trace cursor. Forking
copies mutable state and creates a new accounting branch. SparseCheckpoints selects
roots and completed context/invocation operations; EveryDecision also exposes ordinary
decision boundaries. Pending RLM child decisions also expose complete continuations
with either policy. A context change, a checkpoint and a training boundary are three
separate concepts. Diagnostic task/fold tags do not grant branching permission.

CLM edits `context.json` using managed Python. It accepts nonempty JSON arrays of
messages with standard roles and string content; deletion/insertion/rewriting/reordering
are supported. Validation and Python/workspace state commit atomically. This canonical
format deliberately excludes arbitrary custom roles and multimodal payloads for now.
An invalid edit preserves the last valid state and returns corrective feedback.

Memex references original history records by stable indices; status exposes live
indices and archived summaries. Retrieval returns the exact referenced evidence.
Archives are branch-local. `operation_cost` and RewardConfig.component_weights control
memory shaping without altering task-time discounting.

RLM externalizes task input into Python `context`. Its single/batched LM and RLM calls
share inference admission. Batches preserve input order. Only the root ends the task;
children have local contexts and their generations receive enclosing task/edge credit.
One trainable actor plus separately versioned fixed helpers is supported.

`Runner.advance` may now return while a child invocation is unfinished. Continue
calling it, or use `run`/`iter_run`, to resume the pending work. `fork()` at such a
checkpoint copies every waiting ancestor and active child; child alternatives return
through their own parent continuations to the root task outcome. Batched children
advance concurrently in rounds and keep input order. Checkpoints contain bytes and
rollout data, never live workers. See the [Monty refactor](implementation/12-monty.md).

## Execution and budgets

RLM sessions use Monty 1.1.0, installed with the core package. They save both idle
namespaces and suspended Python call stacks, including local variables and closures.
Monty implements a Python subset: ordinary third-party imports and host filesystem
access are unavailable. Use the four query helpers for sequential or batched child
calls. Unsupported code produces tool feedback; a failed snapshot makes the branch
truncated and non-forkable. Old ProcessSnapshot RLM sessions cannot be converted by
replaying code. Generate fresh rollouts for continuation.

CLM sessions persist through direct namespace/workspace snapshots, not code replay.
Supported state includes serializable data containers, imported modules and
simple top-level functions. Closures, cell-defined classes/instances, file handles,
symlinks and other live resources are explicitly unavailable. Use Path.read_text and
Path.write_text, or remove handle variables after closing them. Unsupported CLM state
rejects the whole edit transaction.
These managed processes execute trusted research code; they are not container isolation.

Inference permits cover backend requests only. Process permits cover active cells only;
a waiting parent releases both. Resident limits fail immediately on exhaustion.
Monty additionally releases its worker at every suspension, so a deep waiting stack
uses checkpoint storage rather than resident processes. A Python cell is charged
once, even if it suspends and resumes many times.
Queue/execution deadlines are distinct on InferenceRequest. Runner.generate_timeout
retains the existing whole-request bound. ResourceBudget limits calls, prompt/output
tokens and cells; descendants consume the parent's scope. Output-token overruns are
reported after the backend responds, so use backend max_tokens to bound single calls.
SearchPass.budget retains task-turn semantics; resource_* search statistics report new
compute separately, including failed calls, descendants, rescoring and value batches.

## Training and examples

Turn records exact prefixes, outputs, likelihoods, sampling controls, actor ownership,
policy/format version and invocation identity. PackedSequence separates invocation,
policy/format, context-version and conditioning discontinuities. Fixed helpers are
excluded from actor loss and rescoring; their observations and costs remain present.
No helper distribution is changed in an actor importance ratio. GRPO remains root-only;
TD and VinePPO retain their estimator restrictions. Missing/misaligned/non-finite actor
likelihoods now fail preparation/export instead of being silently normalized to zero.

Run deterministic, GPU-free demonstrations from the repository root:

```sh
python -m examples.protocols --protocol clm
python -m examples.protocols --protocol memex
python -m examples.protocols --protocol rlm
python -m examples.protocols --protocol rlm-memex
```

Paper-scale reproduction, multiple concurrently trained policy families, container
isolation and CLM Suffix Cache Reuse remain outside this release.

## Complete snapshot checkpointers

Pass `checkpointer=MemoryCheckpointer()` or `SQLiteCheckpointer(path)` to `Runner`
(imported from `focus.harness`) to save complete states independently of sparse search
selection. Recover the same branch with `runner.restore(id)` or
`runner.restore_latest(episode_id, branch_id)`; use `fork()` to create a new search
branch. Two retries for `ConnectionError`/`TimeoutError` are enabled when a checkpointer
is configured; `checkpoint_retries=0` disables automatic retries. There are no pending
writes: work since the last complete snapshot is rerun after failure.

Runner subclasses should override `_advance`, `_compact` and `_finish` instead of
public methods, keeping their result changes inside the shared persistence boundary.
Existing schema-2 rollouts without `checkpoint_id` load with `None`; saving one creates
its first storage record. Legacy execution snapshots retain their existing restrictions.
Checkpointer storage has its own format version and uses trusted cloudpickle payloads.
See [the implementation note](implementation/13-checkpointer.md) for a runnable API
example, resource-accounting semantics and recovery limits.
