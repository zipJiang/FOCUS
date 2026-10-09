# Context runtime migration (checkpoint schema 2)

FOCUS now separates search allocation, context protocols, execution services and
checkpoint selection. This is a breaking API and checkpoint change. Regenerate legacy
schema-0/1 rollouts for continuation; existing artifacts can be read for analysis but
cannot safely restore the new runtime state. Python snapshots are trusted, version-
specific cloudpickle data; restore with the same Python/dependency environment.

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

A checkpoint stores live messages, immutable-history references, branch memory,
execution snapshots, remaining decision/resource budgets and a trace cursor. Forking
copies mutable state and creates a new accounting branch. SparseCheckpoints selects
roots and completed context/invocation operations; EveryDecision also exposes ordinary
decision boundaries. A context change, a checkpoint and a training boundary are three
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

## Execution and budgets

Logical Python sessions persist through direct namespace/workspace snapshots, not code
replay. Supported state includes serializable data containers, imported modules and
simple top-level functions. Closures, cell-defined classes/instances, file handles,
symlinks and other live resources are explicitly unavailable. Use Path.read_text and
Path.write_text, or remove handle variables after closing them. Unsupported RLM state
ends that branch as truncated/non-forkable; CLM rejects the whole edit transaction.
These managed processes execute trusted research code; they are not container isolation.

Inference permits cover backend requests only. Process permits cover active cells only;
a waiting parent releases both. Resident limits fail immediately on exhaustion.
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
