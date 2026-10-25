# Execution services

Several rollout branches can call the same model at once. `ExecutionServices` gives
them shared request capacity and usage accounting, including calls made for compaction,
critic scoring, and token rescoring.

Think of an admission pool as a queue in front of a model server: it lets a fixed
number of requests run and holds a bounded number waiting. A resource budget answers
a different question: how much inference work may this particular branch spend?

## Share capacity within a worker

The example assumes an `actor`, `task`, and input prompt are already configured:

```python
from focus import (
    AdmissionPool, ExecutionServices, LocalInferenceExecutor,
    ResourceBudget, Runner, SamplingParams,
)

async with ExecutionServices(
    inference=LocalInferenceExecutor(
        pool=AdmissionPool(capacity=8, max_pending=64),
        own_clients=True,
    ),
) as services:
    runner = Runner(
        policy=actor,
        env=task,
        system_prompt="Solve the task.",
        services=services,
        resource_budget=ResourceBudget(model_calls=128),
        sampling_params=SamplingParams(max_tokens=1024),
    )
    result = await runner.run_from(task_input)
    usage = services.usage[result.episode_id]
    print(usage.calls, usage.output_tokens)
```

| Component | Effect in this example |
| --- | --- |
| `AdmissionPool` | Runs up to eight requests and queues up to 64 pending operations. |
| `LocalInferenceExecutor` | Sends admitted requests to the configured policy. |
| `ExecutionServices` | Shares the executor and records usage for each episode. |
| `ResourceBudget` | Gives the branch an allowance of 128 model calls. |
| `SamplingParams` | Caps a single generated response at 1,024 tokens. |

Pass the same `services` object to all runners that should share this capacity.
Derived compaction runners inherit it and charge their enclosing branch. The async
context manager closes the services when collection finishes. Here `own_clients=True`
also closes the policies bound to the executor; use `False` when another owner manages
their lifetime.

## Read budgets and usage

A branch checkpoint carries its remaining model-call, prompt-token, and output-token
allowance. Forking copies that allowance to the new branch. For example, forking a
checkpoint with 20 calls left gives each independent branch 20 calls; service telemetry
counts the physical calls across both branches.

Output-token charges happen after a response arrives. Use `max_tokens` for a per-call
output bound and `ResourceBudget` for the branch allowance. `SearchPass.budget` counts
new task transitions during collection. These counters describe different resources.

`ExecutionServices.usage` records actual work during the process lifetime, including
failed attempts. Save it to your experiment log before calling
`release_episode(episode_id)`, which retires settled accounting. Checkpoint resource
allowances remain available for recovery and later forks.

`Runner.generate_timeout` bounds the complete generation request. Lower-level
`InferenceRequest` objects can set separate queue and execution deadlines. A full
admission queue raises `CapacityError`, giving the caller a clear signal to reduce
concurrent submissions.

## Share capacity across Ray workers

A Ray coordinator holds the backend clients and a common admission pool. Workers send
requests to that coordinator, so their combined traffic uses one capacity limit.
Install the `ray` extra for this deployment.

The job owner creates the coordinator with `create_ray_service` and `RayServiceConfig`
from `focus.execution.ray`. Configure its `name`, `namespace`, and backend factory
mappings. Each mapping supplies an importable `_target_` and construction arguments;
the coordinator builds its own policy clients. Policy keys use
`model:version:format.name`. Register critic factories in `RayServiceConfig.rewards`
under stable names.

Once the coordinator exists, a worker connects by name:

```python
from focus.execution.ray import RayInferenceExecutor

executor = RayInferenceExecutor(
    name="focus-inference",
    namespace="focus-training",
    max_outstanding=32,
)
services = ExecutionServices(inference=executor)
```

The name and namespace identify the existing coordinator. `max_outstanding=32` bounds
this worker's outstanding requests. The coordinator separately bounds aggregate
capacity and its queue; bound upstream submissions to keep transport queues controlled.

## Close workers and the shared service

Each worker closes its `ExecutionServices` instance when collection finishes. The
job owner shuts down the shared coordinator with `await handle.shutdown.remote()`
and then `ray.kill(handle)` after all workers have disconnected. Heartbeats reclaim
work from workers that disappear. Requests with an unknown outcome require caller
reconciliation before resubmission.

The integration suite exercises Ray with actual processes:

```bash
uv run --extra ray pytest tests/execution/test_ray.py -q
```

See [checkpoints and recovery](checkpoints.md) for restoring branch allowance and
[context protocols](context-protocols.md) for nested compaction work.
