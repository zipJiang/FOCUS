# Migrate context configuration

Context management now lives in `Runner.protocol`. Use `AppendOnly()` to retain the
conversation, or `Summary(compactor)` to replace long context with a summary. The
allocator still chooses continuations, and the estimator still computes training
signals from the collected tree.

## Replace the compactor argument

Move a compactor previously passed as `Runner(compactor=compactor)` into a `Summary`
protocol:

```python
from focus import AgenticCompactor, Runner, Summary

runner = Runner(
    policy=actor,
    env=task,
    system_prompt="Solve the task.",
    protocol=Summary(AgenticCompactor(max_prompt_tokens=8192)),
)
async with runner.services:
    result = await runner.run_from(task_input)
```

This runner compacts after its rendered prompt exceeds 8,192 tokens, then continues
the same task. To keep the full transcript, use `protocol=AppendOnly()` or the default
protocol. The registry names for configuration files are `summary` and `append_only`.

## Map older components to the current API

| Earlier component | Current action |
| --- | --- |
| `Runner(compactor=...)` | Pass `protocol=Summary(compactor)`. |
| `clm`, `memex`, or `rlm` protocol configurations | Choose `AppendOnly` or `Summary`, or implement the desired operation through `ContextProtocol`. |
| Managed CPython/Monty executors, recursive query helpers, suspended-operation hooks | Remove those runtime integrations and configure application-owned tools for the task. The benchmark `PythonTool` remains available. |
| `ResourceBudget.process_cells` | Remove this field. Current inference budgets count model calls, prompt tokens, and output tokens. |
| Public runner-operation overrides | Extend `_advance`, `_compact`, or `_finish` so persistence and retry cover the complete operation. |

The built-in context protocols are `AppendOnly` and `Summary`. For a custom protocol,
define its context effects and training-token ownership using the
[extension guide](context-protocols.md).

## Choose how to handle saved rollouts

| Saved artifact | How to use it |
| --- | --- |
| Append-only or Summary schema-2 checkpoint | Restore with compatible policy, environment, protocol, and runtime settings. |
| CLM, Memex, or RLM execution state | Inspect with the version that created it; collect fresh rollouts for the current runtime. |
| Schema-0/1 rollout | Use for historical analysis and collect a fresh rollout for continuation. |

For example, changing an old RLM configuration to `Summary` changes how the live
context is managed. Start a fresh task under that configuration so its checkpoint
state and generation history follow the current protocol from the beginning.

## Verify the updated flow

Run one prompt through the configured runner, inspect its selected checkpoints, and
prepare a training batch with the intended estimator. Check that context rewrites
create the expected token regions and that exported actor records contain valid
behavior log probabilities. For the paper's recursive backup recipe, use
`DirectBranchTdEstimator`.

- [Context protocols](context-protocols.md) explains the lifecycle and a concrete extension.
- [Execution services](execution.md) explains shared request capacity and budgets.
- [Checkpoints and recovery](checkpoints.md) shows how to save, restore, and fork states.
