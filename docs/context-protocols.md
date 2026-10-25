# Context protocols

A context protocol controls what the agent sees on its next model call. The task
might still be “compare two models,” while the live conversation changes from a long
search transcript to a short summary of the evidence.

FOCUS gives these jobs separate components: `Env` applies task actions,
`ContextProtocol` manages the conversation, and the allocator chooses where to sample
another continuation. This makes compaction a modular part of the rollout controller.

## Retain or summarize the conversation

| Component | What it does | When to use it |
| --- | --- | --- |
| `AppendOnly()` | Keeps the conversation and appends new actions and observations. | The transcript fits the model's context window. |
| `Summary(compactor)` | Replaces the working context when the compactor's trigger fires. | Long tasks need a shorter working context. |
| `AgenticCompactor` | Uses the acting policy to produce the replacement context. | The agent should summarize its own progress. |
| `PromptTokens` | Triggers compaction above a prompt-token threshold. | Bound the size of the working conversation. |
| `InMemoryWorkspace` | Stores named text entries separately from the conversation. | Keep detailed evidence available across compactions. |

For an existing `actor` and `task`, compact after the prompt exceeds 8,192 tokens:

```python
from focus import AgenticCompactor, Runner, Summary

compactor = AgenticCompactor(
    max_prompt_tokens=8192,
    max_reply_tokens=1024,
)
runner = Runner(
    policy=actor,
    env=task,
    system_prompt="Compare the evidence and answer the question.",
    protocol=Summary(compactor),
)
```

`max_prompt_tokens` decides when to compact; `max_reply_tokens` caps each compaction
reply. For example, the agent can replace pages of tool output with “A's evaluation
contains training users; B's evaluation uses held-out users,” while retaining detailed
notes in its workspace. The task state continues from the same point.

A compaction records its generated turns and a `compaction` effect. It consumes a
runner step; task discount time advances when the environment makes a transition.
Use `await runner.compact(checkpoint)` to request compaction explicitly.

The benchmark factories use structured summaries with **State**, **Evidence**,
**Open questions**, and **Next steps**, followed by assistant-authored memory. Each
factory supplies its task-specific prompt and state rendering.

## Trigger on a round of tool calls

`StateCounter()` triggers when `calls_remaining` reaches zero and refills it from
`calls_max` after compaction. A task state can inherit `RoundBudget` and initialize
both counters to a positive allowance. `ToolEnv` decrements the allowance as tools run.

```python
from focus import AnyOf, PromptTokens, StateCounter

trigger = AnyOf(PromptTokens(8192), StateCounter())
compactor = AgenticCompactor(trigger=trigger)
```

This compacts when either the prompt grows too long or the current round's tool
budget is spent. With `calls_max=5`, the agent summarizes after spending its five
calls, then begins another round with five calls available.

## Add a context operation

The [KeepRecent example](../examples/custom_protocol.py) adds a model-visible
`keep_recent(n)` operation. It keeps the last `n` live messages and records a context
change. The complete generation history remains available for training.

Run this fragment from a FOCUS checkout with an existing `actor` and `task`:

```python
from examples.custom_protocol import KeepRecent
from focus import ContextProtocol, Runner, build

protocol = build(ContextProtocol, {"name": "example_keep_recent"})
runner = Runner(
    policy=actor,
    env=task,
    system_prompt="Use keep_recent to retain the context you need.",
    protocol=protocol,
)
```

Importing `KeepRecent` registers its name. `build` then constructs it from the
configuration mapping. In an application, place the extension in your own package
and import it before resolving its registered name.

Subclass `ContextProtocol` for automatic context behavior or `ToolProtocol` for
operations the model calls. The lifecycle follows the rollout:

| Hook | Responsibility |
| --- | --- |
| `schemas()` | Describe context tools to the model alongside task tools. |
| `initialize(rollout, runner)` | Set up the branch's protocol state. |
| `needs_prepare(rollout, runner)` | Decide whether the next generation needs a context update. |
| `before(rollout, runner)` | Perform that update before generation. |
| `apply(rollout, action, runner, ending=...)` | Apply the action from an already-recorded generation. |
| `compact(rollout, runner)` | Handle an explicit compaction request. |

Set `manages_context=True` when the protocol owns prompt reduction. Store branch data
in `rollout.execution`; keep the protocol object shared and stateless. Return updated
rollout records with `dataclasses.replace` and copy nested mutable data before editing.
This gives each fork an independent protocol state.

A `ToolProtocol.operation` returns `(rollout, feedback, effect_kind)`. Use `context`
for a rewrite selected by `SparseCheckpoints`, and increment `context_version`.
Keep one `context_refs` entry per live message: its original history index for retained
text, or `None` for synthesized text. Append to history as operations occur.

Validate argument types and values, raising `ProtocolError` for model input that
needs correction. `ToolProtocol` returns that error as feedback and retains the last
valid context. It preserves native tool-call/reply batches and validates edited
conversations through the policy. Issue task and context operations in separate
batches; context operations in one batch finish serially before the runner returns.

## Connect context operations to training

The paper assigns each segment's advantage to its policy-generated task and
context-management tokens. Environment observations and conditioning text are masked.
The library retains the exact prefix, output tokens, behavior log probabilities,
sampling settings, and actor identity needed to construct those samples.

A context rewrite changes the next prompt, so token packing begins a new region.
For example, tokens generated from a summary must be trained with that summary as
their prefix. Invocation identity also distinguishes a compactor's nested generations
from task generations. Fixed helper turns are excluded from actor loss and rescoring.

For a new context operation, define which tokens belong to the actor and how its
reward and task duration enter the segment advantage. Verify that choice with the
estimator you use. The task/fold tag describes a generation's purpose; checkpoint
selection uses committed effects, and discounting uses task transitions.

Tests in `tests/protocols` cover operation validation, native tool histories, and
training likelihoods. The harness, preparation, and integration tests cover compaction
and sample preparation. See [checkpoints](checkpoints.md) for branching and recovery.
