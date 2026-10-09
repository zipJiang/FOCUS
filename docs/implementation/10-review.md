# 10 — Review findings and corrections

Review targeted concurrency, state restoration, likelihoods, reward time, accounting
and lifecycle. Confirmed problems were fixed with focused regressions:

- cloudpickle restores function globals into a separate dictionary. Restored functions
  now bind to the live session namespace, including recursive-call callbacks. Closures
  and cell-defined classes/instances are rejected explicitly until faithful support is
  implemented. Simple functions, data containers, modules and workspace files work.
- A child subprocess can create a new session. Cleanup now enumerates descendants and
  terminates them as well as the managed process group. Abrupt worker exit preserves
  its original error and always releases resident/active accounting.
- Value-model inference could remain worker-local under Ray. RayRewardModel now routes
  through the same coordinator pool; Slime requires a registered value_service when
  connecting a runtime with a value model. Actual worker loss is covered by a Ray test.
- Async service setup introduced a race in Slime's runtime cache. Construction is now
  locked, and the returned GenerateHook has explicit aclose(). Owned backend cleanup
  attempts every client even when one close fails, and is idempotent.
- Broad protocol exception handling could disguise a programming/backend error as an
  invalid model action. Only ProtocolError now becomes corrective feedback; transport
  and implementation failures propagate. Mixed task/context batches remain atomic.
- Summary context changes bypassed custom checkpoint selection. The common runner
  now applies CheckpointPolicy after all summary operations. Summary-specific behavior
  has moved completely out of Runner into the Summary protocol.
- Recursive and summary calls, rescoring, and value batches now retain episode resource
  ownership. Search reports resource counters separately from task-turn budgeting and
  reward discounting. release_episode retires settled service accounting when desired.

Before/after example: `x=1; def f(): return x`, then restore and execute `x=2; f()`.
Before the review, the result was 1. The regression now requires 2. This demonstrates
why a serializable object is not automatically a faithful interpreter checkpoint.

Validation: Python 3.12 full suite including real Ray: 1211 passed, 4 optional-dependency
skips. Python 3.13 full suite before the final two cleanup regressions: 1209 passed,
4 skipped; the added local execution suite also passes. Ruff and strict mypy pass.
Final delivery repeats the full matrix after the documentation/examples finish.

Exercise: implement a ContextProtocol that raises ProtocolError on malformed input
and KeyError on a missing backend. Verify that only the first becomes model feedback.
