# 11 — Delivery validation and extension exercises

The branch delivers local and Ray inference services, managed Python sessions,
versioned checkpoints, composable context protocols and actor-only RL export. The
final review also bounds outstanding requests per Ray client and rotates admission
across episodes, branches and invocations. Service queue bounds do not replace a
producer's responsibility to bound the work it submits to Ray's transport.

Run the four standalone examples in examples/protocols.py with --protocol clm, memex,
rlm or rlm-memex. All complete using a deterministic exact-token backend. They print
actor-token counts, cumulative resource usage and the remaining checkpoint budget.
Replace the demo Policy to connect a real model; preserve the runtime wiring.

External migration is documented in docs/MIGRATION.md. There are no Runner compactor
compatibility wrappers: use Summary(compactor). Old continuation checkpoints must be
regenerated. Custom protocols, inference executors and checkpoint policies can be
registered without changing the scheduler. Named reward components can be reweighted
without regenerating a trace.

Validation results are recorded after the final gates below. Distribution builds
produce a wheel and source archive; a clean wheel installation is smoke-tested outside
the checkout. Optional backend tests requiring uninstalled GPU/provider packages are
reported as skips, not as executed service validations.

Tradeoffs: canonical CLM context is standard role/string-content JSON; Python snapshot
support is explicit rather than arbitrary interpreter serialization; Ray service
failure is fatal to affected requests without unknown-outcome replay. The four scope
exclusions in the approved plan remain unchanged.

Customization exercise: register an InferenceExecutor that records each Owner and
forwards serialized requests to a different transport. Reuse the executor tests for
capacity-one recursion, bounded queues, cancellation and deadlines. Then implement a
CheckpointPolicy selecting every second decision and compare search allocations while
holding the generated trace and reward weights fixed.


Final verified gates: Python 3.12.11 and Python 3.13.5 each passed 1212 tests with
4 skips (optional OmegaConf and tiktoken tests). Both ran the real-Ray integration
suite. Ruff format/check and strict mypy passed. All four executable examples passed.
Wheel and source distribution builds passed. A fresh wheel install with no Ray package
passed public imports, registry construction and a managed-function restoration test.

Source-library addition: Context Language Models, UUID
79206745-0A08-4FF5-B01B-EF6C736BCF87, arXiv 2609.37725. Ironquill's default library
selection was preserved; no library environment variables or tags were changed. The
overview attaches verified PDF evidence for CLM, RLM and Memex and links every milestone.
