# FOCUS: Focused Optimization on Critical Underperforming Segments

FOCUS improves multi-turn language-model agents by concentrating rollout and training
compute on the segments where the policy is most likely to fail or where its value
estimate remains uncertain. It represents an episode as a tree of reusable checkpoints,
allocates additional continuations to critical branches, refines checkpoint values from
the resulting evidence, and exports step-level actor and critic samples.

The core library provides the environment and compaction harness, generation backends,
search scheduling, rollout allocation, value refinement, credit assignment, and
trainer-sample export needed to build this training loop.

This repository contains the reusable Python package, its tests, packaging metadata,
and CI. It also carries maintained environments for DeonticBench, LocBench, and
BrowserComp-Plus. Experiment launchers, cluster operations, data-generation helpers,
generated artifacts, and research notes remain outside this release tree.

## Environments

The three benchmark environments live under `focus.environments` and use the same
runner, compaction, workspace, and reward contracts as the core library:

```python
from focus.environments.browsercomp_plus import SearchEnv
from focus.environments.deontic import DeonticEnv
from focus.environments.locbench import LocBenchEnv
```

Each module contains the runtime environment and the small data/repository adapters it
needs. It does not include collection CLIs, training launchers, preprocessing scripts,
evaluation reports, or experiment-specific configurations.

| Environment | Agent tools |
| --- | --- |
| DeonticBench | `list`, `grep`, `read`, `python`, `submit` |
| LocBench | `list`, `grep`, `read`, `submit` |
| BrowserComp-Plus | `browser.search`, `browser.open`, `browser.find`, `submit` |

There is one compaction contract across these environments. Every enabled compactor is
tool-free, writes structured `State`, `Evidence`, `Open questions`, and `Next steps`
notes, preserves exact harness state separately, and resumes from assistant-authored
memory. BrowserComp-Plus uses the bounded document-ID tools above; the older mutable or
character-page search dialect is not part of FOCUS.

## Install

FOCUS requires Python 3.12 or newer.

```bash
pip install -e .
```

Optional integrations are installed separately:

```bash
pip install -e ".[vllm,chat]"
```

Add the `locbench` extra when loading the benchmark's pinned Parquet release:

```bash
pip install -e ".[locbench]"
```

The public Python namespace is `focus`:

```python
from focus import FifoAllocator, RolloutConfig, Runtime, SearchPass, run_search
```

## Development

```bash
uv sync --group dev
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest -q
```

The initial clean snapshot was derived from `zipJiang/rollout-controller` commit
`19494aead05c9446ce33b4d4e500d8fcfa3174e0`.

## Context protocols and shared execution

The runtime supports editable CLM context, indexed Memex memory and recursive/parallel
RLM invocations with shared local or Ray inference admission. Start with the
[migration and extension guide](docs/MIGRATION.md) and run
`python -m examples.protocols --protocol rlm-memex` for a deterministic demonstration.
Implementation decisions and milestone validation are in [implementation notes](docs/implementation/00-overview.md).

RLM executes Python with Monty and can fork inside recursive children. Checkpoints
include waiting ancestor continuations, so each alternative resumes through its
parents to the task outcome. See the [Monty implementation](docs/implementation/12-monty.md)
for execution boundaries and Python compatibility.

Complete rollout persistence is available through `Runner(checkpointer=...)` with
`MemoryCheckpointer` or `SQLiteCheckpointer` from `focus.harness`. Every protocol uses
the same complete-snapshot recovery path, independently of search checkpoint selection.
Transient failures retry from the last intact snapshot; parallel work in a failed round
is rerun together. See [checkpointer usage and recovery semantics](docs/implementation/13-checkpointer.md).
