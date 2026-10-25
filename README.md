# FOCUS: Active Critic Learning for Long-Horizon Language Model Agents

FOCUS uses critic predictions and observed outcomes to choose where additional
rollouts can improve value estimates. It branches from reusable agent states,
refines values from the resulting rollout tree, and prepares actor and critic
training samples.

This repository contains the Python library, benchmark environments, tests,
packaging metadata, and CI.

## Documentation

The Astro + Starlight website is the single source for installation instructions,
API reference, and task tutorials. Its source and runnable examples live in the
companion `focus-publishing` workspace under `docs/`.

To preview the website from that workspace:

```bash
cd docs
npm ci
npm run dev
```

## Development

FOCUS requires Python 3.12 or newer.

```bash
uv sync --group dev
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest -q
uv build
```
