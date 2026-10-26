# Qwen compatibility reference

`reference.json` contains six frozen outputs from `zipJiang/rollout-controller`,
branch `workspace-harness-robust`, commit
`be3d85ace36f10f3b4ae71804ec2c3929ed7ce68`.
Its SHA-256 is `d8c22321210676935f8e0176ff62bbb56ed808c2a377b829f901589b290a2b03`.

The cases combine user, tool, and wrapped-tool feedback with ordinary advancement
and a forced final turn. They use the pinned Qwen3.5-4B Jinja template in
`../templates/qwen35.jinja` with character tokenization and scripted generations.
The records include actual input/output tokens, behavior log probabilities,
messages, completion flags, and packed training tokens, masks, spans, and tags.

This is a fixed compatibility contract. Never replace the reference with output
from current FOCUS to make a failing test pass. An intentional contract change
requires explicit review of the changed behavior and its training consequences.
Reproduction commands and the independent real-tokenizer comparison are recorded
in the publishing workspace at `docs/validation/qwen-thinking-parity.md`.

Run `uv run pytest -q tests/test_qwen_compatibility.py`. It requires no network,
model weights, GPU, or historical checkout. Missing fixtures are test failures.
The same suite also checks that thinking survives exact continuation and forks.
