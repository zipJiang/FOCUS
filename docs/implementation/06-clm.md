# 06 — Editable CLM context

CLM gives the model a context_edit tool. It runs Python against context.json, the live
JSON message array, in a branch-owned managed workspace. A valid edit replaces the
next model-visible context; it may grow, reorder, insert, delete or rewrite messages.
There is no summary schema or required token reduction. Deleted generations stay in
the append-only trace with the exact prefixes on which they were sampled.

For example, `Path('context.json').write_text(json.dumps(messages[-4:]))` retains four
messages. Import pathlib.Path and json in the cell first. Before the next generation,
the document is validated and its Python/workspace snapshot commits together with the
new context version. Invalid JSON, invalid message roles, cell errors or unsupported
snapshot state leave the previous document and session intact, with retry feedback.

Python variables and files persist through direct snapshots. Prefer Path.read_text /
write_text for file operations; file-handle objects are outside the supported snapshot
subset and must be removed from globals (including the closed variable left by `with`).
No generated code is rerun during restoration. Use ExecutionServices as an async
context manager to close process and inference resources after a job.

Interfaces: CLM(timeout, max_document_bytes), validate_document and the context_edit
schema. This is FOCUS's chosen canonical document format, not a claim of paper-scale
CLM reproduction. Suffix Cache Reuse is outside this release.

Validation: next exact prompt changes after deletion while previous tokens remain;
invalid edits retain a usable document; two sibling edits restore independent Python
globals. Full non-Ray suite: 1182 passed, 4 skipped; mypy, Ruff and formatting pass.

Exercise: reverse two user messages in context.json. Inspect the two generation
prefixes and explain why packing must split across the conditioning discontinuity.
