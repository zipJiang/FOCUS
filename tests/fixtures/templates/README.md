Pinned upstream templates used to exercise feedback rendering without model downloads.
Only tokenization is replaced by a character codec; the Jinja template is unchanged.

- `qwen35.jinja`: Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `chat_template.jinja` (Apache-2.0).
- `qwen25_hermes.jinja`: Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, `chat_template` from `tokenizer_config.json` (Apache-2.0).

These cover specific supported templates. A Hermes tool-call parser alone does not
establish that every template using Hermes syntax has the same history behavior.
