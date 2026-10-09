# 09 — Actor training across context and recursive decisions

Packed sequences now carry invocation, policy and format identity. They split at
invocation, policy/format, ownership or context-version changes, and at non-extending
prefixes. Fixed-helper outputs remain in the trace and in subsequent observations,
but their own sequences are omitted from actor samples and their tokens are masked.

Parent, child and memory decisions from the single trainable actor receive their
enclosing edge/task credit. Child completion never contributes a separate task reward.
Rescoring skips fixed helpers, so actor importance ratios leave helper distributions
unchanged. Existing GRPO root-group restrictions and TD/VinePPO estimator contracts
remain in force. Actor versions are checked explicitly at preparation.

Likelihood validation now rejects absent, misaligned or non-finite actor columns at
preparation/export. VLLM/Tinker completion normalization and Slime/SGLang decoding no
longer replace malformed likelihoods with zero (which would mean probability one).
Masked prompt positions may lack likelihoods; selected actor positions may not.
This is a documented correction to previous permissive backend behavior.

Before: a missing logprob could look certain; a helper could enter the actor loss.
After: exact actor spans are validated; helper observations/costs are retained without
training or rescoring their policy. Turn.sampling_params records the sampling controls
used for each invocation's generation.

Validation: strict malformed-likelihood tests, fixed-helper ratio invariance, helper
rescore exclusion and an end-to-end Slime export with RLM, Memex, an actor child and an
inexact fixed helper. Concatenating exported loss-mask tokens reproduces exactly the
actor generations, once each. Full non-Ray suite, mypy and Ruff pass.

Exercise: change only a helper's likelihoods in a saved trace. The actor importance
ratio should remain identical; changing an actor likelihood should change it.
