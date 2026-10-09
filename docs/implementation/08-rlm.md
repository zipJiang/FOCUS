# 08 — Recursive and parallel invocations

RLM stores large input in the Python variable context instead of inserting it into
the initial prompt. The rlm_python tool runs a persistent logical session through
saved Python/workspace snapshots. llm_query and rlm_query run one child; their batched
forms run independent children concurrently and return results in input order.

The protocol decides when parallel calls are independent. RLM does not imply that
every reasoning step is parallel. Here a model-issued batch is the explicit parallel
operation. Parents waiting for callbacks hold neither inference nor active-cell
capacity. Resident process bounds, maximum depth, fanout, child steps and execution
deadlines make exhaustion explicit.

A child Runner inherits ExecutionServices and the enclosing resource-budget scope.
ResourceBudget travels with checkpoints; alternatives inherit remaining resources,
then charge a distinct branch scope. Model calls, prompt/output tokens and Python
cells have separate optional limits. Cumulative service usage includes descendants.
Task transitions and discount time remain separate from resource consumption.

Child generations receive distinct invocation IDs and join the parent trace in causal,
input order. Child task transitions are stripped on attachment: only root task
completion earns task reward. Completed child states retain their local contexts and
archives for inspection. Fixed helper policies use actor=False; the next milestone
makes this ownership explicit in actor training export.

RLM(context=Memex()), RLM(context=CLM()) and RLM(context=Summary(...)) compose context
behavior inside recursive children. Agentic summarization now inherits the parent's
episode and resource scope too. VinePPO probes read remaining checkpoint horizon,
removing the old assumption that task/fold tags determine decision cost.

Validation: capacity-one nested recursion, overlapping parallel children, ordered
results, external-input prompt exclusion, depth rejection, remaining budget after a
fork, and RLM plus memory. Full non-Ray suite: 1191 passed, 4 skipped; typing and Ruff
pass. These are execution/credit semantics, not a paper-reproduction result.

Exercise: set model_calls=2 and ask for a three-child batch. Inspect the explicit
budget failure and explain why no child is allowed a fresh enclosing-task budget.
