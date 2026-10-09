# 03 — Shared local inference and managed Python execution

Previously each call site managed concurrency independently. ExecutionServices now
travels with a Runner, its derived runners, Runtime's TokenScorer and value scoring.
Local inference and value requests share admission by default. Protocols will receive
this same object instead of constructing pools when they create children.

AdmissionPool bounds active requests and pending tickets, rotates episodes and
branches, supports backend-specific capacity, and distinguishes queue deadlines from
execution deadlines. Cancellation returns the permit even between admission and the
backend starting. Capacity belongs to an executing request, never a waiting parent.

Before: parent holds a worker while waiting for child -> capacity-one deadlock.
After: Python sends a callback -> returns active permit -> child runs -> parent resumes.
Resident cells are counted separately and reject overflow rather than waiting behind
an ancestor. ProcessExecutor snapshots Python variables and workspace bytes; branches
restore those bytes instead of replaying code. Live handles and symlinks are explicitly
unsupported. This is a trusted research-code facility, not a security boundary.

Interfaces: focus/execution/{pool,inference,process}.py. Inject an InferenceExecutor
for another transport; configure AdmissionPool for backend capacity. Client ownership
is explicit. ExecutionServices is an async context manager; owners must close it.
Legacy Runner.generate_timeout retains its whole-request bound; direct inference
requests offer independent queue and execution deadlines.

Validation: bounded queue, cancellation, fair admission, deadline, snapshot isolation,
workspace restoration, callback nesting with active capacity one, unsupported resources
and process termination. Full suite: 1174 passed, 4 skipped; strict mypy and Ruff pass.

Exercise: use capacity=1 and max_resident=1, then request a recursive Python cell.
Explain why a capacity error is preferable to an indefinitely waiting parent.
