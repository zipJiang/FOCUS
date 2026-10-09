# 04 — One inference service across Ray workers

An actor-local semaphore cannot bound aggregate requests from multiple Slime workers.
The job owner now creates a named, namespace-scoped coordinator with RayServiceConfig.
Workers connect using RayInferenceExecutor, or make_generate(execution_service=(name,
namespace)). Backend factories are importable configuration targets; clients, event
loops and PolicyFormat objects never cross this boundary.

Two workers -> one coordinator admission queue -> registered backend services.
The coordinator reconstructs each prepared prompt using its own policy format. It
owns dispatch, per-backend limits, cancellation and backend client shutdown. Actor and
request retries are disabled: an unknown outcome is a failure, never a replay.
Worker heartbeat leases bound abandoned requests. Closing a worker client cancels its
requests; only the job owner shuts down the shared coordinator.

Extension points: serializable InferenceRequest and RayServiceConfig.backends. Give
policy versions distinct identities. Coordinator creation intentionally rejects a name
collision instead of reusing an unrelated configuration. Remote value-head scoring
still uses the worker's auxiliary service; connect that endpoint with a custom shared
scoring adapter if it also consumes cluster-global inference capacity.

Validation: actual Ray worker processes submitted six requests through capacity one;
aggregate peak was one. Tests exercise cancellation, owner lease expiry, coordinator
failure and explicit shutdown. Optional imports remain lazy. A separate CI job installs
Ray and runs integration tests; no GPU services are needed.

Exercise: start two service namespaces with the same name and different capacities.
Explain why they cannot be treated as one global limit.

Ray mechanics were checked against the official documentation:
[Async actors](https://docs.ray.io/en/latest/ray-core/actors/async_api.html),
[named actors](https://docs.ray.io/en/latest/ray-core/actors/named-actors.html), and
[cancellation](https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html).
