# Rollout Lifecycle and Recovery

## GPU Layout

`deployment.layout: colocate` requires equal actor, rollout, and total GPU counts.
The Slime launcher passes `--colocate` only for that layout. Split deployments
omit it. The final argv is recorded in the run's command evidence, including in
dry-run mode.

## Trace Memory

Closing an attempt revokes its credential. Sealing still requires every pending
request to drain. After `trace.json` is successfully persisted, the registry
releases the full turns and errors. Repeated sealing reads the immutable record
from disk. Failed writes retain the payload so sealing can be retried.

The registry retains small attempt records to enforce attempt and credential
uniqueness and policy barriers. Training samples retain their current trajectory
evidence for as long as the training backend needs those samples.

## Cleanup Recovery

Docker and Sandbox both stop automatic retries when an attempt produces
`cleanup-required.json`. Queued attempts are checked before starting a Runner;
already running attempts finish or are cancelled with cleanup awaited. A runtime
also scans its `native-rollouts` directory before opening the gateway, so restarting
the same run cannot bypass unresolved cleanup.

1. Read each marker reported by the error, including any nested
   `runner/cleanup-required.json`. Coordinator markers include the attempt identity
   and Runner host (`null` means local execution).
2. Inspect the matching Harbor Docker or Sandbox resources on that host and
   confirm that deletion has completed. Retain the attempt's other evidence.
3. Remove only the markers whose resources have been confirmed clean. Both the
   attempt-level and nested Runner markers must be resolved.
4. Restart or resume the run. Admission reopens when no unresolved marker remains.

There is no automatic deletion of potentially live resources during recovery.

## Embedded Event Loop

Successive synchronous Slime rollout calls reuse one event loop and one cached
runtime. The runtime's semaphore and Sandbox admission semaphores belong to that
loop. Changing the runtime configuration requires closing the previous runtime.
Using a cached runtime from another loop raises `RuntimeLifecycleError` immediately.

`harborrl.rollout.runtime.close_runtime()` cancels and awaits outstanding embedded
tasks, closes the gateway, then closes async generators, the executor, and the
loop. Call it between rollout calls, outside an active event loop; process exit also
registers cleanup. The implementation supports Python 3.10. The manager-side
RolloutService continues to use its own persistent loop.

## Manager Service Shutdown

The manager service has its own executor loop; embedded-loop shutdown does not
close it. A generation drain alone only waits for policy requests, not tools,
Runner processes, verifier execution or environment deletion.

`RolloutService.close()` first blocks new admission, cancels and joins group
Tasks including their asynchronous cleanup, drains remaining generation, and
only then closes runtime/Gateway and the executor loop. Its default cleanup
budget covers the Runner's 120-second cancellation deadline and configured
Sandbox cleanup deadlines. Cancelling a cross-thread Future is not treated as
cleanup acknowledgement.

A cleanup timeout is an explicit error, not successful cancellation. Pending
cleanup keeps running on the executor, and an incomplete shutdown writes
`native-rollouts/cleanup-required.json`. Inspect the resources and resolve that
marker before reusing the run. A subsequent close can join cleanup without
sending another cancellation into an already-running `finally` block. The
controller always closes its HTTP listener, but does not suppress a service
shutdown failure.

## CPU Regression Checks

These checks use synthetic serving, mock resource allocation, real local Runner
subprocesses, shell dry-runs, and stub bootstrap commands. They do not need CUDA,
model weights, Docker, Harbor, Slime, or the Sandbox SDK.

```bash
python3 -m pytest -q \
  tests/harborrl/test_backend_bootstrap.py \
  tests/harborrl/test_native_training.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_terminal.py \
  tests/harborrl/test_rollout_runtime.py \
  tests/harborrl/test_rollout_runtime_lifecycle.py \
  tests/harborrl/test_sandbox.py \
  tests/harborrl/test_sandbox_runtime.py \
  tests/harborrl/test_slime_v032_exporter.py \
  tests/harborrl/test_rollout_service.py \
  tests/harborrl/test_delivery_barrier.py
```

GPU acceptance remains separate: run multi-round split and colocate training,
check serving/training memory behavior, verify policy version transitions, and
exercise cleanup against actual Docker/Sandbox resources. CPU tests establish
the scheduling and lifecycle contracts, not live model or GPU correctness.
