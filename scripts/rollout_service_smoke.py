#!/usr/bin/env python3
"""CPU smoke for the RolloutService control plane (no GPU, no Harbor, no SSH).

Starts a real control server on loopback with a synthetic executor, then
drives the full lifecycle through the production HTTP client: register,
policy gate, activate, idempotent submit, conflict rejection, digest
validation, cancel, drain. Exits 0 only when every contract holds.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from harborrl.backends.slime_v032.rollout_service_client import (
    RolloutServiceClient,
    ServiceRolloutError,
    load_group_result,
)
from harborrl.rollout.harbor_job.collector import collect
from harborrl.rollout.service import (
    RolloutService,
    ServiceError,
    make_control_server,
    manifest_path_from,
)
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    assemble,
    digest,
    publish,
)

SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "max_new_tokens": 64, "stop": []}
TASK = {"id": "smoke-task", "revision": "1", "path": "/task",
        "task_digest": "d" * 64,
        "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0,
                           "raw_range": [0.0, 1.0]}}
RUN_ID = "smoke-run-1"


def step(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def identity(slot, *, group_id, policy_version):
    return Identity(run_id=RUN_ID, batch_id="rollout-0", group_id=group_id,
                    slot_id=slot, attempt_id=f"attempt-{policy_version}-{slot}",
                    trajectory_id=f"trajectory-{policy_version}-{slot}",
                    task_digest=TASK["task_digest"],
                    policy_version=policy_version, harness_digest="claude",
                    reward_digest=RewardProfile().digest,
                    sampling_digest=digest(SAMPLING))


def synthetic_ir(root, slot, reward, *, group_id, policy_version):
    ident = identity(slot, group_id=group_id, policy_version=policy_version)
    attempt_root = root / "receipts" / ident.attempt_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    result = {"id": "trial-" + ident.attempt_id,
              "finished_at": "2026-10-21T00:00:00Z",
              "verifier_environment_mode": "shared",
              "verifier_result": {"rewards": {"reward": reward}}}
    (attempt_root / "result.json").write_text(json.dumps(result))
    (attempt_root / "reward.json").write_text(json.dumps({"reward": reward}))
    evaluation = collect(ident, result["id"], attempt_root / "result.json",
                         attempt_root, RewardProfile(), evidence_kind="synthetic")
    trial = evaluation["harbor_trial_id"]
    turn = {"identity": ident.to_dict(), "harbor_trial_id": trial,
            "request_id": f"r-{slot}", "response_id": f"m-{slot}",
            "engine_id": "engine", "input_ids": [1, 2], "output_ids": [3, 4],
            "logprobs": [-0.2, -0.3], "logprob_semantics": "raw_model",
            "policy_version": policy_version, "tokenizer_digest": "tok",
            "template_digest": "template", "delivery": "consumed_confirmed",
            "consumption_ref": "session#digest", "role": "policy",
            "sampling": SAMPLING, "client_request": {"model": "policy"},
            "serving_input": {"input_ids": [1, 2]},
            "response": {"content": [{"type": "text", "text": "done"}]},
            "finish_reason": "end_turn", "evidence_kind": "synthetic"}
    for key in ("client_request", "serving_input", "response"):
        turn[key + "_digest"] = digest(turn[key])
    seal = {"identity": ident.to_dict(), "harbor_trial_id": trial,
            "pending_requests": 0, "status": "sealed",
            "request_ids": [turn["request_id"]], "turns_digest": digest([turn]),
            "errors": []}
    return assemble(ident, trial, [turn], seal, evaluation)


class Registry:
    def __init__(self):
        self.version, self.accepting = None, False

    def drain(self, timeout):
        self.accepting = False

    def publish_version(self, states, expected):
        self.version, self.accepting = str(expected), True


class Runtime:
    def __init__(self, root):
        self.registry, self.rollout_root = Registry(), root

    def manifest_path(self, ident):
        return manifest_path_from(self.rollout_root, ident)

    def close(self):
        pass


def group_request(group_index, *, policy_version="1", slots=2):
    return {"group_id": f"group-{group_index}", "rollout_id": 0,
            "group_index": group_index, "policy_version": policy_version,
            "task": TASK, "sampling": dict(SAMPLING),
            "slots": [{"slot_id": str(slot)} for slot in range(slots)]}


def main():
    root = Path(tempfile.mkdtemp(prefix="harborrl-service-smoke-"))
    executions = []

    async def executor(config, runtime_, request):
        executions.append(request["group_id"])
        if request["group_index"] >= 1:
            while True:  # later groups block until they are cancelled
                await asyncio.sleep(0.05)
        irs = []
        for descriptor, reward in zip(request["slots"], (1.0, 0.0)):
            ir = synthetic_ir(root, descriptor["slot_id"], reward,
                              group_id=request["group_id"],
                              policy_version=request["policy_version"])
            publish(runtime_.manifest_path(ir["identity"]), ir)
            irs.append(ir)
        return irs

    runtime = Runtime(root)
    service = RolloutService(run_root=root, executor=executor, allow_synthetic=True,
                             group_wait_timeout=10)
    # The smoke owns the service process, so it registers with its own token
    # instead of consuming the one-time generated token.
    service.register_run(RUN_ID, {"run_id": RUN_ID, "group_size": 2,
                                  "purpose": "cpu service smoke"},
                         token="smoke-token")
    service.attach_runtime(RUN_ID, SimpleNamespace(drain_timeout=1), runtime)
    control = make_control_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=control.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{control.server_address[1]}"
    client = RolloutServiceClient(url, RUN_ID, "smoke-token", timeout=10,
                                  poll_interval=0.05)

    results = []
    try:
        results.append(step("healthz answers", client.health()["ok"]))
        # Register is idempotent by manifest digest.
        replay = service.register_run(RUN_ID, {"run_id": RUN_ID,
                                               "group_size": 2,
                                               "purpose": "cpu service smoke"})
        results.append(step("register replay is idempotent", replay["existed"]))
        try:
            service.register_run(RUN_ID, {"run_id": RUN_ID, "other": 1})
            results.append(step("manifest conflict rejected", False))
        except ServiceError as exc:
            results.append(step("manifest conflict rejected", exc.status == 409))

        # Auth: a wrong token never reaches group endpoints.
        bad = RolloutServiceClient(url, RUN_ID, "wrong", timeout=10)
        try:
            bad.group_status("group-0")
            results.append(step("bad token rejected", False))
        except ServiceRolloutError as exc:
            results.append(step("bad token rejected", exc.status == 401))

        # Policy gate: nothing runs before activation.
        try:
            client.submit_group(group_request(0))
            results.append(step("submit before activate rejected", False))
        except ServiceRolloutError as exc:
            results.append(step("submit before activate rejected",
                                exc.code == "stale_policy"))

        states = [{"endpoint": f"http://engine-{i}", "healthy": True,
                   "weight_version": "1"} for i in range(2)]
        activation = client.activate_policy(states, "1")
        results.append(step("policy activation opens admission",
                            activation["policy_version"] == "1"
                            and runtime.registry.accepting))
        pool = json.loads((root / "policy-pool.json").read_text())
        results.append(step("policy pool published to shared run dir",
                            pool["weight_version"] == "1"))

        request = group_request(0)
        first = client.submit_group(request)
        results.append(step("group completes with two slots",
                            first["status"] == "complete"
                            and len(first["result"]["slots"]) == 2))
        assert client.group_status("group-0")["status"] == "complete"
        again = client.submit_group(json.loads(json.dumps(request)))
        results.append(step("resubmit is idempotent",
                            again["result"]["request_digest"]
                            == first["result"]["request_digest"]
                            and executions.count("group-0") == 1,
                            f"executions={executions}"))
        conflicting = group_request(0)
        conflicting["sampling"]["temperature"] = 0.5
        try:
            client.submit_group(conflicting)
            results.append(step("same id different content rejected", False))
        except ServiceRolloutError as exc:
            results.append(step("same id different content rejected",
                                exc.code == "conflict"))

        trajectories = load_group_result(
            RUN_ID, request, first["result"],
            allow_synthetic=any(slot["readiness"] != "RL_READY"
                                for slot in first["result"]["slots"]))
        results.append(step("client validates manifests and digests",
                            [ir["identity"]["slot_id"] for ir in trajectories]
                            == ["0", "1"]))
        export_path = Path(first["result"]["group_export"]["path"])
        export = json.loads(export_path.read_text())
        results.append(step("group export lands on shared storage",
                            export["trajectory_count"] == 2
                            and export["mean"] == 0.5, str(export_path.relative_to(root))))

        # Cancel a group that is still running, then stop admitting.
        slow = group_request(1)
        service.group_wait_timeout = 0.2
        pending = service.submit_group(RUN_ID, slow)
        results.append(step("timeout answer says running and points at status",
                            pending["status"] in ("queued", "running")
                            and "hint" in pending))
        cancelled = client.cancel_group("group-1")
        results.append(step("running group cancels",
                            cancelled["status"] == "cancelled"))

        client.drain()
        try:
            client.submit_group(group_request(2))
            results.append(step("drain closes admission", False))
        except ServiceRolloutError as exc:
            results.append(step("drain closes admission",
                                exc.code == "admission_closed"))
    finally:
        control.shutdown()
        control.server_close()
        thread.join(5)
        service.close()

    print(f"\nrun root: {root}")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
