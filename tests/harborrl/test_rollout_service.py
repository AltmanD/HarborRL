"""RolloutService + SlimeAdapter contracts (P3) over real local HTTP."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_contracts import SAMPLING, receipt

from harborrl.backends.slime_v032.rollout_service_client import (
    RolloutServiceClient,
    ServiceRolloutError,
    load_group_result,
    service_group_request,
)
from harborrl.rollout.service import (
    RolloutService,
    make_control_server,
    manifest_path_from,
)
from harborrl.trajectories.native import Identity, RewardProfile, digest, publish

TASK = {"id": "task", "revision": "1", "path": "/task", "task_digest": "d" * 64,
        "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0,
                           "raw_range": [0.0, 1.0]}}


def service_identity(slot, *, run_id="run-1", group_id="group-0", policy_version="1"):
    return Identity(run_id=run_id, batch_id="rollout-0", group_id=group_id,
                    slot_id=slot, attempt_id=f"attempt-{policy_version}-{slot}",
                    trajectory_id=f"trajectory-{policy_version}-{slot}",
                    task_digest=TASK["task_digest"], policy_version=policy_version,
                    harness_digest="claude", reward_digest=RewardProfile().digest,
                    sampling_digest=digest(SAMPLING))


def service_ir(tmp_path, slot, reward=1, **identity_overrides):
    from harborrl.trajectories.native import assemble

    ident = service_identity(slot, **identity_overrides)
    evaluation = receipt(tmp_path / slot, ident, reward)
    trial = evaluation["harbor_trial_id"]
    turn = {"identity": ident.to_dict(), "harbor_trial_id": trial,
            "request_id": f"r-{slot}", "response_id": f"m-{slot}", "engine_id": "engine",
            "input_ids": [1, 2], "output_ids": [3, 4], "logprobs": [-0.2, -0.3],
            "logprob_semantics": "raw_model", "policy_version": ident.policy_version,
            "tokenizer_digest": "tok", "template_digest": "template",
            "delivery": "consumed_confirmed", "consumption_ref": "session#digest",
            "role": "policy", "sampling": SAMPLING, "client_request": {"model": "policy"},
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


class FakeRegistry:
    def __init__(self):
        self.version = None
        self.accepting = False

    def drain(self, timeout):
        self.accepting = False

    def publish_version(self, states, expected):
        self.version = str(expected)
        self.accepting = True


class FakeRuntime:
    def __init__(self, root):
        self.registry = FakeRegistry()
        self.rollout_root = Path(root)

    def manifest_path(self, identity):
        return manifest_path_from(self.rollout_root, identity)

    def close(self):
        pass


def write_group_irs(tmp_path, request, rewards=(1, 0)):
    runtime = FakeRuntime(tmp_path)
    irs = []
    for descriptor, reward in zip(request["slots"], rewards):
        ir = service_ir(tmp_path, descriptor["slot_id"], reward,
                        policy_version=request["policy_version"])
        publish(runtime.manifest_path(ir["identity"]), ir)
        irs.append(ir)
    return irs


def group_request(group_index=0, policy_version="1", slots=2, task=TASK):
    return {
        "group_id": f"group-{group_index}", "rollout_id": 0,
        "group_index": group_index, "policy_version": policy_version,
        "task": task, "sampling": dict(SAMPLING),
        "slots": [{"slot_id": str(slot)} for slot in range(slots)],
    }


@pytest.mark.parametrize("probability", [1, 1.0], ids=["int", "float"])
def test_service_executor_normalizes_sampling_lock(monkeypatch, probability):
    from harborrl.rollout import service

    request = group_request()
    request["sampling"].update(temperature=probability, top_p=probability)

    async def run_group(config, current, **kwargs):
        assert kwargs["sampling_digest"] == digest(SAMPLING)
        return []

    monkeypatch.setattr(service, "run_group", run_group)
    assert asyncio.run(service.default_executor(None, None, request)) == []
    assert type(request["sampling"]["temperature"]) is type(probability)


@pytest.mark.parametrize("probability", [1, 1.0], ids=["int", "float"])
def test_slime_sampling_uses_the_same_numeric_lock(probability):
    from harborrl.backends.slime_v032.rollout import sampling_params

    args = SimpleNamespace(rollout_max_response_len=SAMPLING["max_new_tokens"],
                           rollout_temperature=probability, rollout_top_p=probability)
    params = sampling_params(args)
    assert digest(params) == digest(SAMPLING)
    assert type(params["temperature"]) is float
    assert type(params["top_p"]) is float


class ServiceHarness:
    def __init__(self, tmp_path, executor, **kwargs):
        self.tmp_path = tmp_path
        self.runtime = FakeRuntime(tmp_path)
        self.service = RolloutService(run_root=tmp_path, executor=executor, allow_synthetic=True, **kwargs)
        self.control = make_control_server(self.service, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.control.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.control.server_address[1]}"
        self.run_id = "run-1"
        registration = self.service.register_run(
            self.run_id, {"run_id": self.run_id, "group_size": 2})
        self.token = registration["run_token"]
        self.service.attach_runtime(self.run_id, SimpleNamespace(drain_timeout=1),
                                    self.runtime)

    def client(self):
        return RolloutServiceClient(self.url, self.run_id, self.token, timeout=10,
                                    poll_interval=0.05)

    def close(self):
        self.control.shutdown()
        self.control.server_close()
        self.thread.join(5)
        self.service.close()


@pytest.fixture()
def harness(tmp_path):
    calls = []

    async def executor(config, runtime_, request):
        calls.append(json.loads(json.dumps(request)))
        return write_group_irs(tmp_path, request)

    instance = ServiceHarness(tmp_path, executor, group_wait_timeout=5)
    instance.executor_calls = calls
    try:
        yield instance
    finally:
        instance.close()


def activated_states(version="1"):
    return [{"endpoint": f"http://engine-{index}", "healthy": True,
             "weight_version": version} for index in range(2)]


def test_register_auth_and_idempotency(harness):
    client = harness.client()
    assert client.health()["ok"]
    # Wrong tokens are rejected on every run-scoped endpoint.
    bad = RolloutServiceClient(harness.url, harness.run_id, "wrong-token", timeout=10)
    with pytest.raises(ServiceRolloutError) as exc:
        bad.group_status("group-0")
    assert exc.value.status == 401
    with pytest.raises(ServiceRolloutError) as exc:
        bad.submit_group(group_request())
    assert exc.value.status == 401


def test_policy_gate_and_activation(harness, tmp_path):
    client = harness.client()
    with pytest.raises(ServiceRolloutError) as exc:
        client.submit_group(group_request())
    assert exc.value.status == 409 and exc.value.code == "stale_policy"
    assert not (tmp_path / "policy-pool.json").exists()

    client.activate_policy(activated_states(), "1")
    assert harness.runtime.registry.version == "1"
    assert harness.runtime.registry.accepting
    pool = json.loads((tmp_path / "policy-pool.json").read_text())
    assert pool["weight_version"] == "1"

    # A partially unhealthy pool is never admitted.
    from harborrl.rollout.service import ServiceError

    broken = activated_states()
    broken[1]["healthy"] = False
    with pytest.raises(ServiceError, match="pool"):
        harness.service.activate_policy(harness.run_id, broken, "2")


def test_submit_is_idempotent_and_conflicts_rejected(harness):
    client = harness.client()
    client.activate_policy(activated_states(), "1")
    request = group_request()
    first = client.submit_group(request)
    assert first["status"] == "complete"
    assert len(first["result"]["slots"]) == 2
    assert len(harness.executor_calls) == 1

    again = client.submit_group(json.loads(json.dumps(request)))
    assert again["status"] == "complete"
    assert again["result"]["request_digest"] == first["result"]["request_digest"]
    assert len(harness.executor_calls) == 1  # no second execution

    conflicting = group_request()
    conflicting["sampling"]["temperature"] = 0.5
    with pytest.raises(ServiceRolloutError) as exc:
        client.submit_group(conflicting)
    assert exc.value.status == 409 and exc.value.code == "conflict"

    other = client.submit_group(group_request(group_index=1))
    assert other["status"] == "complete"
    assert len(harness.executor_calls) == 2


def test_client_validates_result_and_rejects_tampering(harness, tmp_path):
    client = harness.client()
    client.activate_policy(activated_states(), "1")
    request = group_request()
    result = client.submit_group(request)["result"]
    trajectories = load_group_result(harness.run_id, request, result,
                                     allow_synthetic=True)
    assert [ir["identity"]["slot_id"] for ir in trajectories] == ["0", "1"]

    slot = result["slots"][0]
    manifest = Path(slot["manifest_path"])
    forged = json.loads(json.dumps(result))
    forged["slots"][0]["manifest_sha256"] = "0" * 64
    with pytest.raises(ServiceRolloutError, match="digest mismatch"):
        load_group_result(harness.run_id, request, forged, allow_synthetic=True)
    tampered = json.loads(json.dumps(result))
    tampered["slots"][1]["manifest_path"] = str(tmp_path / "missing.json")
    with pytest.raises(ServiceRolloutError, match="cannot read slot manifest"):
        load_group_result(harness.run_id, request, tampered, allow_synthetic=True)
    assert manifest.is_file()  # nothing above mutated shared evidence

    # A stale policy version inside a result is rejected.
    fresh = write_group_irs(tmp_path, group_request(policy_version="2"), rewards=(1, 0))
    stale_result = json.loads(json.dumps(result))
    stale_result["slots"][0]["manifest_path"] = str(
        harness.runtime.manifest_path(fresh[0]["identity"]))
    stale_result["slots"][0]["manifest_sha256"] = digest(fresh[0])
    with pytest.raises(ServiceRolloutError, match="identity differs"):
        load_group_result(harness.run_id, request, stale_result, allow_synthetic=True)


def test_drain_closes_admission(harness):
    client = harness.client()
    client.activate_policy(activated_states(), "1")
    client.drain()
    with pytest.raises(ServiceRolloutError) as exc:
        client.submit_group(group_request())
    assert exc.value.status == 409 and exc.value.code == "admission_closed"
    assert not harness.runtime.registry.accepting


@pytest.mark.parametrize("cancel_whole_run", [False, True])
def test_cancel_and_query_after_timeout(tmp_path, cancel_whole_run):
    release = threading.Event()

    async def slow_executor(config, runtime_, request):
        if request["group_index"] == 0:
            while not release.is_set():
                await asyncio.sleep(0.05)  # cancellable wait
            return write_group_irs(tmp_path, request)
        while True:  # the second group blocks until it is cancelled
            await asyncio.sleep(0.05)

    harness = ServiceHarness(tmp_path, slow_executor, group_wait_timeout=0.2)
    try:
        client = harness.client()
        client.activate_policy(activated_states(), "1")
        request = group_request()
        # Short server wait: the response must say running and hint a query.
        pending = harness.service.submit_group(harness.run_id, request)
        assert pending["status"] in ("queued", "running")
        assert "hint" in pending
        status = client.group_status("group-0")
        assert status["status"] in ("queued", "running")
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and status["status"] in ("queued", "running"):
            time.sleep(0.05)
            status = client.group_status("group-0")
        assert status["status"] == "complete"

        request2 = group_request(group_index=1)
        submitted = harness.service.submit_group(harness.run_id, request2)
        assert submitted["status"] in ("queued", "running")
        if cancel_whole_run:
            result = harness.service.cancel_run(harness.run_id)
            assert result["groups"] == {"group-0": "complete", "group-1": "cancelled"}
            cancelled = client.group_status("group-1")
        else:
            cancelled = client.cancel_group("group-1")
        assert cancelled["status"] == "cancelled"
        with pytest.raises(ServiceRolloutError, match="cancelled|not complete"):
            load_group_result(harness.run_id, request2, cancelled)
    finally:
        release.set()
        harness.close()


def test_register_run_conflict_on_different_manifest(tmp_path):
    async def executor(config, runtime_, request):
        return write_group_irs(tmp_path, request)

    harness = ServiceHarness(tmp_path, executor)
    try:
        from harborrl.rollout.service import ServiceError

        same_manifest = {"run_id": "run-1", "group_size": 2}
        assert harness.service.register_run("run-1", same_manifest)["existed"]
        with pytest.raises(ServiceError) as exc:
            harness.service.register_run("run-1", {"run_id": "run-1", "x": 1})
        assert exc.value.status == 409
    finally:
        harness.close()


def test_service_group_request_and_run_service_groups(tmp_path, monkeypatch):
    """Full SlimeAdapter path: fake Slime samples -> HTTP -> IRs bound to samples."""
    from harborrl.backends.slime_v032 import rollout as slime_rollout

    calls = []

    async def executor(config, runtime_, request):
        calls.append(request["group_id"])
        return write_group_irs(tmp_path, request)

    harness = ServiceHarness(tmp_path, executor)
    try:
        monkeypatch.setenv("HARBORRL_ROLLOUT_SERVICE_URL", harness.url)
        monkeypatch.setenv("HARBORRL_ROLLOUT_SERVICE_TOKEN", harness.token)
        monkeypatch.setenv("RUN_ID", harness.run_id)
        monkeypatch.setenv("RUN_DIR", str(tmp_path))

        class Sample:
            class Status:
                COMPLETED = "completed"

            def __init__(self, group_index):
                self.metadata = {"native_task": dict(TASK)}
                self.group_index = group_index

        args = SimpleNamespace(rollout_max_response_len=SAMPLING["max_new_tokens"],
                               rollout_temperature=1.0, rollout_top_p=1.0,
                               rollout_top_k=-1, rollout_stop=[],
                               sglang_router_ip="127.0.0.1", sglang_router_port=30000)
        group = [Sample(0), Sample(0)]
        monkeypatch.setenv("HARBORRL_ROLLOUT_SERVICE_TOKEN", harness.token)
        # The adapter announces serving and derives the admitted version from
        # the live pool itself before requesting groups.
        from harborrl.backends.slime_v032 import rollout_service_client

        monkeypatch.setattr(rollout_service_client, "fetch_pool_states",
                            lambda endpoint, **kwargs: activated_states())

        with pytest.raises(ServiceRolloutError, match='synthetic_evidence'):
            asyncio.run(slime_rollout.run_service_groups(args, [group], 0))
        assert all('native_ir' not in sample.metadata for sample in group)
        assert calls == ["group-0"]
        assert harness.service.runs[harness.run_id].serving == (
            "http://127.0.0.1:30000")

        request = service_group_request(
            group, slime_rollout.sampling_params(args), 0, "1")
        assert request["group_id"] == "group-0"
        assert request["group_index"] == 0
        assert [slot["slot_id"] for slot in request["slots"]] == ["0", "1"]
    finally:
        harness.close()


def test_announce_factory_attaches_config_and_runtime(tmp_path):
    """A GPU announce through the factory makes the run executable alone."""
    async def executor(config, runtime_, request):
        assert config is runtime_.config
        return write_group_irs(tmp_path, request)

    service = RolloutService(run_root=tmp_path, executor=executor, allow_synthetic=True,
                             group_wait_timeout=5,
                             runtime_factory=lambda run_id, router: SimpleNamespace(
                                 registry=FakeRegistry(), rollout_root=tmp_path,
                                 manifest_path=lambda ident: manifest_path_from(tmp_path, ident),
                                 config=SimpleNamespace(drain_timeout=1)))
    try:
        registration = service.register_run("run-9", {"run_id": "run-9"})
        result = service.announce_serving("run-9", "http://gpu:30000/")
        assert result["runtime"] == "attached"
        assert service.runs["run-9"].serving == "http://gpu:30000"
        assert service.runs["run-9"].config is not None
        service.activate_policy("run-9", activated_states(), "1")
        submitted = service.submit_group("run-9", group_request())
        assert submitted["status"] == "complete"
        assert registration["run_token"]
    finally:
        service.close()


def test_service_group_request_rejects_mixed_groups():
    class Sample:
        def __init__(self, group_index, task):
            self.metadata = {"native_task": dict(task)}
            self.group_index = group_index

    other = dict(TASK)
    other["revision"] = "2"
    with pytest.raises(ValueError, match="mixes task locks"):
        service_group_request([Sample(0, TASK), Sample(0, other)], dict(SAMPLING), 0, "1")
    with pytest.raises(ValueError, match="group identity"):
        service_group_request([Sample(0, TASK), Sample(1, TASK)], dict(SAMPLING), 0, "1")
