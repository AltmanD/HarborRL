"""Deployment preflight graph: role-layered checks with faked SSH transport."""
from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

from test_deployment_config import deployment_mapping
from test_native_training import native_config

from harborrl.deployment import preflight as deployment_preflight
from harborrl.deployment.schema import embedded_local, validate


def test_bindable_rejects_existing_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        ok, _ = deployment_preflight._bindable(*listener.getsockname())
        assert not ok


def test_bindable_allows_http_server_restart_after_connection_close():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        address = listener.getsockname()
        with socket.create_connection(address, timeout=2) as client:
            accepted, _ = listener.accept()
            with accepted:
                accepted.settimeout(2)
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b''
                client.shutdown(socket.SHUT_WR)
                assert accepted.recv(1) == b''
    ok, detail = deployment_preflight._bindable(*address)
    assert ok, detail


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def fake_remote(responses):
    """Responses keyed by command prefix; unmatched shell tests succeed."""
    calls = []
    probe = json.dumps({"harbor_version": "0.23.0", "python": "3.12.13"})

    def run_ssh(spec, command, *, timeout=30, connect_timeout=15):
        calls.append((dict(spec), command))
        for prefix, result in responses.items():
            if command.startswith(prefix):
                return completed(**result) if isinstance(result, dict) else result
        if "runner --probe" in command:
            return completed(stdout=probe)
        if command.startswith("test "):
            return completed(stdout="ok\n")  # shell test protocol: && echo ok
        return completed()

    return run_ssh, calls


def resolvable(deployment):
    """Point service URLs at loopback so preflight resolution checks pass."""
    manager = deployment["manager"]
    manager["control_url"] = f"http://127.0.0.1:{manager['control_port']}"
    manager["gateway_url"] = f"http://127.0.0.1:{manager['gateway_port']}"
    return deployment


def service_deployment(tmp_path):
    return validate(deployment_mapping(tmp_path))


def training_plan(tmp_path):
    config_path, task = native_config(tmp_path)
    from harborrl.config import load_config
    from harborrl.data.harbor.native_inspector import inspect_native

    catalog_path = Path(
        json.loads(config_path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"])
    catalog = json.loads(catalog_path.read_text())
    catalog[0]["task_digest"] = inspect_native(task)["task_digest"]
    catalog_path.write_text(json.dumps(catalog))
    return {"config": load_config(config_path)}


def by_name(checks, name):
    return [check for check in checks if check.get("check") == name or check.get("path") == name]


def test_gpu_layer_probes_remote_machine_semantics(tmp_path, monkeypatch):
    plan, deployment = training_plan(tmp_path), resolvable(service_deployment(tmp_path))
    (tmp_path / "runs").mkdir()
    (tmp_path / "secrets").mkdir()
    token = tmp_path / "secrets" / "run-token"
    token.write_text("token\n")
    token.chmod(0o600)
    run_ssh, calls = fake_remote({
        "nvidia-smi": completed(stdout="0\n1\n2\n3\n4\n5\n6\n7\n"),
    })
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    assert all(check["ok"] for check in checks), checks
    gpu_checks = [check for check in checks if check["role"] == "gpu_target"]
    assert {check["check"] for check in gpu_checks} >= {
        "gpu_target.ssh", "gpu_target.source_dir", "gpu_target.train_python",
        "gpu_target.expected_gpus"}
    assert all(check["machine"] == "gpu-8x-h200" for check in gpu_checks)
    # Model paths are evaluated on the GPU machine, not from the manager.
    tested = [command for _, command in calls if "test -d" in command]
    assert any("/model" in command for command in tested)


def test_gpu_layer_reports_shortage_and_unreachable_ssh(tmp_path, monkeypatch):
    plan, deployment = training_plan(tmp_path), service_deployment(tmp_path)
    (tmp_path / "runs").mkdir()
    token = tmp_path / "secrets" / "run-token"
    token.parent.mkdir()
    token.write_text("token\n")
    token.chmod(0o600)
    run_ssh, _ = fake_remote({"nvidia-smi": completed(stdout="0\n1\n")})
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    shortage = by_name(checks, "gpu_target.expected_gpus")
    assert shortage and not shortage[0]["ok"]
    assert "visible GPUs" in shortage[0]["detail"]

    run_ssh, _ = fake_remote({"": completed(returncode=255, stderr="Permission denied")})
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    unreachable = by_name(checks, "gpu_target.ssh")
    assert unreachable and not unreachable[0]["ok"]
    assert unreachable[0]["role"] == "gpu_target"
    assert "hint" in unreachable[0]


def test_worker_layer_validates_runner_runtime_and_docker(tmp_path, monkeypatch):
    plan, deployment = training_plan(tmp_path), service_deployment(tmp_path)
    (tmp_path / "runs").mkdir()
    token = tmp_path / "secrets" / "run-token"
    token.parent.mkdir()
    token.write_text("token\n")
    token.chmod(0o600)
    probe = json.dumps({"harbor_version": "0.23.0", "python": "3.12.13"})
    run_ssh, _ = fake_remote({
        "docker info": completed(returncode=1, stderr="Cannot connect to the Docker daemon"),
        "cd ": completed(stdout=probe),
    })
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    docker = [check for check in checks if check["check"] == "worker.docker"]
    assert len(docker) == 2 and not any(check["ok"] for check in docker)
    assert all("Docker daemon" in check["hint"] for check in docker)
    harbor = [check for check in checks if check["check"] == "worker.harbor_version"]
    assert all(check["ok"] for check in harbor)

    run_ssh, _ = fake_remote({"cd ": completed(stdout=json.dumps(
        {"harbor_version": "0.22.0", "python": "3.12.13"}))})
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    harbor = [check for check in checks if check["check"] == "worker.harbor_version"]
    assert not any(check["ok"] for check in harbor)
    # a wrong python version is also a failed runtime gate
    run_ssh, _ = fake_remote({"cd ": completed(stdout=json.dumps(
        {"harbor_version": "0.23.0", "python": "3.11.4"}))})
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = deployment_preflight.preflight(plan, deployment)
    runtime = [check for check in checks if check["check"] == "worker.runner_runtime"]
    assert not any(check["ok"] for check in runtime)


def test_manager_layer_checks_storage_token_and_ports(tmp_path):
    plan = training_plan(tmp_path)
    deployment = resolvable(service_deployment(tmp_path))
    checks = deployment_preflight.preflight(plan, deployment)
    assert not by_name(checks, "storage.run_root")[0]["ok"]  # run root missing

    (tmp_path / "runs").mkdir()
    token = tmp_path / "secrets" / "run-token"
    token.parent.mkdir()
    token.write_text("token\n")
    token.chmod(0o644)
    checks = deployment_preflight.preflight(plan, deployment)
    assert not by_name(checks, "manager.run_token_file")[0]["ok"]
    token.chmod(0o600)
    checks = deployment_preflight.preflight(plan, deployment)
    assert by_name(checks, "manager.run_token_file")[0]["ok"]
    assert by_name(checks, "manager.control_port")[0]["ok"]


def test_control_loop_layer_defers_live_dials_and_inference(tmp_path, monkeypatch):
    plan, deployment = training_plan(tmp_path), service_deployment(tmp_path)
    (tmp_path / "runs").mkdir()
    token = tmp_path / "secrets" / "run-token"
    token.parent.mkdir()
    token.write_text("token\n")
    token.chmod(0o600)
    run_ssh, _ = fake_remote({})
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    monkeypatch.setattr(deployment_preflight.socket, "getaddrinfo",
                        lambda host, port: [(None, None, None, None, (host, port))])
    checks = deployment_preflight.preflight(plan, deployment)
    inference = by_name(checks, "connectivity.manager_to_gpu_inference")
    assert inference and inference[0]["ok"] and "deferred" in inference[0]["detail"]
    dials = [check for check in checks if check["check"].endswith(".dial")]
    assert dials and all("deferred" not in check["detail"] or check["ok"] for check in dials)


def test_doctor_gates_local_checks_on_gpu_transport(tmp_path, monkeypatch):
    from harborrl.platform import preflight as platform_preflight

    plan = training_plan(tmp_path)
    plan["deployment"] = {"config": service_deployment(tmp_path),
                          "source": "deployment.yaml"}
    run_ssh, _ = fake_remote({"nvidia-smi": completed(stdout="0\n" * 8)})
    monkeypatch.setattr(platform_preflight, "_deployment",
                        lambda p: p.get("deployment", {}).get("config"))
    monkeypatch.setattr(deployment_preflight.remote, "run_ssh", run_ssh)
    checks = platform_preflight.doctor(plan)
    # Local GPU dependency probes do not run for a remote GPU target.
    assert not any(check.get("dependency") == "ray" for check in checks)
    assert not any(check.get("resource") == "visible GPU budget" for check in checks)
    assert not any(check.get("service") == "external Harbor worker" for check in checks)
    assert any(check.get("role") == "gpu_target" for check in checks)

    embedded = {"config": embedded_local(run_root=str(tmp_path / "runs")),
                "source": "deployment.yaml"}
    plan["deployment"] = embedded
    checks = platform_preflight.doctor(plan)
    assert any(check.get("dependency") == "ray" for check in checks)
    assert any(check.get("resource") == "visible GPU budget" for check in checks)

    # The auto-derived embedded-local topology keeps the legacy check set.
    plan["deployment"] = {"config": embedded_local(), "source": "embedded-local"}
    legacy = platform_preflight.doctor(plan)
    assert not any(check.get("role") == "gpu_target" for check in legacy)
    assert any(check.get("dependency") == "ray" for check in legacy)
