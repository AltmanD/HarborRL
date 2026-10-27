"""HTTP and process lifecycle contracts without allocating training resources."""
import http.client
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml
from test_options import answers as options_answers

from harborrl.options.wizard import build_config_bundle
from harborrl.webui.cli import workspace_lock
from harborrl.webui.jobs import Jobs
from harborrl.webui.server import Console, make_server

answers = options_answers


@pytest.fixture
def console(tmp_path):
    instance = Console(tmp_path / "web")
    yield instance
    instance.jobs.close()


@pytest.fixture
def server(console):
    server = make_server(console)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield server
    server.shutdown()
    thread.join(5)
    server.server_close()


def request(server, console, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    defaults = {"Authorization": "Bearer " + console.token, "Content-Type": "application/json",
                "Origin": f"http://127.0.0.1:{server.server_port}"}
    connection.request(method, path, body=json.dumps(body) if body is not None else None,
                       headers={**defaults, **(headers or {})})
    response = connection.getresponse()
    payload = response.read()
    result = (response.status, dict(response.getheaders()), payload)
    connection.close()
    return result


def test_static_and_bootstrap(server, console):
    for path in ("/", "/app.js", "/style.css", "/lucide.min.js", "/brand.jpg"):
        status, headers, payload = request(server, console, "GET", path)
        assert status == 200 and payload
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert console.token.encode() not in payload
    status, _, payload = request(server, console, "GET", "/api/bootstrap")
    assert status == 200
    assert json.loads(payload)["catalog"]["catalog_version"] == 1


@pytest.mark.parametrize("headers", [{"Authorization": "Bearer invalid"}, {"Origin": "https://external.example"},
                                    {"Host": "external.example"}, {"Content-Type": "text/plain"}])
def test_request_boundary(server, console, headers):
    status, _, _ = request(server, console, "POST", "/api/configurations", {}, headers=headers)
    assert status in {403, 415}
    assert not console.list_configurations()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "", "a/b", "a\\b", "a b", ".hidden"])
def test_output_paths(console, answers, name):
    with pytest.raises(ValueError):
        console.save({"name": name, "answers": answers})


def test_preview_save_and_snapshot(server, console, answers):
    body = {"name": "experiment", "answers": answers, "check_files": True}
    status, _, payload = request(server, console, "POST", "/api/preview", body)
    assert status == 200, payload
    assert json.loads(payload)["validation"]["status"] == "valid_with_deferred_checks"
    assert not console.directory("experiment").exists()
    status, _, payload = request(server, console, "POST", "/api/configurations", body)
    assert status == 201, payload
    assert request(server, console, "POST", "/api/configurations", body)[0] == 400
    status, _, payload = request(server, console, "POST", "/api/jobs", {"configuration": "experiment", "mode": "dry-run"})
    assert status == 201, payload
    job = json.loads(payload)
    for _ in range(600):
        record = console.jobs.list()[0]
        if record["state"] != "running":
            break
        time.sleep(.05)
    assert record["state"] == "succeeded", console.jobs.log(job["id"])
    snapshot = Path(record["snapshot"])
    training = yaml.safe_load((snapshot / "config.yaml").read_text())
    deployment = yaml.safe_load((snapshot / "deployment.yaml").read_text())
    assert training["output"]["root"] == deployment["storage"]["run_root"] == record["output_root"]
    saved = console.directory("experiment") / "config.yaml"
    assert yaml.safe_load(saved.read_text())["output"]["root"] != record["output_root"]
    assert json.loads(console.jobs.log(job["id"])["text"])["config"]["schema_version"] == 2


@pytest.mark.parametrize("mode", ["train", "doctor", "shell"])
def test_live_requires_confirmation(console, answers, mode, monkeypatch):
    console.save({"name": "test", "answers": answers})
    def forbidden(*args, **kwargs):
        raise AssertionError("Unconfirmed live job must not execute")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        console.start({"configuration": "test", "mode": mode})


def test_process_lifecycle(console, answers, tmp_path, monkeypatch):
    console.save({"name": "test", "answers": answers})
    real_popen = subprocess.Popen
    calls = []
    def stand_in(command, **kwargs):
        calls.append(command)
        return real_popen([sys.executable, "-u", "-c", "import time; print('training stand-in', flush=True); time.sleep(30)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", stand_in)
    job = console.start({"configuration": "test", "mode": "train", "confirmed": True})
    assert calls[0][4] == "train" and "--dry-run" not in calls[0]
    with pytest.raises(ValueError, match="active"):
        console.start({"configuration": "test", "mode": "dry-run"})
    for _ in range(600):
        if console.jobs.log(job["id"])["text"]:
            break
        time.sleep(.05)
    child = console.jobs.child
    stopped = console.jobs.stop(job["id"])
    assert stopped["state"] == "stopping"
    child.wait(timeout=5)
    assert console.jobs.list()[0]["state"] == "stopped"
    assert console.jobs.log(job["id"])["text"]
    recovered = Jobs(console.jobs.root)
    assert recovered.list()[0]["state"] == "stopped"


def test_restart_blocks_unknown_job(console, answers, tmp_path):
    root = console.jobs.root / "abc"
    root.mkdir()
    (root / "job.json").write_text(json.dumps({"id": "abc", "state": "running", "created_at": "2026-10-22"}))
    recovered = Jobs(console.jobs.root)
    assert recovered.list()[0]["state"] == "unknown"
    with pytest.raises(ValueError, match="interrupted"):
        recovered.start("test", build_config_bundle(answers, output_directory=tmp_path / "out"), "dry-run")
    recovered.reconcile("abc")
    assert recovered.list()[0]["state"] == "interrupted"


def test_workspace_lock(tmp_path):
    with workspace_lock(tmp_path), pytest.raises(ValueError, match="Another console"), workspace_lock(tmp_path):
        pass
    with workspace_lock(tmp_path):
        pass


def test_symlink_cannot_escape(console, tmp_path, answers):
    outside = tmp_path / "outside"
    outside.mkdir()
    (console.configurations / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        console.save({"name": "escape", "answers": answers})
    assert not list(outside.iterdir())


def test_failed_spawn_and_log_limit(console, answers, tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("injected launch failure")
    monkeypatch.setattr(subprocess, "Popen", failed)
    bundle = build_config_bundle(answers, output_directory=tmp_path / "out")
    job = console.jobs.start("test", bundle, "dry-run")
    assert job["state"] == "failed"
    Path(job["log_path"]).write_bytes(b"x" * 100000)
    result = console.jobs.log(job["id"])
    assert len(result["text"]) == 65536 and result["truncated"]
    training_log = Path(job["output_root"]) / "training/one/train-stdout.log"
    training_log.parent.mkdir(parents=True)
    training_log.write_text("step=1 reward=0.5")
    assert console.jobs.log(job["id"])["training_text"] == "step=1 reward=0.5"


def test_import_and_bad_body(server, console):
    status, _, payload = request(server, console, "POST", "/api/import", {"text": "answers_version: 1\npreset: embedded-local"})
    assert status == 200 and json.loads(payload)["answers"]["preset"] == "embedded-local"
    assert request(server, console, "POST", "/api/import", {"text": "[]"})[0] == 400
    assert request(server, console, "GET", "/../../pyproject.toml")[0] == 404


@pytest.mark.parametrize("text", ["answers_version: true\npreset: embedded-local", "answers_version: 1\npreset: embedded-local\ntraining: []", "answers_version: 1\npreset: embedded-local\nunknown: x"])
def test_import_rejects_invalid_answers(server, console, text):
    assert request(server, console, "POST", "/api/import", {"text": text})[0] == 400


def test_preview_never_starts_process(server, console, answers, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not start a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert request(server, console, "POST", "/api/preview", {"name": "offline", "answers": answers, "check_files": True})[0] == 200


def test_local_stop_reaches_descendants(tmp_path):
    import os
    import signal

    from harborrl.config import ROOT
    script = '''
import os, signal, subprocess, sys, time
from pathlib import Path
from harborrl.deployment.training_target import LocalTrainingTarget
root = Path(sys.argv[1])
worker = "import signal,sys,time; from pathlib import Path; signal.signal(signal.SIGINT, lambda *a: (Path(sys.argv[1]).write_text('stopped'), sys.exit(0))); Path(sys.argv[1]+'.ready').touch(); time.sleep(30)"
parent = "import subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); Path(sys.argv[2]+'.pid').write_text(str(child.pid)); time.sleep(30)"
target = LocalTrainingTarget([sys.executable, '-c', parent, worker, str(root / 'worker')], cwd=root, env=os.environ.copy())
target.launch()
(root / 'group').write_text(str(target.child.pid))
sys.exit(target.wait())
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            if (tmp_path / "worker.ready").exists():
                break
            time.sleep(.05)
        assert (tmp_path / "worker.ready").exists()
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        for _ in range(100):
            if (tmp_path / "worker").exists():
                break
            time.sleep(.05)
        assert (tmp_path / "worker").read_text() == "stopped"
    finally:
        if (tmp_path / "group").exists():
            try:
                os.killpg(int((tmp_path / "group").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
