"""Offline wizard contracts, including the existing training CLI boundary."""
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_native_training import native_config

from harborrl.options.catalog import TRAINING_FIELDS, build_catalog, dataset_metadata
from harborrl.options.cli import main
from harborrl.options.presets import PRESETS, worker_defaults
from harborrl.options.validation import validate_paths
from harborrl.options.wizard import build_config_bundle, publish_bundle


@pytest.fixture
def answers(tmp_path):
    _path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    catalog = tmp_path / "catalog.json"
    entries = json.loads(catalog.read_text())
    entries[0]["task_digest"] = inspect_native(task)["task_digest"]
    catalog.write_text(json.dumps(entries))
    model = tmp_path / "model"
    model.mkdir()
    return {"answers_version": 1, "preset": "embedded-local", "training": {
        "tasks": {"catalog": str(catalog)}, "harness": {"profile": str(tmp_path / "claude.json")},
        "harbor": {"python": sys.executable},
        "model": {"checkpoint": str(model), "reference": str(model)},
        "gateway": {"tokenizer_digest": "1" * 64, "template_digest": "2" * 64},
    }}


def service_answers(answers, preset):
    answers = deepcopy(answers)
    answers["preset"] = preset
    answers["deployment"] = {
        "manager": {"control_url": "http://controller.example:49200", "gateway_url": "http://controller.example:49184"},
        "gpu_target": {"ssh": {"host": "gpu", "user": "trainer", "port": 22},
                       "source_dir": "/remote/source", "train_python": "/remote/python"},
    }
    if preset == "service-ssh":
        worker = worker_defaults()
        worker["ssh"]["host"] = "runner"
        answers["deployment"]["rollout"] = {"workers": [worker]}
        del answers["training"]["harbor"]["python"]
    else:
        answers["sandbox_config"] = "/site/sandbox.yaml"
    return answers


def test_catalog_contract(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline operation attempted subprocess")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    from harborrl.config.native import FIELDS
    assert set(TRAINING_FIELDS) == {f"{s}.{f}" for s, fields in FIELDS.items() for f in fields}
    first = build_catalog()
    assert json.dumps(first, sort_keys=True) == json.dumps(build_catalog(), sort_keys=True)
    assert all(not row["available"] for row in first["datasets"] if row["status"] == "planned")
    from harborrl.config import ROOT
    assert dataset_metadata() == yaml.safe_load((ROOT / "configs/harbor_hub/manifests.yaml").read_text())


@pytest.mark.parametrize("dimension", [None, "models", "algorithms", "tasks", "harnesses", "datasets", "deployments", "fields", "compat"])
def test_catalog_cli(dimension, capsys):
    assert main(["options", *([dimension] if dimension else []), "--json"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert not captured.err
    assert result["catalog_version"] == 1
    if dimension:
        assert len(result) == 3


@pytest.mark.parametrize("preset", [p["id"] for p in PRESETS])
def test_presets_and_legacy_dry_run(preset, answers, tmp_path, capsys):
    if preset != "embedded-local":
        answers = service_answers(answers, preset)
    original = deepcopy(answers)
    output = tmp_path / "new site"
    bundle = build_config_bundle(answers, output_directory=output)
    assert answers == original
    assert not output.exists()
    publish_bundle(bundle, output)
    result = validate_paths(output / "config.yaml", output / "deployment.yaml")
    assert result["status"] == "valid_with_deferred_checks", result
    if preset != "service-sandbox-local-runner":
        from harborrl.cli import main as legacy_main
        assert legacy_main(["train", str(output / "config.yaml"), "--deployment", str(output / "deployment.yaml"), "--dry-run"]) == 0
        assert json.loads(capsys.readouterr().out)["config"]["schema_version"] == 2


def test_local_files_and_remote_ownership(answers, tmp_path):
    for preset in ("embedded-local", "service-ssh"):
        current = answers if preset == "embedded-local" else service_answers(answers, preset)
        out = tmp_path / preset
        publish_bundle(build_config_bundle(current, output_directory=out), out)
        result = validate_paths(out / "config.yaml", out / "deployment.yaml", check_files=True)
        assert result["status"] == "valid_with_deferred_checks", result
        if preset == "service-ssh":
            assert any(c["code"] == "remote_paths" for c in result["checks"])
    (tmp_path / "claude.json").unlink()
    assert validate_paths(out / "config.yaml", out / "deployment.yaml")["status"] != "invalid"
    assert validate_paths(out / "config.yaml", out / "deployment.yaml", check_files=True)["status"] == "invalid"


@pytest.mark.parametrize("mutation", ["unknown", "missing", "zero_digest", "gpu", "gateway", "provider"])
def test_reject_bad_answers(mutation, answers, tmp_path):
    answers = service_answers(answers, "service-ssh")
    if mutation == "unknown":
        answers["training"]["sampling"] = {"typo": 1}
    elif mutation == "missing":
        del answers["training"]["tasks"]
    elif mutation == "zero_digest":
        answers["training"]["gateway"]["tokenizer_digest"] = "0" * 64
    elif mutation == "gpu":
        answers["deployment"]["gpu_target"]["expected_gpus"] = 0
    elif mutation == "gateway":
        answers["training"]["gateway"]["advertised_url"] = "http://other.example:49184"
    else:
        answers["deployment"]["rollout"]["provider"] = "sandbox"
    with pytest.raises(ValueError):
        build_config_bundle(answers, output_directory=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_noninteractive_and_failure_json(answers, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit) as exc:
        main(["init", "--out-dir", str(tmp_path / "output")])
    assert exc.value.code == 2
    capsys.readouterr()
    assert main(["validate", str(tmp_path / "missing"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
    file = tmp_path / "answers.yaml"
    file.write_text(yaml.safe_dump(answers))
    assert main(["init", "--answers", str(file), "--out-dir", str(tmp_path / "output")]) == 0
    assert "harborrl train" in capsys.readouterr().out
    assert main(["init", "--answers", str(file), "--out-dir", str(tmp_path / "output")]) == 1


def test_publication_failure_lock_and_traversal(answers, tmp_path, monkeypatch):
    output = tmp_path / "output"
    bundle = build_config_bundle(answers, output_directory=output)
    lock = tmp_path / ".output.init.lock"
    lock.write_text("other process")
    with pytest.raises(FileExistsError):
        publish_bundle(bundle, output)
    assert lock.read_text() == "other process"
    lock.unlink()
    bad = deepcopy(bundle)
    bad["files"]["../escape"] = "bad"
    with pytest.raises(ValueError):
        publish_bundle(bad, output)
    def fail(*args, **kwargs):
        raise OSError("injected rename failure")
    with monkeypatch.context() as context:
        context.setattr(Path, "rename", fail)
        with pytest.raises(OSError, match="injected"):
            publish_bundle(bundle, output)
    assert not output.exists() and not list(tmp_path.glob(".output.init*"))
    publish_bundle(bundle, output)
    before = (output / "config.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        publish_bundle(bundle, output)
    assert (output / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("confirmation", ["yes", "no"])
def test_interactive(answers, tmp_path, monkeypatch, confirmation):
    t = answers["training"]
    responses = iter(["1", sys.executable, sys.executable, t["tasks"]["catalog"], t["harness"]["profile"],
                      t["model"]["checkpoint"], t["model"]["reference"], "", "1" * 64, "2" * 64,
                      "", "", "1", "", "", "", "", confirmation])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    output = tmp_path / "interactive"
    assert main(["init", "--out-dir", str(output)]) == 0
    assert output.exists() == (confirmation == "yes")


def test_probe_failure_and_alias_parsing(tmp_path, monkeypatch):
    from harborrl.options.probe import probe_local, ssh_aliases
    config = tmp_path / "ssh-config"
    config.write_text("Host train runner *.example !excluded\n  HostName host\nMatch exec unsafe\n  User user\n")
    assert ssh_aliases(config) == ["runner", "train"]
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 3
        raise subprocess.TimeoutExpired(args[0], 3)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = probe_local(tmp_path / "output")
    assert result["gpu"]["status"] == result["python"]["status"] == "unknown"
    assert not (tmp_path / "output").exists()


def test_sandbox_files_and_remote_runner(answers, tmp_path, capsys):
    from harborrl.cli import main as legacy_main
    from harborrl.data.harbor.native_inspector import inspect_native
    task = tmp_path / "task"
    (task / "task.toml").write_text('version = "1.0"\n[environment]\ndocker_image = "task:locked"\n')
    catalog = tmp_path / "catalog.json"
    entries = json.loads(catalog.read_text())
    entries[0]["task_digest"] = inspect_native(task)["task_digest"]
    catalog.write_text(json.dumps(entries))
    sandbox = tmp_path / "sandbox.yaml"
    sandbox.write_text(yaml.safe_dump({"domain": "https://sandbox.example", "credential_file": "/missing/secret",
                                      "bindings": [{"image": "task:locked", "max_live_instances": 2}]}))
    answers = service_answers(answers, "service-sandbox-local-runner")
    answers["sandbox_config"] = str(sandbox)
    for remote in (False, True):
        if remote:
            worker = worker_defaults()
            worker["ssh"]["host"] = "runner"
            worker["docker"] = False
            answers["deployment"]["rollout"] = {"workers": [worker]}
            answers["training"]["harbor"] = {"runner_transport": "ssh"}
        output = tmp_path / f"sandbox-{remote}"
        publish_bundle(build_config_bundle(answers, output_directory=output), output)
        report = validate_paths(output / "config.yaml", output / "deployment.yaml", check_files=True)
        assert report["status"] == "valid_with_deferred_checks", report
        assert legacy_main(["train", str(output / "config.yaml"), "--deployment", str(output / "deployment.yaml"), "--dry-run"]) == 0
        capsys.readouterr()
    sandbox.write_text(sandbox.read_text().replace("task:locked", "other:locked"))
    assert validate_paths(output / "config.yaml", output / "deployment.yaml", check_files=True)["status"] == "invalid"


def test_offline_import_boundary(answers, tmp_path):
    from harborrl.config import ROOT
    output = tmp_path / "offline"
    publish_bundle(build_config_bundle(answers, output_directory=output), output)
    script = '''
import builtins, socket, subprocess, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'transformers', 'sglang', 'harbor', 'opensandbox', 'httpx'}
    return original(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('offline command attempted an external operation')
builtins.__import__ = guarded
socket.socket = forbidden
subprocess.run = subprocess.Popen = forbidden
from harborrl.cli import main
assert main(['options', '--json']) == 0
assert main(['validate', sys.argv[2], '--deployment', sys.argv[3], '--check-files', '--json']) == 0
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT), str(output / "config.yaml"),
                             str(output / "deployment.yaml")], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_concurrent_publication(answers, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    output = tmp_path / "concurrent"
    bundle = build_config_bundle(answers, output_directory=output)
    barrier = Barrier(2)
    def attempt():
        barrier.wait(timeout=10)
        try:
            publish_bundle(bundle, output)
        except FileExistsError:
            return False
        return True
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: attempt(), range(2))) == [False, True]
    assert {p.name for p in output.iterdir()} == set(bundle["files"])


def test_relative_paths_and_invalid_yaml(answers, tmp_path, capsys):
    answers["training"]["tasks"]["catalog"] = "../catalog.json"
    output = tmp_path / "relative"
    publish_bundle(build_config_bundle(answers, output_directory=output), output)
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["tasks"]["catalog"] == str(tmp_path / "catalog.json")
    bad = tmp_path / "bad.yaml"
    bad.write_text("[invalid")
    assert main(["validate", str(bad), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_embedded_explicit_conflict(answers, tmp_path):
    answers["training"]["gateway"]["advertised_url"] = "http://localhost:49184"
    answers["deployment"] = {"manager": {"gateway_url": "http://127.0.0.1:49184"}}
    with pytest.raises(ValueError, match="conflicts"):
        build_config_bundle(answers, output_directory=tmp_path / "conflict")


def test_options_text_is_concise(capsys):
    assert main(["options"]) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) < 50
    assert "locally_available=False" in output


def test_public_answer_templates(tmp_path):
    from harborrl.config import ROOT
    for file in sorted((ROOT / "examples/options").glob("*.yaml")):
        answers = yaml.safe_load(file.read_text())
        with pytest.raises(ValueError, match="SHA-256"):
            build_config_bundle(answers, output_directory=tmp_path / file.stem)
        answers["training"]["gateway"].update(tokenizer_digest="1" * 64, template_digest="2" * 64)
        assert build_config_bundle(answers, output_directory=tmp_path / file.stem)["validation"]["status"] != "invalid"
