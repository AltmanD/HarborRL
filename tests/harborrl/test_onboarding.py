"""Onboarding contracts: drafts, shared Local environment, real preparation and CPU lesson."""
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_options import answers as options_answers

from harborrl.config import ROOT, native
from harborrl.hello import main as hello_main
from harborrl.options.cli import main
from harborrl.options.fingerprints import prepare_model
from harborrl.options.onboarding import load_draft, reference_examples, save_draft
from harborrl.options.presets import MissingAnswers, build_configs, defaults
from harborrl.options.validation import validate_mappings
from harborrl.options.wizard import build_config_bundle

answers = options_answers


def test_local_environment_applies_once_to_every_role(answers, tmp_path):
    shared = {"source_dir": str(ROOT), "python": sys.executable}
    answers["local_environment"] = shared
    answers["training"]["harbor"]["python"] = "/old/runner"
    answers["deployment"] = {"gpu_target": {"train_python": "/old/gpu"},
                             "manager": {"source_dir": "/old/controller"},
                             "storage": {"run_root": str(tmp_path / "custom-storage")}}
    training, deployment = build_configs(answers, tmp_path / "site")
    assert training["harbor"]["python"] == deployment["manager"]["controller_python"] == deployment["gpu_target"]["train_python"] == sys.executable
    assert deployment["manager"]["source_dir"] == deployment["gpu_target"]["source_dir"] == str(ROOT)
    assert training["output"]["root"] == deployment["storage"]["run_root"] == str(tmp_path / "custom-storage")
    t, d = defaults("embedded-local", tmp_path)
    assert t["harbor"]["python"] == d["gpu_target"]["train_python"] == d["manager"]["controller_python"] == sys.executable


def test_blank_shared_environment_reports_visible_fields(answers, tmp_path):
    answers["local_environment"] = {"source_dir": "", "python": ""}
    with pytest.raises(MissingAnswers) as caught:
        build_config_bundle(answers, output_directory=tmp_path)
    assert [row["path"] for row in caught.value.fields] == ["local_environment.python", "local_environment.source_dir"]
    assert all(row["help"] and row["step"] == 0 for row in caught.value.fields)


def test_incomplete_draft_revision_restart_and_publish_separation(tmp_path):
    path = tmp_path / "draft.json"
    incomplete = {"answers_version": 1, "preset": "embedded-local", "training": {"model": {"checkpoint": ""}}}
    assert save_draft(path, incomplete)["revision"] == 1
    reopened = load_draft(path)
    assert reopened["answers"] == incomplete
    with pytest.raises(ValueError, match="changed"):
        save_draft(path, incomplete)
    incomplete["training"]["model"]["checkpoint"] = "/new/model"
    assert save_draft(path, incomplete, revision=reopened["revision"])["revision"] == 2
    with pytest.raises(ValueError, match="changed"):
        save_draft(path, incomplete, revision=1)
    assert load_draft(path)["answers"] == incomplete
    with pytest.raises(MissingAnswers):
        build_config_bundle(incomplete, output_directory=tmp_path / "runnable")
    assert not (tmp_path / "runnable").exists()
    assert not list(tmp_path.glob(".*.lock"))


def test_draft_rejects_unknown_fields_and_symlink(tmp_path):
    answers = {"answers_version": 1, "preset": "embedded-local", "training": {"typo": 1}}
    with pytest.raises(ValueError, match="unknown"):
        save_draft(tmp_path / "bad.json", answers)
    outside = tmp_path / "outside.json"
    outside.write_text("keep")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="regular file"):
        save_draft(link, {"answers_version": 1, "preset": "embedded-local"})
    assert outside.read_text() == "keep"


@pytest.mark.parametrize("ending", [":save", "EOF", "interrupt"])
def test_cli_keeps_partial_progress_and_resumes(tmp_path, monkeypatch, capsys, ending):
    output = tmp_path / "site"
    shared_source = str(tmp_path / "my-checkout")
    responses = iter(["1", shared_source])
    def reply(_):
        try:
            return next(responses)
        except StopIteration:
            if ending == "EOF":
                raise EOFError
            if ending == "interrupt":
                raise KeyboardInterrupt
            return ending
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", reply)
    assert main(["init", "--out-dir", str(output)]) == 0
    path = Path(str(output) + ".draft.json")
    assert load_draft(path)["answers"]["local_environment"]["source_dir"] == shared_source
    assert not output.exists()
    monkeypatch.setattr("builtins.input", lambda _: ":save")
    assert main(["init", "--out-dir", str(output), "--resume", str(path)]) == 0
    assert "Draft kept" in capsys.readouterr().out
    assert load_draft(path)["answers"]["local_environment"]["source_dir"] == shared_source


@pytest.mark.parametrize("ending", [":save", "EOF", "interrupt"])
def test_cli_keeps_prepared_draft_at_publish_prompt(answers, tmp_path, monkeypatch, capsys, ending):
    output = tmp_path / "ready"
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("harborrl.options.cli.interactive_answers", lambda *args, **kwargs: answers)
    monkeypatch.setattr("harborrl.options.fingerprints.prepare_model",
                        lambda *args: {"fingerprints": answers["training"]["gateway"]})
    def reply(_):
        if ending == "EOF":
            raise EOFError
        if ending == "interrupt":
            raise KeyboardInterrupt
        return ending
    monkeypatch.setattr("builtins.input", reply)
    assert main(["init", "--out-dir", str(output)]) == 0
    draft = load_draft(str(output) + ".draft.json")
    assert draft["answers"] == answers
    assert "Draft saved" in capsys.readouterr().out
    assert not output.exists()


def test_cli_reference_and_empty_draft_without_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    output = tmp_path / "empty"
    assert main(["init", "--out-dir", str(output), "--save-draft"]) == 0
    assert load_draft(str(output) + ".draft.json")["answers"]["preset"] == "embedded-local"
    for example in ("local", "sandbox"):
        output = tmp_path / example
        assert main(["init", "--out-dir", str(output), "--example", example]) == 0
        draft = load_draft(str(output) + ".draft.json")
        assert draft["answers"]["training"]["model"]["checkpoint"]
        assert draft["answers"]["training"]["gateway"]["tokenizer_digest"] is None
        assert not output.exists()


def test_examples_are_reference_recipes_not_fake_ready_configs(tmp_path):
    local, sandbox = reference_examples(tmp_path / "site")
    assert local["answers"]["local_environment"] == {"source_dir": str(ROOT), "python": sys.executable}
    assert sandbox["answers"]["sandbox_config"].endswith("sandbox.yaml")
    assert sandbox["answers"]["deployment"]["gpu_target"]["ssh"]["host"]
    for example in (local, sandbox):
        assert example["status"] == "reference" and example["notes"]
        with pytest.raises(MissingAnswers) as caught:
            build_configs(example["answers"], tmp_path)
        assert {row["path"] for row in caught.value.fields} == {"training.gateway.tokenizer_digest", "training.gateway.template_digest"}


def test_prepare_runs_target_python_and_real_fingerprint_function(tmp_path, monkeypatch):
    # A tiny tokenizer test double avoids a Transformers install while exercising the actual child script and hashing function.
    module = tmp_path / "transformers.py"
    module.write_text('''import os
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        assert checkpoint == "fixture-model"
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return cls()
    def get_vocab(self):
        return {"hello": 1, "ok": 2}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [1, 2]
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    answers = {"answers_version": 1, "preset": "embedded-local", "local_environment": {"source_dir": str(ROOT), "python": sys.executable},
               "training": {"model": {"checkpoint": "fixture-model"}}}
    result = prepare_model(answers, tmp_path)
    from harborrl.trajectories.native import digest
    assert result["target"] == "local" and result["python"] == sys.executable
    assert result["fingerprints"]["tokenizer_digest"] == digest({"class": "AutoTokenizer", "vocab": [("hello", 1), ("ok", 2)]})
    from harborrl.options.fingerprints import SCRIPT
    assert "tokenizer_fingerprint(tokenizer)" in SCRIPT
    assert len(result["fingerprints"]["template_digest"]) == 64


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "invalid"])
def test_prepare_failure_is_actionable(tmp_path, monkeypatch, failure):
    answers = {"answers_version": 1, "preset": "embedded-local", "training": {"model": {"checkpoint": "fixture-model"}}}
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 1 if failure == "nonzero" else 0, "HARBORRL_FINGERPRINT={}", "missing tokenizer")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="preparation|tokenizer"):
        prepare_model(answers, tmp_path, timeout=1)


def test_remote_preparation_quotes_paths_and_uses_batch_ssh(tmp_path, monkeypatch):
    answers = {"answers_version": 1, "preset": "service-sandbox-local-runner",
               "training": {"model": {"checkpoint": "/models/model with 'quote"}},
               "deployment": {"gpu_target": {"ssh": {"host": "gpu-example"}, "source_dir": "/srv/code space", "train_python": "/srv/venv/bin/python"}}}
    def run(command, **kwargs):
        assert command[0] == "ssh" and "BatchMode=yes" in command
        assert "gpu-example" in command and kwargs["cwd"] is None
        assert "cd '/srv/code space'" in command[-1] and "HF_HUB_OFFLINE=1" in command[-1]
        assert "'\"'\"'" in command[-1]
        return subprocess.CompletedProcess(command, 0, "HARBORRL_FINGERPRINT=" + json.dumps({"tokenizer_digest": "a" * 64, "template_digest": "b" * 64}), "")
    monkeypatch.setattr(subprocess, "run", run)
    assert prepare_model(answers, tmp_path)["target"] == "gpu-example"


def test_hello_artifacts_catalog_profile_and_fixture_boundary(tmp_path):
    root = tmp_path / "hello"
    assert hello_main(["--out-dir", str(root)]) == 0
    summary = json.loads((root / "hello-result.json").read_text())
    assert summary["status"] == "passed" and summary["rewards"] == [0.0, 1.0]
    assert summary["advantages"][0] < 0 < summary["advantages"][1]
    for index in (0, 1):
        ir = json.loads((root / f"attempt-{index}" / "native-ir.json").read_text())
        assert ir["readiness"] == "CONTRACT_READY"
        assert "hello_world_fixture" in json.dumps(ir)
        assert "hello_world_verifier" in json.dumps(ir)
    t, d = defaults("embedded-local", tmp_path / "config")
    t["tasks"]["catalog"] = summary["catalog"]
    t["harness"]["profile"] = summary["profile"]
    t["model"].update(checkpoint=str(root), reference=str(root))
    t["gateway"].update(tokenizer_digest="1" * 64, template_digest="2" * 64)
    assert native.catalog(t)[0]["id"] == "hello_world"
    report = validate_mappings(t, d, config_path=tmp_path / "config/config.yaml", check_files=True)
    assert report["status"] != "invalid", report
    before = (root / "hello-result.json").read_bytes()
    assert hello_main(["--out-dir", str(root)]) == 1
    assert (root / "hello-result.json").read_bytes() == before


def test_saved_local_bundle_reopens_shared_environment(answers, tmp_path):
    from harborrl.webui.server import Console
    console = Console(tmp_path / "web")
    try:
        console.save({"name": "local", "answers": answers})
        reopened = console.configuration("local")["answers"]
        assert reopened["local_environment"]["python"] == sys.executable
        edited = deepcopy(reopened)
        edited["local_environment"]["python"] = str(tmp_path / "new-python")
        bundle = build_config_bundle(edited, output_directory=tmp_path / "new")
        t = yaml.safe_load(bundle["files"]["config.yaml"])
        d = yaml.safe_load(bundle["files"]["deployment.yaml"])
        assert t["harbor"]["python"] == d["gpu_target"]["train_python"] == d["manager"]["controller_python"] == str(tmp_path / "new-python")
    finally:
        console.jobs.close()
