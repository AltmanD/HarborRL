"""Shared onboarding help, reference recipes and resumable draft documents."""
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from harborrl.config import ROOT

from .presets import check_answers, defaults

FIELD_HELP = {
    "local_environment.source_dir": ("Shared source directory", "Absolute HarborRL checkout directory on this machine. Controller, GPU and rollout use this one directory.", 0),
    "local_environment.python": ("Shared Python interpreter", "Absolute Python executable in your training environment, with HarborRL, Harbor and the training dependencies installed. Detected from the process running this wizard.", 0),
    "deployment.manager.source_dir": ("Controller source directory", "Absolute HarborRL checkout directory on the controller running this wizard.", 0),
    "deployment.manager.controller_python": ("Controller Python", "Absolute Python executable on the controller with HarborRL installed.", 0),
    "deployment.manager.control_url": ("Control origin", "Controller HTTP origin reachable from the GPU, including port 49200; for example http://controller.example:49200.", 0),
    "deployment.manager.gateway_url": ("Gateway origin", "Controller HTTP origin reachable from task containers or the configured relay, including port 49184.", 0),
    "deployment.gpu_target.ssh.host": ("GPU SSH destination", "SSH alias from the controller user's SSH config, or user@hostname; test with ssh <alias> first.", 0),
    "deployment.gpu_target.source_dir": ("GPU source directory", "Absolute HarborRL checkout path on the GPU machine, using the same code revision as the controller.", 0),
    "deployment.gpu_target.train_python": ("GPU training Python", "Absolute Python executable on the GPU machine with Transformers and the training backend installed.", 0),
    "training.harbor.python": ("Harbor runner Python", "Absolute Python executable on the runner with Harbor 0.23.0 installed. Sandbox uses a Python 3.12+ runner with the sandbox extra.", 0),
    "deployment.storage.run_root": ("Storage run root", "Writable absolute output directory. Service mode requires the same shared filesystem path on controller, GPU and runners.", 0),
    "deployment.rollout.workers": ("Rollout workers", "Add at least one SSH worker with a checkout, Python interpreter, Docker and task root.", 0),
    "sandbox_config": ("Sandbox configuration", "Path to your Sandbox YAML on the runner; configure the service endpoint, credential file reference and task image bindings. See docs/sandbox-rollout.md.", 0),
    "training.model.args_file": ("Model args preset", "Slime model preset matching the checkpoint architecture; qwen3-8B is the Qwen3 8B example. GPU memory requirements depend on the model.", 1),
    "training.model.checkpoint": ("Checkpoint directory", "Existing Hugging Face model directory on the GPU target containing config, tokenizer and model weights. Used to calculate fingerprints automatically.", 1),
    "training.model.reference": ("Reference checkpoint", "Existing reference model checkpoint on the GPU target in the format required by the training backend, for example its converted torch_dist directory.", 1),
    "training.tasks.catalog": ("Task catalog", "Existing locked JSON/JSONL catalog on the controller. Run harborrl hello for a bundled task and catalog, or harborrl hub --help to prepare your own.", 1),
    "training.harness.profile": ("Claude Code profile", "Existing JSON agent profile with cli_version, model, max_turns and timeouts. Run harborrl hello to generate a working profile example.", 1),
    "training.gateway.tokenizer_digest": ("Automatic model fingerprints", "Select a checkpoint and Python environment, then choose Prepare model. HarborRL calculates both fingerprints with the real tokenizer on the GPU target.", 1),
    "training.gateway.template_digest": ("Automatic chat template fingerprint", "Calculated together with the tokenizer fingerprint by Prepare model; no manual SHA-256 entry is needed.", 1),
    "training.gateway.host": ("Gateway bind address", "Listener address on this machine. Use 0.0.0.0 if Docker task containers need to connect from outside the host loopback interface.", 1),
    "training.gateway.advertised_url": ("Gateway advertised origin", "HTTP origin reachable from task containers. For local Docker use a host address reachable from containers or configured host networking; loopback works only when that network is shared.", 1),
    "training.deployment.layout": ("GPU layout", "colocate shares GPUs between training and serving; split assigns separate GPU budgets to each.", 2),
    "training.deployment.num_gpus": ("Total GPUs", "Number of GPUs reserved for this run. Check the model memory requirements before choosing a budget.", 2),
    "training.deployment.actor_gpus": ("Actor GPUs", "GPUs used for weight updates. In colocate mode this equals the total GPU budget.", 2),
    "training.deployment.rollout_gpus": ("Serving GPUs", "GPUs used for model inference during rollout. In colocate mode this equals the total GPU budget.", 2),
    "training.deployment.actor_tensor_parallel_size": ("Actor tensor parallelism", "Number of GPUs that jointly hold one training model replica; must divide the actor GPU budget.", 2),
    "training.deployment.rollout_gpus_per_engine": ("Serving tensor parallelism", "GPUs assigned to each serving engine; must divide the serving GPU budget.", 2),
    "training.training.num_rollout": ("Rollout batches", "Number of rollout batches to collect and train on before this run finishes.", 2),
    "training.training.learning_rate": ("Learning rate", "Positive optimizer learning rate; 1e-6 is a conservative starting value to review for your model.", 2),
    "training.training.save_interval": ("Save interval", "Save training checkpoints every this many rollout batches.", 2),
    "training.training.backend_contract": ("Algorithm contract", "Select the training adapter contract supported by your installed Slime revision.", 2),
    "training.sampling.group_size": ("Samples per group", "Generate at least two attempts for the same prompt so GRPO can compare their verifier rewards.", 2),
    "training.sampling.groups_per_batch": ("Groups per batch", "Number of complete prompt groups in one training batch. Samples per batch equals this times samples per group.", 2),
    "training.sampling.max_tokens": ("Maximum output tokens", "Maximum generated tokens per model response. Increase for tasks that require longer reasoning or tool use.", 2),
    "training.sampling.max_context": ("Context limit", "Maximum prompt plus response tokens supported by the selected model and serving runtime.", 2),
    "training.sampling.max_attempts": ("Maximum attempts", "Maximum task execution attempts allowed for each rollout slot.", 2),
}


def describe_field(path):
    label, help_text, step = FIELD_HELP.get(path, (path.split(".")[-1].replace("_", " "),
                                                       "Enter the value on the machine assigned to this role.", 0))
    if path.startswith("deployment.rollout.workers."):
        key = path.split(".")[-1]
        label = {"host": "Worker SSH destination", "source_dir": "Worker source directory",
                 "runner_python": "Worker runner Python", "0": "Worker task root",
                 "id": "Worker ID"}.get(key, label)
        help_text = {"host": "SSH alias reachable from the controller.",
                     "source_dir": "Absolute HarborRL checkout directory on this worker.",
                     "runner_python": "Absolute Python executable with Harbor installed on this worker.",
                     "0": "Absolute directory containing task files on this worker.",
                     "id": "Unique short worker name, using letters, numbers and hyphens."}.get(key, help_text)
    return {"path": path, "label": label, "help": help_text, "step": step}


def reference_examples(output_directory):
    """Site-independent, filled recipes. Paths/hosts are illustrative, never readiness claims."""
    output = Path(output_directory).resolve()
    result = []
    for preset, label in (("embedded-local", "Local best practice"),
                          ("service-sandbox-local-runner", "Sandbox best practice")):
        t, d = defaults(preset, output)
        t["tasks"]["catalog"] = str(output.parent / "hello-world" / "catalog.json")
        t["harness"]["profile"] = str(output.parent / "hello-world" / "claude-code-profile.json")
        t["model"].update(checkpoint="/srv/models/Qwen3-8B", reference="/srv/models/Qwen3-8B_torch_dist")
        # Four colocated GPUs is a topology example, not a model memory guarantee.
        t["deployment"].update(num_gpus=4, actor_gpus=4, rollout_gpus=4)
        d["gpu_target"]["expected_gpus"] = 4
        answers = {"answers_version": 1, "preset": preset, "training": t, "deployment": d}
        if preset == "embedded-local":
            answers["local_environment"] = {"source_dir": str(ROOT), "python": sys.executable}
            t["gateway"].update(host="0.0.0.0", advertised_url="http://host.docker.internal:49184")
            d["manager"].update(bind="0.0.0.0", gateway_url="http://host.docker.internal:49184")
            notes = ["The shared source directory and Python are detected on the machine running HarborRL; edit them once for all three roles.",
                     "Docker containers must resolve host.docker.internal to the host (on Linux configure host-gateway or replace it with a reachable host address)."]
        else:
            d["manager"].update(control_url="http://controller.example:49200", gateway_url="http://controller.example:49184")
            d["gpu_target"].update(ssh={"host": "gpu-training"}, source_dir="/srv/HarborRL", train_python="/srv/venvs/harborrl/bin/python")
            d["storage"]["run_root"] = t["output"]["root"] = "/srv/shared/harborrl-runs"
            t["gateway"]["advertised_url"] = d["manager"]["gateway_url"]
            t["harbor"]["python"] = sys.executable
            answers["sandbox_config"] = "/srv/site/sandbox.yaml"
            t["harbor"]["environment"]["config"] = answers["sandbox_config"]
            notes = ["Controller and Sandbox runner are on this host; the training GPU is reached over SSH. Use a Python 3.12+ Sandbox runner.",
                     "Replace the example hostnames and paths; mount the shared run root at the same absolute path on every participating host.",
                     "Provide your Sandbox endpoint, credential file and task image bindings in sandbox.yaml."]
        result.append({"id": preset, "title": label, "answers": answers,
                       "notes": notes + ["Replace model and input paths with existing files, then Prepare model to calculate fingerprints.",
                                         "Four GPUs illustrates colocate allocation; run Doctor and check model memory before training."],
                       "status": "reference"})
    return result


def load_draft(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Draft must be a regular file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("draft_version") != 1 or type(document.get("revision")) is not int
            or document["revision"] < 1):
        raise ValueError("Unsupported draft document")
    check_answers(document["answers"], path.parent)
    return document


def save_draft(path, answers, *, revision=None):
    path = Path(path).expanduser().absolute()
    check_answers(answers, path.parent)
    if path.is_symlink():
        raise ValueError("Draft must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name("." + path.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(descriptor)
        current = load_draft(path)["revision"] if path.exists() else 0
        if revision != current and not (revision is None and current == 0):
            raise ValueError("Draft changed since it was opened. Reopen it before saving, or use a new name.")
        document = {"draft_version": 1, "revision": current + 1, "answers": deepcopy(answers)}
        descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return document
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()
