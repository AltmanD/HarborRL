"""Explicit tokenizer preparation, separate from offline preview/validation."""
import json
import os
import re
import shlex
import subprocess

from harborrl.deployment.remote import ssh_command
from harborrl.deployment.schema import _ssh_spec

from .presets import defaults, merge

SCRIPT = '''import json, sys
from transformers import AutoTokenizer
from harborrl.rollout.runtime import tokenizer_fingerprint
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
a, b = tokenizer_fingerprint(tokenizer)
print('HARBORRL_FINGERPRINT=' + json.dumps({'tokenizer_digest': a, 'template_digest': b}))
'''


def prepare_model(answers, output_directory, *, timeout=90):
    from .presets import check_answers
    check_answers(answers, output_directory)
    t, d = defaults(answers["preset"], output_directory)
    t = merge(t, answers.get("training", {}))
    d = merge(d, answers.get("deployment", {}))
    local = answers.get("local_environment", {})
    target = d["gpu_target"]
    source = local.get("source_dir", target["source_dir"])
    python = local.get("python", target["train_python"])
    checkpoint = t["model"]["checkpoint"]
    if not all(isinstance(v, str) and v.strip() for v in (source, python, checkpoint)):
        raise ValueError("Select the checkpoint directory, GPU source directory and Python interpreter before Prepare model.")
    command = [python, "-c", SCRIPT, checkpoint]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if target["transport"] == "ssh":
        spec = _ssh_spec(target["ssh"], "GPU SSH destination")
        remote = (f"cd {shlex.quote(source)} && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                  f"PYTHONPATH={shlex.quote(source)} " + shlex.join(command))
        command = ssh_command(spec, remote)
        cwd = None
    else:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
        cwd = source
    try:
        result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Model preparation could not finish: {exc}. Check the selected Python and SSH access; your draft can still be saved.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout)[-2500:].strip()
        raise ValueError("Could not load the local tokenizer. Use the GPU training Python with Transformers installed, "
                         "check the checkpoint files and SSH access. Custom tokenizer code is not automatically executed. "
                         f"Your draft can still be saved.\n{detail}")
    try:
        line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("HARBORRL_FINGERPRINT="))
        fingerprints = json.loads(line.split("=", 1)[1])
        if (set(fingerprints) != {"tokenizer_digest", "template_digest"}
                or any(not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64 for value in fingerprints.values())):
            raise ValueError("invalid fingerprints")
    except (StopIteration, ValueError, TypeError) as exc:
        raise ValueError("Model preparation returned no valid tokenizer fingerprints") from exc
    return {"fingerprints": fingerprints, "checkpoint": checkpoint, "python": python,
            "target": target["ssh"]["host"] if target["transport"] == "ssh" else "local"}
