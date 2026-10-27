"""Explicit, bounded local hints. Never connect to a remote machine."""
import os
import shlex
import subprocess
import sys
from pathlib import Path


def command_hint(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        return {"method": command[0], "status": "observed", "value": result.stdout.strip(), "reason": None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"method": command[0], "status": "unknown", "value": None, "reason": str(exc)}


def ssh_aliases(path):
    aliases = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0].lower() == "host":
            aliases.update(word for word in words[1:] if not any(c in word for c in "*?!"))
    return sorted(aliases)


def probe_local(output_directory):
    try:
        aliases = {"method": "ssh_config_hosts", "status": "observed",
                       "value": ssh_aliases(Path.home() / ".ssh/config"), "reason": "Include and Match are not evaluated"}
    except (OSError, ValueError) as exc:
        aliases = {"method": "ssh_config_hosts", "status": "unknown", "value": None, "reason": str(exc)}
    parent = Path(output_directory).expanduser().absolute().parent
    return {
        "gpu": command_hint(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]),
        "python": command_hint([sys.executable, "-I", "--version"]),
        "ssh_aliases": aliases,
        "output_parent": {"method": "os.access", "status": "hint", "value": os.access(parent, os.W_OK),
                              "reason": "Access hint only; no write was attempted"},
    }
