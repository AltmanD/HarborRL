"""Non-interactive SSH transport for deployment preflight and launch.

Every invocation runs BatchMode: a password prompt is a failed check, never an
interactive hang. User, port and identity are inherited from the invoking
user's SSH config unless the deployment spec narrows them.
"""
from __future__ import annotations

import shlex
import subprocess

SSH_OPTIONS = ("BatchMode=yes", "ClearAllForwardings=yes", "StrictHostKeyChecking=accept-new")


def ssh_command(spec, remote_command, *, connect_timeout=15):
    """Assemble one non-interactive ssh argv for a normalized spec."""
    argv = ["ssh"]
    for option in SSH_OPTIONS:
        argv += ["-o", option]
    argv += ["-o", f"ConnectTimeout={int(connect_timeout)}"]
    if spec.get("port"):
        argv += ["-p", str(spec["port"])]
    if spec.get("identity"):
        argv += ["-i", str(spec["identity"])]
    from harborrl.deployment.schema import destination

    argv += [destination(spec), remote_command]
    return argv


def run_ssh(spec, command, *, timeout=30, connect_timeout=15):
    """Run one remote command; transport failures surface as returncode 255."""
    argv = ssh_command(spec, command, connect_timeout=connect_timeout)
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(argv, 255, "", f"{type(exc).__name__}: {exc}")


def test_remote(spec, expression, *, timeout=30, connect_timeout=15):
    """Evaluate a shell test expression on the target; True when it passes."""
    result = run_ssh(spec, f"test {expression} && echo ok", timeout=timeout,
                     connect_timeout=connect_timeout)
    return result.returncode == 0 and result.stdout.strip() == "ok", result


def remote_read(spec, command, *, timeout=30, connect_timeout=15):
    """Return (returncode, stdout, stderr) for a remote command."""
    result = run_ssh(spec, command, timeout=timeout, connect_timeout=connect_timeout)
    return result.returncode, result.stdout, result.stderr


def quote(value):
    return shlex.quote(str(value))
