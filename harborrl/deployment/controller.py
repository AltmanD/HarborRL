"""Service-mode dispatch: the manager-side controller flow (P1 + P3).

An independent CPU manager runs the same preflight graph, materializes the
run on shared storage, starts the RolloutService control plane, launches the
existing Slime job on the GPU target over SSH, and waits for the shared
exit-code record. Rollout execution itself moves to the manager only through
the service; training stays on the GPU.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
from pathlib import Path

from harborrl.config import ROOT
from harborrl.deployment import remote
from harborrl.deployment.schema import destination
from harborrl.deployment.training_target import SshTrainingTarget
from harborrl.platform.evidence import (
    _materialize,
    _source_audit,
    effective_config,
    new_run_root,
    run_environment,
)
from harborrl.rollout.service import RolloutService, make_control_server
from harborrl.trajectories.native import publish

TOKEN_MODE = 0o600


def check_gpu_control(deployment):
    manager, gpu = deployment['manager'], deployment['gpu_target']
    timeout = deployment['connectivity']['http_timeout_s']
    result = remote.run_ssh(gpu['ssh'] or {'host': 'localhost'},
        f"curl --noproxy '*' -fsS --max-time {timeout} " + remote.quote(manager['control_url'] + '/healthz'),
        timeout=timeout + 15)
    if result.returncode != 0:
        raise RuntimeError('GPU cannot reach manager control service')
    payload = json.loads(result.stdout)
    if payload != {'ok': True, 'service': 'harborrl-rollout-service-v1'}:
        raise RuntimeError('GPU reached an unexpected control service')
    return {'gpu_to_controller': True, 'control_url': manager['control_url']}


def _ensure_token(token_file):
    """Load the run token, creating a fresh one on first use."""
    path = Path(token_file)
    if path.is_file():
        token = path.read_text().strip()
        if token:
            return token
    import secrets

    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, TOKEN_MODE)
    with os.fdopen(fd, "w") as stream:
        stream.write(token + "\n")
    return token


def _rollout_environment(plan, deployment, environment):
    """Environment overlay that points the GPU job at the manager service."""
    manager = deployment["manager"]
    workers = deployment["rollout"]["workers"]
    overlay = dict(environment)
    overlay.update({
        "HARBORRL_ROLLOUT_MODE": "service",
        "HARBORRL_ROLLOUT_SERVICE_URL": manager["control_url"],
        "HARBORRL_ROLLOUT_SERVICE_TOKEN_FILE": manager['run_token_file'],
        "TRAIN_PYTHON": deployment['gpu_target']['train_python'],
        "HARBORRL_NATIVE_GATEWAY_HOST": manager["bind"],
        "HARBORRL_NATIVE_GATEWAY_PORT": str(manager["gateway_port"]),
        "HARBORRL_NATIVE_GATEWAY_URL": manager["gateway_url"],
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(
            [destination(worker["ssh"]) for worker in workers]),
        "HARBORRL_NATIVE_RUNNER_PYTHON": (plan['config']['harbor']['python']
                                         if deployment['rollout']['provider'] == 'sandbox'
                                         else workers[0]['runner_python']),
    })
    if deployment['rollout']['provider'] == 'sandbox':
        if 'HARBORRL_NATIVE_SANDBOX' not in overlay:
            raise ValueError('sandbox deployment requires sandbox training config')
        transport = plan['config']['harbor'].get('runner_transport', 'ssh')
        if (transport == 'local' and workers) or (transport == 'ssh' and not workers):
            raise ValueError('sandbox runner transport and deployment workers disagree')
        overlay['HARBORRL_NATIVE_RUNNER_TRANSPORT'] = transport
    return overlay


def _runtime_factory(deployment, environment, *, concurrency_limit, group_size, tokenizer=None):
    """Build the per-run rollout runtime when the GPU job announces serving."""

    def factory(run_id, router_url):
        from harborrl.rollout.runtime import RolloutRuntime, RolloutRuntimeConfig

        env = {**os.environ, **environment}
        config = RolloutRuntimeConfig.from_environment(
            env, endpoint=router_url, concurrency_limit=concurrency_limit,
            group_size=group_size,
            gateway_host=deployment["manager"]["bind"],
            gateway_port=deployment["manager"]["gateway_port"],
            gateway_url=deployment["manager"]["gateway_url"])
        resolved = tokenizer
        if resolved is None:
            from transformers import AutoTokenizer

            resolved = AutoTokenizer.from_pretrained(
                env["HF_CKPT"], trust_remote_code=True)
        return RolloutRuntime(config, resolved)

    return factory


def dispatch_service(plan, deployment, *, checks):
    manager, gpu, storage = deployment["manager"], deployment["gpu_target"], deployment["storage"]
    config = plan["config"]
    root = new_run_root(storage["run_root"])
    root.mkdir(parents=True, mode=0o700)
    run_id = root.name

    catalog_path, prompt_path = _materialize(plan, root)
    source = _source_audit(root)
    environment = run_environment(plan, root, catalog_path, prompt_path)
    token = _ensure_token(manager["run_token_file"])
    environment = _rollout_environment(plan, deployment, environment)

    sampling = config["sampling"]
    concurrency = sampling["group_size"] * sampling["groups_per_batch"]
    service = RolloutService(
        run_root=root,
        runtime_factory=_runtime_factory(
            deployment, environment,
            concurrency_limit=concurrency, group_size=sampling["group_size"]),
        token_file=manager["run_token_file"])
    control = make_control_server(service, host=manager["bind"],
                                  port=manager["control_port"])
    control_thread = None
    target = None
    exit_code = 1
    try:
        service.register_run(run_id, {
            "run_id": run_id, "deployment_source": plan["deployment"]["source"],
            "source_commit": source["commit"],
            "config_digest": plan["deployment"]["digest"],
            "group_size": sampling["group_size"],
            "workers": [worker["id"] for worker in deployment["rollout"]["workers"]],
        }, token=token)
        control_thread = threading.Thread(
            target=control.serve_forever, name="harborrl-control-plane", daemon=True)
        control_thread.start()
        publish(root / 'control-readiness.json', check_gpu_control(deployment))

        publish(root / "launch.json", {
            "plan": plan, "run_dir": str(root), "source": source,
            "materialized_catalog": str(catalog_path), "prompt_data": str(prompt_path),
            "control_url": manager["control_url"]})
        publish(root / "effective-config.json",
                effective_config(plan, environment, dict(environment), source, checks))

        target = SshTrainingTarget(
            gpu["ssh"] or {"host": "localhost"}, source_dir=gpu["source_dir"], run_dir=root,
            environment_file=gpu.get("environment_file"),
            command_timeout=deployment["connectivity"]["command_timeout_s"] + 30,
            ssh_timeout=deployment["connectivity"]["ssh_timeout_s"])
        # The launcher executes inside the GPU checkout; manager-side absolute
        # paths are rewritten so shared-storage artifacts keep one spelling.
        target.write_launcher(
            ["bash", str(ROOT / "harborrl/platform/slime_train.sh")], environment,
            path_rewrites=[(str(manager["source_dir"]), str(gpu["source_dir"]))])
        target.launch()

        def forward_signal(signum, frame):
            if target is not None:
                target.cancel()

        previous = {sig: signal.signal(sig, forward_signal)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            exit_code = target.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except BaseException:
        import traceback

        traceback.print_exc()
        if target is not None:
            # Best-effort teardown; the run record keeps the failure evidence.
            with contextlib.suppress(Exception):
                target.cancel()
        raise
    finally:
        with contextlib.suppress(Exception):
            service.drain_run(run_id)
        service.close()
        if control_thread is not None:
            control.shutdown()
            control_thread.join(5)
        control.server_close()
        print(json.dumps({"run_dir": str(root), "exit_code": exit_code}), file=sys.stderr)
    return exit_code
