"""Extracted rollout core (P2): gateway, traces and complete groups.

This module owns rollout execution and must not import Slime or Ray: training
frameworks live in ``harborrl.backends.*`` adapters that build a
``RolloutRuntimeConfig`` and hand over a tokenizer. Group semantics, worker
scheduling, retries and the RL_READY contract stay exactly as the embedded
pipeline enforced them.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from harborrl.models.qwen import normalize_chat_template_ids
from harborrl.rollout.harbor_job.coordinator import GroupSlots, run_attempt
from harborrl.trajectories.native import Identity, RewardProfile, digest, read_json

TASK_ENTRY_KEYS = {"id", "revision", "path", "task_digest", "reward_profile"}


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    """Deployment-independent description of one run's rollout machinery."""

    run_id: str
    run_root: str
    catalog_path: str
    profile_path: str
    gateway_host: str
    gateway_port: int
    gateway_url: str
    runner_python: str
    workers: tuple
    task_worker_overrides: dict
    max_attempts: int
    max_output_tokens: int
    max_context: int
    concurrency_limit: int
    group_size: int
    expected_tokenizer_digest: str
    expected_template_digest: str
    audited_raw_logprobs: bool
    drain_timeout: float
    model_tag: str
    endpoint: str
    sandbox: dict = None
    runner_transport: str = 'ssh'

    @staticmethod
    def _required(env, name):
        value = env.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rollout runtime requires {name}")
        return value

    @classmethod
    def from_environment(cls, env, *, endpoint, concurrency_limit, group_size,
                         gateway_host=None, gateway_port=None, gateway_url=None,
                         workers=None, runner_python=None):
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rollout runtime requires a serving endpoint")
        if type(group_size) is not int or group_size < 2:
            raise ValueError("rollout runtime requires a group size of at least two")
        if workers is None:
            workers = json.loads(cls._required(env, "HARBORRL_NATIVE_RUNNER_WORKERS"))
        if runner_python is None:
            runner_python = cls._required(env, "HARBORRL_NATIVE_RUNNER_PYTHON")
        try:
            overrides = json.loads(env.get("HARBORRL_NATIVE_TASK_WORKER_OVERRIDES") or "{}")
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_TASK_WORKER_OVERRIDES") from exc
        return cls(
            run_id=cls._required(env, "RUN_ID"),
            run_root=cls._required(env, "RUN_DIR"),
            catalog_path=cls._required(env, "HARBORRL_NATIVE_CATALOG"),
            profile_path=cls._required(env, "HARBORRL_NATIVE_PROFILE"),
            gateway_host=gateway_host or env.get("HARBORRL_NATIVE_GATEWAY_HOST", "0.0.0.0"),
            gateway_port=int(gateway_port or cls._required(env, "HARBORRL_NATIVE_GATEWAY_PORT")),
            gateway_url=gateway_url or cls._required(env, "HARBORRL_NATIVE_GATEWAY_URL"),
            runner_python=runner_python,
            workers=tuple(workers),
            task_worker_overrides=dict(overrides),
            max_attempts=int(env.get("HARBORRL_NATIVE_MAX_ATTEMPTS", "2")),
            max_output_tokens=int(env.get("ROLLOUT_MAX_RESPONSE_LEN", "8192")),
            max_context=int(env.get("ROLLOUT_MAX_CONTEXT_LEN", "32768")),
            concurrency_limit=max(1, int(concurrency_limit)),
            group_size=int(group_size),
            expected_tokenizer_digest=cls._required(env, "HARBORRL_NATIVE_TOKENIZER_DIGEST"),
            expected_template_digest=cls._required(env, "HARBORRL_NATIVE_TEMPLATE_DIGEST"),
            audited_raw_logprobs=env.get("HARBORRL_NATIVE_AUDITED_LOGPROBS") == "1",
            drain_timeout=float(env.get("HARBORRL_NATIVE_DRAIN_TIMEOUT", "30")),
            model_tag=env.get("MODEL_TAG", env.get("MODEL_ARGS_FILE", "qwen3-8B")),
            endpoint=endpoint.rstrip("/"),
            sandbox=json.loads(env.get('HARBORRL_NATIVE_SANDBOX', 'null')),
            runner_transport=env.get('HARBORRL_NATIVE_RUNNER_TRANSPORT', 'ssh'),
        )

    def environment(self):
        """Render the environment mapping this config was built from."""
        return {
            "RUN_ID": self.run_id, "RUN_DIR": self.run_root,
            "HARBORRL_NATIVE_CATALOG": self.catalog_path,
            "HARBORRL_NATIVE_PROFILE": self.profile_path,
            "HARBORRL_NATIVE_GATEWAY_HOST": self.gateway_host,
            "HARBORRL_NATIVE_GATEWAY_PORT": str(self.gateway_port),
            "HARBORRL_NATIVE_GATEWAY_URL": self.gateway_url,
            "HARBORRL_NATIVE_RUNNER_PYTHON": self.runner_python,
            "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(list(self.workers)),
            "HARBORRL_NATIVE_TASK_WORKER_OVERRIDES": json.dumps(
                self.task_worker_overrides, sort_keys=True),
            "HARBORRL_NATIVE_MAX_ATTEMPTS": str(self.max_attempts),
            "ROLLOUT_MAX_RESPONSE_LEN": str(self.max_output_tokens),
            "ROLLOUT_MAX_CONTEXT_LEN": str(self.max_context),
            "HARBORRL_NATIVE_TOKENIZER_DIGEST": self.expected_tokenizer_digest,
            "HARBORRL_NATIVE_TEMPLATE_DIGEST": self.expected_template_digest,
            "HARBORRL_NATIVE_AUDITED_LOGPROBS": str(int(self.audited_raw_logprobs)),
            "HARBORRL_NATIVE_DRAIN_TIMEOUT": str(self.drain_timeout),
            "MODEL_TAG": self.model_tag,
            "HARBORRL_NATIVE_SANDBOX": json.dumps(self.sandbox),
            "HARBORRL_NATIVE_RUNNER_TRANSPORT": self.runner_transport,
        }


def tokenizer_fingerprint(tokenizer):
    messages = [{"role": "user", "content": "HarborRL native tokenizer/template fingerprint"}]
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    rendered_ids = normalize_chat_template_ids(
        tokenizer.apply_chat_template(messages, **kwargs))
    vocabulary = tokenizer.get_vocab()
    tokenizer_digest = digest({
        "class": type(tokenizer).__name__,
        "vocab": sorted((str(key), int(value)) for key, value in vocabulary.items()),
    })
    template = digest({"messages": messages, "kwargs": kwargs, "token_ids": rendered_ids})
    return tokenizer_digest, template


class RolloutRuntime:
    """One gateway, one trace registry and one rollout root per run."""

    def __init__(self, config, tokenizer, *, backend=None):
        from harborrl.gateway.server import Gateway, make_server
        from harborrl.gateway.sglang_backend import SGLangBackend
        from harborrl.gateway.trace import TraceRegistry
        from harborrl.math_registry import MATH_REGISTRY

        self.config = config
        self.run_root = Path(config.run_root).resolve()
        self.catalog = read_json(config.catalog_path)
        self.profile = read_json(config.profile_path)
        self.rollout_root = self.run_root / "native-rollouts"
        self.rollout_root.mkdir(parents=True, exist_ok=True)
        self.registry = TraceRegistry(self.run_root / "native-traces")
        self.semaphore = asyncio.Semaphore(config.concurrency_limit)
        if config.sandbox:
            from harborrl.rollout.harbor_job.sandbox_config import SandboxAdmission
            self.sandbox_admission = SandboxAdmission(config.sandbox)
        tokenizer_digest, template_digest = tokenizer_fingerprint(tokenizer)
        if (tokenizer_digest != config.expected_tokenizer_digest
                or template_digest != config.expected_template_digest):
            raise ValueError("live tokenizer/template digest differs from the native lock")
        model = MATH_REGISTRY.model_for_preset(config.model_tag)
        self.backend = backend or SGLangBackend(
            config.endpoint, tokenizer, model.adapter,
            tokenizer_digest=tokenizer_digest, template_digest=template_digest,
            max_context=config.max_context,
            audited_raw_logprobs=config.audited_raw_logprobs,
        )
        gateway = Gateway(self.registry, self.backend, model=self.profile["model"],
                          max_output_tokens=config.max_output_tokens)
        self.server = make_server(gateway, host=config.gateway_host,
                                  port=config.gateway_port)
        self._start_gateway()

    def _start_gateway(self):
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="harborrl-native-gateway", daemon=True
        )
        self.thread.start()

    def close(self):
        self.server.shutdown()
        if getattr(self.thread, "is_alive", lambda: False)():
            self.thread.join(5)
        self.server.server_close()

    def manifest_path(self, identity):
        """Deterministic shared-storage path of one trajectory manifest."""
        if isinstance(identity, dict):
            identity = Identity(**identity)
        return self.rollout_root / identity.group_id / identity.attempt_id / "trajectory.v2.json"


# Historical name kept for existing hooks and evidence readers.
NativeRolloutRuntime = RolloutRuntime

_RUNTIME = None
_RUNTIME_LOCK = asyncio.Lock()


async def runtime(config, *, tokenizer=None, tokenizer_loader=None):
    """Process-wide runtime cache for the embedded GPU-side rollout path."""
    global _RUNTIME
    if _RUNTIME is None:
        async with _RUNTIME_LOCK:
            if _RUNTIME is None:
                if tokenizer is None:
                    if tokenizer_loader is None:
                        raise ValueError("runtime requires a tokenizer or a loader")
                    tokenizer = tokenizer_loader()
                _RUNTIME = RolloutRuntime(config, tokenizer)
    return _RUNTIME


def publish_native_version(runtime_, states, expected):
    """Open the next admission epoch only after the complete pool is healthy."""
    runtime_.registry.drain(runtime_.config.drain_timeout)
    runtime_.registry.publish_version(states, str(expected))


def current_policy_version(runtime_):
    """Publish and lock the version currently served by the rollout pool."""
    states = runtime_.backend.pool_states()
    from harborrl.platform.policy_pool import publish_pool
    version = publish_pool(states)
    runtime_.registry.drain(runtime_.config.drain_timeout)
    runtime_.registry.publish_version(states, version)
    return version


def close_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.close()
        _RUNTIME = None


def task_entry(sample):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    entry = metadata.get("native_task")
    if not isinstance(entry, dict) or set(entry) != TASK_ENTRY_KEYS:
        raise ValueError("native rollout requires a locked task catalog entry")
    return entry


def slot_identity(config, group_index, slot, rollout_id, entry,
                  reward_digest, sampling_digest, policy_version, *, harness_digest=None):
    return Identity(
        run_id=config.run_id,
        batch_id=f"rollout-{rollout_id}",
        group_id=f"group-{group_index}",
        slot_id=str(slot),
        attempt_id=secrets.token_hex(16),
        trajectory_id=secrets.token_hex(16),
        task_digest=entry["task_digest"],
        policy_version=str(policy_version),
        harness_digest=harness_digest if harness_digest is not None else digest(read_json(config.profile_path)),
        reward_digest=reward_digest,
        sampling_digest=sampling_digest,
    )


def _group_slots(config, runtime_, group_index, rollout_id, entry,
                 sampling_digest, policy_version, size):
    reward_profile = RewardProfile(**entry["reward_profile"])
    harness_digest = digest(runtime_.profile)
    identities = [
        slot_identity(config, group_index, index, rollout_id, entry,
                      reward_profile.digest, sampling_digest, policy_version,
                      harness_digest=harness_digest)
        for index in range(size)
    ]
    return GroupSlots(identities, max_attempts=config.max_attempts), reward_profile


async def run_slot(config, runtime_, group_index, slot, rollout_id, entry,
                   sampling_digest, policy_version):
    """Compatibility entry point for executing an individual group slot."""
    slots, reward_profile = _group_slots(
        config, runtime_, group_index, rollout_id, entry,
        sampling_digest, policy_version, config.group_size)
    return await _run_slot(config, runtime_, slots, slot, entry, reward_profile)


async def _run_slot(config, runtime_, slots, slot, entry, reward_profile):
    """Execute one slot with the group's shared identity and retry bookkeeping."""
    workers = config.workers
    overrides = config.task_worker_overrides
    for attempt_index in range(slots.max_attempts):
        identity = slots.start(str(slot))
        runner_host = overrides.get(entry["id"])
        if config.runner_transport == 'local':
            runner_host = None
        elif runner_host is None:
            if not workers:
                raise ValueError("rollout runtime requires at least one worker destination")
            runner_host = workers[(slot + attempt_index) % len(workers)]
        root = runtime_.rollout_root / identity.group_id / identity.attempt_id
        admission = nullcontext()
        if config.sandbox:
            from harborrl.rollout.harbor_job.sandbox_config import task_binding
            admission = runtime_.sandbox_admission.lease(task_binding(config.sandbox, entry['path']), root)
        try:
            async with admission, runtime_.semaphore:
                ir = await run_attempt(
                    identity,
                    task_path=entry["path"],
                    profile=runtime_.profile,
                    gateway_url=config.gateway_url,
                    runner_python=config.runner_python,
                    runner_host=runner_host,
                    root=root,
                    registry=runtime_.registry,
                    reward_profile=reward_profile,
                    max_output_tokens=config.max_output_tokens,
                    **({'sandbox': config.sandbox} if config.sandbox else {}),
                )
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
            slots.finish(str(slot), error=f"{type(exc).__name__}: {exc}")
            if config.sandbox and ((root / 'cleanup-required.json').exists() or runtime_.sandbox_admission.blocked):
                raise RuntimeError('sandbox cleanup unresolved; automatic retry blocked') from exc
            continue
        slots.finish(str(slot), ir)
        return ir
    raise RuntimeError(f"native slot {slot} exhausted its retry budget")


async def run_group(config, runtime_, *, entry, group_index, rollout_id,
                    sampling_digest, policy_version, group_size=None):
    """Execute one complete group and return per-slot IRs in slot order."""
    size = config.group_size if group_size is None else int(group_size)
    slots, reward_profile = _group_slots(
        config, runtime_, group_index, rollout_id, entry,
        sampling_digest, policy_version, size)
    tasks = [asyncio.create_task(
        _run_slot(config, runtime_, slots, slot, entry, reward_profile))
        for slot in range(size)
    ]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
