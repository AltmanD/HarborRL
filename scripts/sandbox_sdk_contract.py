"""Check the actual optional SDK/Harbor imports without allocating resources."""
import asyncio
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from harbor.models.trial.config import TrialConfig
from opensandbox import Sandbox

from harborrl.harnesses.offline_claude import OfflineClaudeCode
from harborrl.rollout.harbor_job.environments.opensandbox import (
    ManagedOpenSandboxEnvironment,
)
from harborrl.trajectories.native import read_json

sdk = ManagedOpenSandboxEnvironment._load_opensandbox(None)
assert {'extensions', 'skip_health_check', 'ready_timeout'} <= inspect.signature(Sandbox.create).parameters.keys()
assert 'import_path' in TrialConfig.model_fields['environment'].annotation.model_fields
assert not inspect.isabstract(OfflineClaudeCode)


async def cleanup_contract():
    with tempfile.TemporaryDirectory() as directory:
        env = object.__new__(ManagedOpenSandboxEnvironment)
        env._startup_transport = None
        env.cfg = {'cleanup_timeout_s': 1}
        env.evidence = Path(directory)
        env.binding = {'image': 'image-test'}
        env.record = {'sandbox_id': 'sandbox-test', 'runtime_digest': 'test', 'events': []}
        calls = []

        async def kill():
            calls.append('kill')

        async def close():
            calls.append('close')

        env._sandbox = SimpleNamespace(kill=kill, close=close)
        await env.stop()
        assert calls == ['kill', 'close']
        assert read_json(env.evidence / 'sandbox-lifecycle.json')['state'] == 'deleted'

        async def failed_kill():
            raise RuntimeError('secret-must-not-appear')

        env._sandbox = SimpleNamespace(kill=failed_kill, close=close)
        try:
            await env.stop()
        except RuntimeError as exc:
            assert str(exc) == 'Sandbox deletion failed'
        else:
            raise AssertionError('deletion failure was swallowed')
        assert 'secret-must-not-appear' not in (env.evidence / 'cleanup-required.json').read_text()
        assert read_json(env.evidence / 'sandbox-lifecycle.json')['state'] == 'delete_failed'


asyncio.run(cleanup_contract())
print(json.dumps({'sdk_symbols': sorted(sdk), 'agent': OfflineClaudeCode.__name__, 'status': 'passed'}))
