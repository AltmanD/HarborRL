import asyncio

import httpx
import pytest

pytest.importorskip('harbor.environments.opensandbox')

from harborrl.rollout.harbor_job.environments.opensandbox import (
    ManagedOpenSandboxEnvironment,
)


@pytest.mark.parametrize('delete_status,expected', [(204, 'deleted'), (500, 'delete_failed')])
def test_allocated_id_cleanup_requires_confirmed_absence(tmp_path, monkeypatch, delete_status, expected):
    calls = []

    async def handler(request):
        calls.append((request.method, request.url.path))
        assert request.url.path == '/v1/sandboxes/sbx-owned'
        if request.method == 'DELETE':
            return httpx.Response(delete_status)
        return httpx.Response(200 if len(calls) == 1 else 404)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_class(
        **kwargs, transport=httpx.MockTransport(handler)))
    env = object.__new__(ManagedOpenSandboxEnvironment)
    env.cfg = {'domain': 'https://platform.test', 'cleanup_timeout_s': 1}
    env.binding = {'image': 'owned'}
    env._api_key = 'test-only'
    env.evidence = tmp_path
    env.record = {'sandbox_id': 'sbx-owned', 'runtime_digest': 'test', 'events': []}
    asyncio.run(env._cleanup_allocated())
    assert env.record['state'] == expected
    assert (tmp_path / 'cleanup-required.json').exists() == (expected == 'delete_failed')
    assert calls[:2] == [('GET', '/v1/sandboxes/sbx-owned'), ('DELETE', '/v1/sandboxes/sbx-owned')]
