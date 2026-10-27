import asyncio

import httpx
from harborrl_h_cluster.transport import SandboxStartupTransport


def test_transient_endpoint_retries_do_not_recreate_sandbox():
    async def run():
        calls, ids = [], []

        async def serve(request):
            calls.append((request.method, request.url.path))
            if request.method == 'POST':
                return httpx.Response(201, json={'id': 'sbx-one'})
            if len(calls) < 4:
                return httpx.Response(503)
            return httpx.Response(200, json={'endpoint': 'ready'})

        transport = SandboxStartupTransport('https://platform.test/v1', ids.append, timeout=10,
            transport=httpx.MockTransport(serve), retry_delay=0)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post('https://platform.test/v1/sandboxes', json={})
            response = await client.get('https://platform.test/v1/sandboxes/sbx-one/endpoints/44772')
            assert response.status_code == 200
        assert ids == ['sbx-one']
        assert sum(method == 'POST' for method, _ in calls) == 1
        assert len(calls) == 4

    asyncio.run(run())


def test_mutations_proxy_requests_and_auth_failures_are_not_retried():
    async def run():
        calls = []

        async def serve(request):
            calls.append(request)
            return httpx.Response(401 if '/endpoints/' in request.url.path else 503)

        transport = SandboxStartupTransport('https://platform.test/v1', lambda _: None, timeout=10,
            transport=httpx.MockTransport(serve), retry_delay=0)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post('https://platform.test/v1/sandboxes')
            await client.get('https://platform.test/v1/sandboxes/sbx-one/proxy/44772')
            await client.get('https://platform.test/v1/sandboxes/sbx-one/endpoints/44772')
        assert len(calls) == 3

    asyncio.run(run())


def test_endpoint_retry_deadline_returns_final_failure():
    async def run():
        calls = []

        async def serve(request):
            calls.append(request)
            return httpx.Response(503)

        transport = SandboxStartupTransport('https://platform.test/v1', lambda _: None, timeout=0,
            transport=httpx.MockTransport(serve), retry_delay=0)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get('https://platform.test/v1/sandboxes/sbx-one/endpoints/44772')
        assert response.status_code == 503
        assert len(calls) == 1

    asyncio.run(run())
