import asyncio

import httpx

from harborrl.rollout.harbor_job.sandbox_transport import SandboxStartupTransport


def test_generic_transport_observes_allocation_without_retry():
    async def run():
        calls, allocations = [], []

        def handler(request):
            calls.append(request.method)
            return httpx.Response(200, json={'id': 'sbx-one'}) if request.method == 'POST' else httpx.Response(503)

        transport = SandboxStartupTransport('https://sandbox.example/v1', allocations.append,
                                           transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post('https://sandbox.example/v1/sandboxes')
            result = await client.get('https://sandbox.example/v1/sandboxes/sbx-one/endpoints/44772')
            assert result.status_code == 503
        assert calls == ['POST', 'GET'] and allocations == ['sbx-one']

    asyncio.run(run())
