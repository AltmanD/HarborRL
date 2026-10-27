"""Observe SDK allocation IDs without retrying creation requests."""
import re

import httpx


class SandboxStartupTransport(httpx.AsyncBaseTransport):
    def __init__(self, base_url, on_created, *, transport=None):
        self.base_url = base_url.rstrip('/')
        self.on_created = on_created
        self.transport = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        url = str(request.url).split('?', 1)[0]
        response = await self.transport.handle_async_request(request)
        if request.method == 'POST' and url == self.base_url + '/sandboxes' and response.is_success:
            await response.aread()
            try:
                sandbox_id = response.json()['id']
                if isinstance(sandbox_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', sandbox_id):
                    self.on_created(sandbox_id)
            except (ValueError, KeyError, TypeError):
                pass  # Let the SDK reject malformed creation responses.
        return response

    async def aclose(self):
        await self.transport.aclose()
