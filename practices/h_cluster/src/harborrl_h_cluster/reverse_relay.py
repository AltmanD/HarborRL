"""Manager-initiated sandbox relay over the authenticated platform proxy.

Experimental transport: HTTP/SSE bodies are buffered and no upstream POST is
retried. Credentials and request bodies must never enter diagnostic logs.
"""
import asyncio
import base64

import httpx

from .relay_server import MAX_BODY, PATHS, REQUEST_HEADERS, RESPONSE_HEADERS


async def pump(proxy_url, proxy_headers, relay_token, gateway_url, stop, *, timeout=110):
    headers = {**proxy_headers, 'X-Relay-Token': relay_token}
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as upstream, \
            httpx.AsyncClient(timeout=30, trust_env=False, headers=headers) as platform:
        while not stop.is_set():
            pending = await platform.get(proxy_url.rstrip('/') + '/relay/next')
            if pending.status_code == 204:
                await asyncio.sleep(0.25)
                continue
            pending.raise_for_status()
            item = pending.json()
            if (item['method'] != 'POST' or item['path'].split('?', 1)[0] not in PATHS
                    or not isinstance(item['id'], str)
                    or len(item['id']) != 32 or any(c not in '0123456789abcdef' for c in item['id'])):
                raise ValueError('invalid relay request')
            body = base64.b64decode(item['body'], validate=True)
            if len(body) > MAX_BODY:
                raise ValueError('relay body too large')
            request_headers = {k: v for k, v in item['headers'].items() if k.lower() in REQUEST_HEADERS}
            try:
                async with upstream.stream('POST', gateway_url.rstrip('/') + item['path'],
                                           content=body, headers=request_headers) as response:
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_BODY // 2:
                            raise ValueError('relay response too large')
                    result = {'status': response.status_code,
                              'headers': {k: v for k, v in response.headers.items()
                                          if k.lower() in RESPONSE_HEADERS},
                              'body': base64.b64encode(content).decode()}
            except (httpx.HTTPError, ValueError):
                result = {'status': 502, 'headers': {'content-type': 'application/json'},
                          'body': base64.b64encode(b'{"error":"upstream_failed_no_retry"}').decode()}
            delivered = await platform.post(proxy_url.rstrip('/') + '/relay/result/' + item['id'], json=result)
            delivered.raise_for_status()
