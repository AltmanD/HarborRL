"""Lifecycle of one manager-initiated relay inside a real sandbox."""
import asyncio
import secrets
import tempfile
from pathlib import Path

import httpx

from harborrl.trajectories.native import publish
from harborrl_h_cluster import relay_server
from harborrl_h_cluster.reverse_relay import pump


async def start_relay(environment, gateway_url):
    cfg = {**environment.cfg, **environment.cfg['options']}
    token = secrets.token_hex(32)
    await environment.upload_file(Path(relay_server.__file__), '/tmp/harborrl-relay.py')
    with tempfile.NamedTemporaryFile(mode='w', encoding='ascii') as secret:
        secret.write(token)
        secret.flush()
        await environment.upload_file(Path(secret.name), '/tmp/harborrl-relay-token')
    result = await environment.exec('chmod 600 /tmp/harborrl-relay-token && '
        'nohup python3 /tmp/harborrl-relay.py --token-file /tmp/harborrl-relay-token '
        f"--port {cfg['relay_port']} --timeout {cfg['relay_timeout_s']} "
        '> /tmp/harborrl-relay.log 2>&1 < /dev/null &')
    if result.return_code:
        raise RuntimeError('sandbox relay process failed to start')
    endpoint = await environment._sandbox.get_endpoint(cfg['relay_port'])
    url = endpoint.endpoint
    if not url.startswith(('http://', 'https://')):
        url = cfg['protocol'] + '://' + url
    headers = endpoint.headers or {}
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        for _ in range(20):
            try:
                response = await client.get(url.rstrip('/') + '/relay/health',
                                           headers={**headers, 'X-Relay-Token': token})
                if response.status_code == 200 and response.json() == {'ok': True}:
                    break
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError('sandbox relay readiness failed')
    publish(environment.evidence / 'relay-ready.json', {
        'transport': 'platform_pull', 'port': cfg['relay_port'],
        'upstream': gateway_url, 'sandbox_id': environment.record['sandbox_id'],
        'response_mode': 'buffered', 'timeout_s': cfg['relay_timeout_s']})
    return asyncio.create_task(pump(url, headers, token, gateway_url, asyncio.Event(),
                                    timeout=cfg['relay_timeout_s'] - 10))
