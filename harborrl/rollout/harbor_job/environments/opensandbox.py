"""Audited lifecycle around Harbor's OpenSandbox environment."""
import asyncio
import fcntl
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import httpx
from harbor.environments.opensandbox import OpenSandboxEnvironment

from harborrl.rollout.harbor_job.sandbox_config import read_credential
from harborrl.trajectories.native import digest, publish


@asynccontextmanager
async def startup_slot(directory, count):
    """Process-shared creation slots; kernel releases locks on runner death."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    held = None
    try:
        while held is None:
            for index in range(count):
                handle = (directory / f'{index}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = handle
                    break
                except BlockingIOError:
                    handle.close()
            if held is None:
                await asyncio.sleep(0.2)
        yield
    finally:
        if held is not None:
            held.close()


class ManagedOpenSandboxEnvironment(OpenSandboxEnvironment):
    @classmethod
    def validate_config(cls, cfg):
        if cfg['options'] or any(b['options'] for b in cfg['bindings']):
            raise ValueError('standard OpenSandbox adapter does not accept platform options')

    @classmethod
    def agent_gateway(cls, cfg, gateway_url):
        return gateway_url

    @classmethod
    async def network_check(cls, cfg):
        return {'status': 'not_probed', 'reason': 'Use the explicit sandbox smoke test for allocation checks'}

    async def validate_binding(self):
        pass

    def extensions(self):
        return {}

    def api_params(self):
        return {}

    def startup_transport(self, allocated):
        from harborrl.rollout.harbor_job.sandbox_transport import (
            SandboxStartupTransport,
        )
        return SandboxStartupTransport(self.cfg['domain'].rstrip('/') + '/v1', allocated)

    def __init__(self, *args, sandbox_config, binding, evidence_dir, startup_dir, gateway_url=None, **kwargs):
        self.validate_config(sandbox_config)
        self.cfg, self.binding = sandbox_config, binding
        self.evidence = Path(evidence_dir)
        self.startup_dir = startup_dir
        self.gateway_url = gateway_url
        self._startup_transport = None
        self.record = {'runtime_digest': digest(sandbox_config),
                       'image': binding['image'], 'sandbox_id': None, 'state': 'pending', 'events': []}
        super().__init__(*args, domain=self.cfg['domain'], api_key=read_credential(self.cfg),
                         protocol=self.cfg['protocol'], use_server_proxy=True,
                         ready_timeout_sec=self.cfg['create_timeout_s'],
                         sandbox_timeout_sec=self.cfg['lifecycle_minutes'] * 60,
                         request_timeout_sec=self.cfg['request_timeout_s'], **kwargs)

    def record_state(self, state):
        self.record['state'] = state
        self.record['events'].append({'state': state, 'time': time.time()})
        publish(self.evidence / 'sandbox-events' / f'{len(self.record["events"]):04d}.json', self.record)
        with tempfile.NamedTemporaryFile(mode='w', dir=self.evidence, delete=False) as stream:
            json.dump(self.record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, self.evidence / 'sandbox-lifecycle.json')

    def cleanup_required(self, reason):
        publish(self.evidence / 'cleanup-required.json', {
            'sandbox_id': self.record['sandbox_id'], 'image': self.binding['image'],
            'reason': reason, 'runtime_digest': self.record['runtime_digest']})

    async def _create_sandbox(self, sdk):
        await self.validate_binding()
        async with startup_slot(self.startup_dir, self.cfg['startup_concurrency']):
            self.record_state('creating')
            def allocated(sandbox_id):
                self.record['sandbox_id'] = sandbox_id
                self.record_state('allocated')

            self._startup_transport = self.startup_transport(allocated)
            connection = self._build_connection_config(sdk).model_copy(update={'transport': self._startup_transport})
            try:
                sandbox = await asyncio.wait_for(sdk['Sandbox'].create(
                    self.binding['image'], connection_config=connection,
                    timeout=timedelta(minutes=self.cfg['lifecycle_minutes']),
                    ready_timeout=timedelta(seconds=self.cfg['create_timeout_s']),
                    extensions=self.extensions(),
                    resource=self._build_resource(), env=dict(self._persistent_env),
                    network_policy=self._build_network_policy(sdk), skip_health_check=True),
                    self.cfg['create_timeout_s'])
            except BaseException:  # noqa: BLE001 - persist unknown allocation on cancellation too
                if self.record['sandbox_id']:
                    await self._cleanup_allocated()
                else:
                    self.record_state('creation_unknown')
                    self.cleanup_required('Creation outcome unknown; reconcile platform inventory before retry')
                await self._startup_transport.aclose()
                raise RuntimeError('Sandbox creation outcome unknown') from None
            self._sandbox = sandbox
            self.record['sandbox_id'] = sandbox.id
            self.record_state('created')
            try:
                await asyncio.wait_for(self._wait_until_ready(sandbox), self.cfg['create_timeout_s'])
                self.record_state('ready')
                return sandbox
            except BaseException:  # noqa: BLE001 - cleanup before redacting SDK errors
                await self.stop(delete=True)
                raise RuntimeError('Sandbox readiness failed') from None

    async def _cleanup_allocated(self):
        """The SDK may already have killed an instance whose endpoints failed."""
        self.record_state('deleting')
        url = self.cfg['domain'].rstrip('/') + '/v1/sandboxes/' + self.record['sandbox_id']
        try:
            async with asyncio.timeout(self.cfg['cleanup_timeout_s']):
                async with httpx.AsyncClient(timeout=30, headers={'OPEN-SANDBOX-API-KEY': self._api_key},
                                             params=self.api_params()) as client:
                    status = await client.get(url)
                    if status.status_code != 404:
                        (await client.delete(url)).raise_for_status()
                        status = await client.get(url)
                    if status.status_code != 404:
                        raise RuntimeError('allocation deletion unconfirmed')
            self.record_state('deleted')
        except BaseException:  # noqa: BLE001 - retain debt if cleanup is interrupted
            self.record_state('delete_failed')
            self.cleanup_required('Allocated Sandbox deletion was not confirmed')

    async def start(self, force_build):
        try:
            await super().start(force_build)
        except BaseException:
            if self._sandbox is not None:
                await self.stop(delete=True)
            raise

    async def stop(self, delete=True):
        if self._sandbox is None:
            if self._startup_transport is not None:
                await self._startup_transport.aclose()
            return
        sandbox = self._sandbox
        self.record_state('deleting')
        try:
            await asyncio.wait_for(sandbox.kill(), self.cfg['cleanup_timeout_s'])
            self.record_state('deleted')
            (self.evidence / 'cleanup-required.json').unlink(missing_ok=True)
            self._sandbox = None
        except BaseException:  # noqa: BLE001 - preserve cleanup debt on cancellation
            self.record_state('delete_failed')
            self.cleanup_required('Sandbox kill was not acknowledged')
            raise RuntimeError('Sandbox deletion failed') from None
        finally:
            try:
                await asyncio.wait_for(sandbox.close(), 10)
            except Exception:  # noqa: BLE001 - close errors must not expose SAT headers
                self.logger.warning('Sandbox SDK connection close failed')
            if self._startup_transport is not None:
                await self._startup_transport.aclose()
