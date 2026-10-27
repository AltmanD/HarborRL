"""Explicit single-instance live probe. Uses real quota and always attempts deletion."""
import argparse
import asyncio
import json
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from harborrl.rollout.harbor_job.sandbox_config import environment_class, load_sandbox
from harborrl.rollout.harbor_job.sandbox_probe import check
from harborrl.trajectories.native import publish


async def run(args):
    cfg = load_sandbox(args.config)
    check(cfg)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    task = root / 'environment'
    task.mkdir()
    binding = cfg['bindings'][0]
    adapter = environment_class(cfg)
    env = adapter(
        environment_dir=task, environment_name='sandbox-smoke', session_id=root.name,
        trial_paths=TrialPaths(root / 'trial'), task_env_config=EnvironmentConfig(docker_image=binding['image']),
        sandbox_config=cfg, binding=binding, evidence_dir=str(root), startup_dir=str(root / 'startup'),
        gateway_url=args.gateway)
    results = {}
    try:
        await env.start(False)
        ok = await env.exec('printf sandbox-ok')
        assert ok.return_code == 0 and ok.stdout == 'sandbox-ok', 'positive command failed'
        negative = await env.exec('exit 7')
        assert negative.return_code == 7, 'negative command exit code lost'
        source = root / 'source.bin'
        source.write_bytes(b'harborrl\x00sandbox\xff\n')
        await env.upload_file(source, '/tmp/harborrl-smoke.bin')
        await env.download_file('/tmp/harborrl-smoke.bin', root / 'download.bin')
        assert source.read_bytes() == (root / 'download.bin').read_bytes(), 'file roundtrip failed'
        agent_gateway = adapter.agent_gateway(cfg, args.gateway)
        gateway = urlsplit(agent_gateway)
        if gateway.scheme not in ('http', 'https') or not gateway.hostname or gateway.path not in ('', '/'):
            raise ValueError('gateway must be an HTTP origin')
        command = f'curl -sS -o /dev/null --connect-timeout 10 --max-time 20 {shlex.quote(agent_gateway)}'
        connected = await env.exec(command)
        publish(root / 'gateway-probe.json', {'exit_code': connected.return_code,
                                            'stdout': connected.stdout, 'stderr': connected.stderr})
        assert connected.return_code == 0, 'Sandbox cannot reach Gateway HTTP endpoint'
        results.update(command=True, negative_exit=True, files=True, gateway_http=True)
    finally:
        await env.stop(True)
    results['deleted'] = True
    publish(root / 'smoke-result.json', results)
    print(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gateway', required=True)
    asyncio.run(run(parser.parse_args()))
