"""Offline dependency check or explicit live Environment check (no allocation)."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from harborrl.rollout.harbor_job.sandbox_config import (
    environment_class,
    load_sandbox,
    read_credential,
)


def check(cfg):
    from harborrl.rollout.harbor_job.runner import probe
    result = probe()
    version = importlib.metadata.version('opensandbox')
    if version != '0.1.9':
        raise ValueError('sandbox runner requires opensandbox==0.1.9')
    from opensandbox.models.execd import RunCommandOpts

    adapter = environment_class(cfg)
    if not {'timeout', 'envs', 'working_directory'} <= RunCommandOpts.model_fields.keys():
        raise ValueError('unsupported SDK command options')
    checksum = None
    if cfg['installation']:
        with Path(cfg['installation']['binary']).open('rb') as source:
            checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        if checksum != cfg['installation']['sha256']:
            raise ValueError('offline binary checksum mismatch')
    read_credential(cfg)
    result.update(opensandbox_version=version, environment=adapter.__name__, binary_sha256=checksum)
    return result


async def network_check(cfg):
    return await environment_class(cfg).network_check(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?')
    parser.add_argument('--stdin', action='store_true')
    parser.add_argument('--network', action='store_true')
    args = parser.parse_args()
    try:
        cfg = json.load(sys.stdin) if args.stdin else load_sandbox(args.config)
        result = check(cfg)
        if args.network:
            result['environments'] = asyncio.run(network_check(cfg))
        print(json.dumps(result))
    except Exception as exc:  # noqa: BLE001 - redact authenticated SDK exceptions
        # SDK/HTTP exceptions may contain authenticated endpoint URLs.
        print(json.dumps({'ok': False, 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
