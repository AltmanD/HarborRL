"""Pinned native Claude executable deployment, without package-manager fallback."""
import asyncio
import hashlib
import json
import shlex
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode

from harborrl.trajectories.native import publish


def binary_checksum(binary):
    with binary.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class OfflineClaudeCode(ClaudeCode):
    def __init__(self, *args, installation, **kwargs):
        self.installation = installation
        super().__init__(*args, **kwargs)

    async def install(self, environment):
        lock = self.installation
        binary = Path(lock['binary'])
        checksum = await asyncio.to_thread(binary_checksum, binary)
        if checksum != lock['sha256']:
            raise ValueError('offline Claude binary checksum mismatch')
        # Harbor invokes Claude with this HOME-local prefix on every command.
        probe = ('import json, os, platform; print(json.dumps(dict('
                 'system=platform.system(), machine=platform.machine(), '
                 'libc=os.confstr("CS_GNU_LIBC_VERSION"), home=os.environ.get("HOME", ""))))')
        result = await environment.exec('python3 -c ' + shlex.quote(probe))
        try:
            system = json.loads(result.stdout or '')
        except ValueError:
            system = {}
        if (result.return_code or system.get('system') != 'Linux' or system.get('machine') != 'x86_64'
                or not str(system.get('libc', '')).startswith('glibc ')):
            raise ValueError('offline binary requires Linux x86_64 glibc and a writable HOME')
        home = system.get('home', '')
        if not home.startswith('/') or home == '/':
            raise ValueError('invalid agent HOME')
        target = home + '/.local/bin/claude'
        staging = target + '.harborrl-upload'
        result = await environment.exec('mkdir -p ' + shlex.quote(home + '/.local/bin'))
        if result.return_code:
            raise ValueError('agent tool directory is not writable')
        await environment.upload_file(binary, staging)
        command = (f'printf "%s  %s\\n" {shlex.quote(lock["sha256"])} {shlex.quote(staging)} | sha256sum -c -'
                   f' && chmod 755 {shlex.quote(staging)} && mv {shlex.quote(staging)} {shlex.quote(target)}')
        result = await environment.exec(command)
        if result.return_code:
            raise ValueError('offline Claude installation checksum/permissions failed')
        result = await environment.exec(f'{shlex.quote(target)} --version')
        version = (result.stdout or '').strip().split()
        if result.return_code or not version or version[0] != lock['version']:
            raise ValueError('offline Claude installation/version probe failed')
        publish(self.logs_dir / 'offline-installation.json', {
            'version': lock['version'], 'sha256': lock['sha256'], 'target': target,
            'platform': system, 'mode': lock['mode']})
