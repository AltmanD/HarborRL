"""Public core must remain usable without the optional SDK or cluster package."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from harborrl.rollout.harbor_job.sandbox_config import load_sandbox, read_credential


def test_config_loading_does_not_import_optional_packages(tmp_path):
    config = tmp_path / 'sandbox.yaml'
    config.write_text(yaml.safe_dump({'domain': 'https://sandbox.example', 'credential_file': 'key',
                                    'bindings': [{'image': 'image', 'max_live_instances': 1}]}))
    code = """
import sys
from harborrl.rollout.harbor_job.sandbox_config import load_sandbox
cfg = load_sandbox(sys.argv[1])
assert cfg['installation'] is None and cfg['options'] == {}
assert not any(k == 'harbor' or k.startswith(('harbor.', 'opensandbox', 'harborrl_h_cluster', 'torch', 'ray')) for k in sys.modules)
"""
    subprocess.run([sys.executable, '-c', code, str(config)], check=True)


def test_core_has_no_cluster_imports_or_platform_keys():
    root = Path(__file__).resolve().parents[2] / 'harborrl'
    for path in root.rglob('*.py'):
        source = path.read_text()
        assert 'harborrl_h_cluster' not in source, str(path)
        assert 'brainbox' not in source.lower(), str(path)
        assert 'environment_id' not in source, str(path)


@pytest.mark.parametrize('adapter', ['invalid', '../module:Class', 'module:Class:Other'])
def test_adapter_path_rejected_without_import(tmp_path, adapter):
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump({'domain': 'https://sandbox.example', 'credential_file': 'key',
        'bindings': [{'image': 'image', 'max_live_instances': 1}], 'adapter': adapter}))
    with pytest.raises(ValueError, match='module:Class'):
        load_sandbox(path)


def test_generic_token_does_not_require_platform_key_format(tmp_path):
    path = tmp_path / 'credential'
    path.write_text('test-only\n')
    path.chmod(0o600)
    assert read_credential({'credential_file': str(path)}) == 'test-only'


def test_nested_provider_secrets_never_enter_runtime_manifest(tmp_path):
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump({'domain': 'https://sandbox.example', 'credential_file': 'key',
        'bindings': [{'image': 'image', 'max_live_instances': 1}],
        'options': {'nested': {'api_key': 'test-only'}}}))
    with pytest.raises(ValueError, match='secrets'):
        load_sandbox(path)
