import asyncio

import httpx
import pytest
from harbor.environments.opensandbox import OpenSandboxEnvironment
from harborrl_h_cluster.environment import BrainboxEnvironment, validate_environment


def settings():
    return {'domain': 'https://sandbox.example', 'request_timeout_s': 1, 'use_server_proxy': True,
            'options': {'project': 'test', 'gateway_transport': 'platform_pull', 'relay_port': 49185,
                        'relay_timeout_s': 180},
            'bindings': [{'image': 'image', 'max_live_instances': 2,
                          'options': {'environment_id': 'env-test', 'required_mount_paths': ['/data']}}]}


@pytest.mark.parametrize('memory,normalized', [('1024Mi', '1Gi'), ('2048Mi', '2Gi'), ('512Mi', '512Mi')])
def test_resource_units(monkeypatch, memory, normalized):
    monkeypatch.setattr(OpenSandboxEnvironment, '_build_resource', lambda _: {'cpu': '2', 'memory': memory})
    assert object.__new__(BrainboxEnvironment)._build_resource() == {'cpu': '2', 'memory': normalized}


@pytest.mark.parametrize('change', ['port', 'project', 'binding', 'duplicate'])
def test_options_reject_invalid_platform_configuration(change):
    cfg = settings()
    if change == 'port':
        cfg['options']['relay_port'] = 44772
    elif change == 'project':
        cfg['options']['project'] = ''
    elif change == 'binding':
        cfg['bindings'][0]['options']['environment_id'] = '../other'
    else:
        cfg['bindings'].append(cfg['bindings'][0])
    with pytest.raises(ValueError):
        BrainboxEnvironment.validate_config(cfg)


def test_platform_routing_and_extensions():
    cfg = settings()
    BrainboxEnvironment.validate_config(cfg)
    assert BrainboxEnvironment.agent_gateway(cfg, 'http://gateway') == 'http://127.0.0.1:49185'
    env = object.__new__(BrainboxEnvironment)
    env.cfg, env.binding = cfg, cfg['bindings'][0]
    assert env.extensions() == {'environmentId': 'env-test'}
    assert env.api_params() == {'project': 'test'}
    cfg['options']['gateway_transport'] = 'direct'
    assert BrainboxEnvironment.agent_gateway(cfg, 'http://gateway') == 'http://gateway'


@pytest.mark.parametrize('ports,capacity,mounts,accepted', [
    ([44772, 49185], 2, ['/data'], True), ([44772], 2, ['/data'], False),
    ([44772, 49185], 1, ['/data'], False), ([44772, 49185], 2, [], False),
])
def test_environment_preflight_checks_platform_contract(monkeypatch, ports, capacity, mounts, accepted):
    cfg = settings()

    def handler(request):
        assert request.method == 'GET'
        assert request.url.params['project'] == 'test'
        return httpx.Response(200, json={'image': {'uri': 'image'}, 'instanceCapacity': capacity,
            'ports': [{'containerPort': p} for p in ports], 'volumes': [{'mountPath': p} for p in mounts]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client(**kw, transport=httpx.MockTransport(handler)))
    if accepted:
        assert asyncio.run(validate_environment(cfg, cfg['bindings'][0], 'test-only'))['capacity'] == 2
    else:
        with pytest.raises(ValueError):
            asyncio.run(validate_environment(cfg, cfg['bindings'][0], 'test-only'))
