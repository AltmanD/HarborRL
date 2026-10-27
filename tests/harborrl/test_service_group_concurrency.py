import asyncio
import threading
from types import SimpleNamespace

import pytest

from harborrl.backends.slime_v032 import rollout, rollout_service_client


@pytest.mark.parametrize('fail', [False, True])
def test_service_groups_overlap_preserve_order_and_await_failure_cleanup(monkeypatch, fail):
    barrier = threading.Barrier(2, timeout=3)
    cancelled, completed = threading.Event(), threading.Event()
    activated, failed = threading.Event(), threading.Event()

    class Client:
        def __init__(self, *args):
            self.run_id = 'run'

        def announce_serving(self, _):
            pass

        def activate_policy(self, *_):
            activated.set()

        def submit_group(self, request):
            assert activated.is_set()
            barrier.wait()
            if fail and request['group_id'] == 'first':
                failed.set()
                raise ValueError('group failed')
            if fail:
                assert failed.wait(3)
                assert not cancelled.is_set()
            completed.set()
            return {'result': {'id': request['group_id']}}

        def cancel_run(self):
            cancelled.set()

    def load_result(run_id, request, result, *, allow_synthetic):
        assert allow_synthetic is False
        return [result['id']]

    monkeypatch.setenv('HARBORRL_ROLLOUT_SERVICE_URL', 'http://localhost')
    monkeypatch.setenv('RUN_ID', 'run')
    monkeypatch.delenv('HARBORRL_ROLLOUT_SERVICE_TOKEN_FILE', raising=False)
    monkeypatch.setattr(rollout_service_client, 'RolloutServiceClient', Client)
    monkeypatch.setattr(rollout_service_client, 'fetch_pool_states', lambda _: [{'weight_version': '1'}])
    monkeypatch.setattr(rollout_service_client, 'service_group_request', lambda group, *_: {'group_id': group})
    monkeypatch.setattr(rollout_service_client, 'load_group_result', load_result)
    monkeypatch.setattr(rollout, 'lock_policy_version', lambda _: '1')
    monkeypatch.setattr(rollout, 'sampling_params', lambda _: {})
    monkeypatch.setattr(rollout, 'bind_group_samples', lambda *_: None)
    args = SimpleNamespace(sglang_router_ip='localhost', sglang_router_port=8000)
    if fail:
        with pytest.raises(ValueError, match='group failed'):
            asyncio.run(rollout.run_service_groups(args, ['first', 'second'], 0))
        assert cancelled.is_set()
    else:
        assert asyncio.run(rollout.run_service_groups(args, ['first', 'second'], 0)) == [['first'], ['second']]
        assert not cancelled.is_set()
    assert completed.is_set()
