import asyncio
import base64
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from harborrl_h_cluster.relay_server import Mailbox, make_server
from harborrl_h_cluster.reverse_relay import pump


@contextlib.contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_relay_auth_timeout_and_no_redelivery():
    async def run(url):
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(url + '/relay/next')).status_code == 403
            assert (await client.post(url + '/unexpected', content=b'x')).status_code == 404
            pending = asyncio.create_task(client.post(url + '/v1/messages', content=b'body'))
            headers = {'X-Relay-Token': 'secret'}
            for _ in range(100):
                item = await client.get(url + '/relay/next', headers=headers)
                if item.status_code == 200:
                    break
                await asyncio.sleep(.01)
            assert item.status_code == 200
            assert (await client.get(url + '/relay/next', headers=headers)).status_code == 204
            assert (await pending).status_code == 504
            delivered = await client.post(url + '/relay/result/' + item.json()['id'],
                headers=headers, json={'status': 200, 'headers': {}, 'body': base64.b64encode(b'late').decode()})
            assert delivered.status_code == 409
    with serving(make_server('127.0.0.1', 0, Mailbox('secret', timeout=.5))) as url:
        asyncio.run(run(url))


@pytest.mark.parametrize('status,content_type,body', [
    (200, 'text/event-stream', b'event: message_start\ndata: {"id":"test"}\n\n'),
    (429, 'application/json', b'{"error":"rate_limited"}'),
])
def test_reverse_roundtrip_preserves_bytes_and_headers(status, content_type, body):
    calls = []
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append((self.path, self.headers.get('x-api-key'),
                          self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    async def run(relay, gateway):
        stop = asyncio.Event()
        task = asyncio.create_task(pump(relay, {}, 'secret', gateway, stop))
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(relay + '/v1/messages?beta=true',
                    headers={'x-api-key': 'attempt-token'}, content=b'{"stream":true}')
            assert response.status_code == status
            assert response.content == body
            assert response.headers['content-type'] == content_type
        finally:
            stop.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    with serving(make_server('127.0.0.1', 0, Mailbox('secret'))) as relay, \
            serving(ThreadingHTTPServer(('127.0.0.1', 0), Upstream)) as gateway:
        asyncio.run(run(relay, gateway))
    assert calls == [('/v1/messages?beta=true', 'attempt-token', b'{"stream":true}')]


def test_real_gateway_trace_and_session_audit_through_relay(tmp_path):
    from harborrl.gateway.server import Gateway
    from harborrl.gateway.server import make_server as make_gateway
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.harbor_job.audit import audit_session
    from harborrl.trajectories.native import Identity, RewardProfile, digest

    class Backend:
        def count_tokens(self, converted):
            return 2

        def generate(self, converted, ident, response_id):
            return {'input_ids': [1, 2], 'output_ids': [3], 'logprobs': [-.2], 'logprob_semantics': 'raw_model',
                    'policy_version': ident.policy_version, 'tokenizer_digest': 'tok', 'template_digest': 'tpl',
                    'engine_id': 'synthetic', 'serving_input': {'input_ids': [1, 2]}, 'sampling': converted['sampling'],
                    'finish_reason': 'end_turn', 'evidence_kind': 'synthetic', 'content': [{'type': 'text', 'text': 'done'}]}

    sampling = {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'max_new_tokens': 64, 'stop': []}
    identity = Identity(run_id='run', batch_id='batch', group_id='group', slot_id='0', attempt_id='attempt',
                        trajectory_id='trajectory', task_digest='task', policy_version='1', harness_digest='claude',
                        reward_digest=RewardProfile().digest, sampling_digest=digest(sampling))
    registry = TraceRegistry(tmp_path / 'traces')
    credential = registry.register(identity, 'trial')
    gateway = Gateway(registry, Backend(), model='policy')

    async def run(relay, upstream):
        stop = asyncio.Event()
        task = asyncio.create_task(pump(relay, {}, 'secret', upstream, stop))
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(relay + '/v1/messages',
                    headers={'x-api-key': credential}, json={'model': 'policy', 'max_tokens': 64,
                        'messages': [{'role': 'user', 'content': 'hello'}]})
                assert response.status_code == 200
                message = response.json()
            session = tmp_path / 'session.jsonl'
            session.write_text(json.dumps({'sessionId': 'relay-session', 'type': 'assistant',
                                           'message': message}) + '\n')
            audit_session(session, registry, identity.attempt_id)
            registry.close(identity.attempt_id)
            turns, seal = registry.seal(identity.attempt_id)
            assert seal['errors'] == []
            assert len(turns) == 1
            assert turns[0]['delivery'] == 'consumed_confirmed'
        finally:
            stop.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    with serving(make_server('127.0.0.1', 0, Mailbox('secret'))) as relay, \
            serving(make_gateway(gateway)) as upstream:
        asyncio.run(run(relay, upstream))
