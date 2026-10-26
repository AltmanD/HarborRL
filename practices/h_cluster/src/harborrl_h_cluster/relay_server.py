"""Sandbox-local HTTP mailbox. Standalone stdlib file, uploaded by the runner.

The manager pulls each request once. Upstream inference is never retried after
an ambiguous transport failure. This prototype buffers responses, including SSE.
"""
import argparse
import base64
import hmac
import json
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 16 * 1024 * 1024
PATHS = {'/v1/messages', '/v1/messages/count_tokens'}
REQUEST_HEADERS = {'content-type', 'x-api-key', 'authorization', 'anthropic-version', 'anthropic-beta'}
RESPONSE_HEADERS = {'content-type', 'request-id', 'retry-after'}


class Mailbox:
    def __init__(self, token, *, timeout=120, capacity=8):
        self.token, self.timeout = token, timeout
        self.slots = threading.BoundedSemaphore(capacity)
        self.queue = queue.Queue(maxsize=capacity)
        self.pending = {}
        self.lock = threading.Lock()


def make_server(host, port, mailbox):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, headers=None):
            self.send_response(status)
            for key, value in (headers or {}).items():
                if key.lower() in RESPONSE_HEADERS:
                    self.send_header(key, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, status, data):
            self.reply(status, json.dumps(data).encode(), {'content-type': 'application/json'})

        def authorized(self):
            return hmac.compare_digest(self.headers.get('X-Relay-Token', ''), mailbox.token)

        def read_body(self):
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= MAX_BODY or self.headers.get('Transfer-Encoding'):
                raise ValueError('invalid body length')
            return self.rfile.read(length)

        def do_GET(self):
            if not self.authorized():
                self.json_reply(403, {'error': 'forbidden'})
                return
            if self.path == '/relay/health':
                self.json_reply(200, {'ok': True})
                return
            if self.path != '/relay/next':
                self.json_reply(404, {'error': 'not_found'})
                return
            while True:
                try:
                    item = mailbox.queue.get_nowait()
                except queue.Empty:
                    self.reply(204, b'')
                    return
                with mailbox.lock:
                    active = item['id'] in mailbox.pending
                if active:
                    self.json_reply(200, item)
                    return

        def do_POST(self):
            try:
                self.handle_post()
            except (ValueError, KeyError, TypeError):
                self.json_reply(400, {'error': 'invalid_request'})

        def handle_post(self):
            if self.path.startswith('/relay/result/'):
                if not self.authorized():
                    self.json_reply(403, {'error': 'forbidden'})
                    return
                data = json.loads(self.read_body())
                if type(data['status']) is not int or not 200 <= data['status'] <= 599:
                    raise ValueError('invalid status')
                body = base64.b64decode(data['body'], validate=True)
                headers = data['headers']
                if not isinstance(headers, dict) or any(
                    not isinstance(k, str) or not isinstance(v, str) or '\r' in v or '\n' in v
                    for k, v in headers.items()
                ):
                    raise ValueError('invalid response headers')
                with mailbox.lock:
                    pending = mailbox.pending.get(self.path.rsplit('/', 1)[-1])
                    if pending is None or pending['event'].is_set():
                        self.json_reply(409, {'error': 'request_expired_or_completed'})
                        return
                    pending['result'] = (data['status'], body, headers)
                    pending['event'].set()
                self.json_reply(200, {'ok': True})
                return
            if self.path.split('?', 1)[0] not in PATHS:
                self.json_reply(404, {'error': 'not_found'})
                return
            body = self.read_body()
            if not mailbox.slots.acquire(blocking=False):
                self.json_reply(503, {'error': 'relay_capacity'})
                return
            identity = secrets.token_hex(16)
            pending = {'event': threading.Event()}
            try:
                with mailbox.lock:
                    mailbox.pending[identity] = pending
                try:
                    mailbox.queue.put_nowait({'id': identity, 'path': self.path, 'method': 'POST',
                        'headers': {k: v for k, v in self.headers.items() if k.lower() in REQUEST_HEADERS},
                        'body': base64.b64encode(body).decode()})
                except queue.Full:
                    self.json_reply(503, {'error': 'relay_queue_full'})
                    return
                if not pending['event'].wait(mailbox.timeout):
                    self.json_reply(504, {'error': 'relay_deadline'})
                    return
                self.reply(*pending['result'])
            finally:
                with mailbox.lock:
                    mailbox.pending.pop(identity, None)
                mailbox.slots.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--port', type=int, default=49185)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    server = make_server('0.0.0.0', args.port,
                         Mailbox(Path(args.token_file).read_text().strip(), timeout=args.timeout))
    server.serve_forever()
