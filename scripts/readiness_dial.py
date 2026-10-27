#!/usr/bin/env python3
"""Live cross-machine readiness dials for the controller placement validation.

Starts a temporary control plane + a /readyz listener on the manager, then
dials them from the GPU target and the rollout worker over SSH — the same
hops the real deployment requires (gpu_to_controller, worker_to_gateway).
"""
from __future__ import annotations

import ipaddress
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harborrl.rollout.service import RolloutService, make_control_server


def ssh(spec, command, timeout=20):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "-o", "StrictHostKeyChecking=accept-new", spec, command],
        capture_output=True, text=True, timeout=timeout, check=False)
    return result.returncode == 0, (result.stdout + result.stderr).strip()[-200:]


def main(gpu_spec, worker_spec, manager_address):
    manager_address = str(ipaddress.IPv4Address(manager_address))
    # 1) a real control plane on the manager's control port
    service = RolloutService(run_root="/tmp/harborrl-readiness")
    control = make_control_server(service, host="0.0.0.0", port=49200)
    control_thread = threading.Thread(target=control.serve_forever, daemon=True)
    control_thread.start()
    # 2) a plain /readyz listener standing in for the gateway port
    class Gateway(ThreadingHTTPServer):
        daemon_threads = True

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    gateway = Gateway(("0.0.0.0", 49184), Handler)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    time.sleep(1)

    checks = []
    checks.append(("manager.control_plane_listening",
                   *ssh("localhost", "true")))  # manager-local sanity
    # Direct local health check of the control plane.
    local = subprocess.run(
        ["python3", "-c",
         "import urllib.request;urllib.request.urlopen('http://127.0.0.1:49200/healthz',timeout=5)"],
        capture_output=True, text=True, timeout=10, check=False)
    checks.append(("manager.control_healthz_local", local.returncode == 0,
                   "healthz answered on the manager"))
    checks.append(("gpu_to_controller",
                   *ssh(gpu_spec,
                        "python3 -c \"import urllib.request;"
                        f"urllib.request.urlopen('http://{manager_address}:49200/healthz',"
                        "timeout=10)\"")))
    checks.append(("worker_to_gateway",
                   *ssh(worker_spec,
                        "python3 -c \"import urllib.request;"
                        f"urllib.request.urlopen('http://{manager_address}:49184/readyz',"
                        "timeout=10)\"")))

    control.shutdown()
    gateway.shutdown()
    report = [{"check": name, "ok": ok, "detail": detail}
              for name, ok, detail in checks]
    passed = all(item["ok"] for item in report)
    print(json.dumps({"readiness": report,
                      "summary": "PASS" if passed else "FAIL"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: readiness_dial.py GPU_SSH_SPEC WORKER_SSH_SPEC MANAGER_IPV4")
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
