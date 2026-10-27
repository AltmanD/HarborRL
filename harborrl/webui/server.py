"""Loopback-only HTTP boundary for the existing configuration services."""
import hmac
import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from harborrl.options.catalog import build_catalog
from harborrl.options.presets import PRESETS, defaults
from harborrl.options.validation import read_mapping, validate_mappings
from harborrl.options.wizard import build_config_bundle, publish_bundle

from .jobs import Jobs

ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/lucide.min.js": ("lucide.min.js", "text/javascript; charset=utf-8"),
          "/brand.jpg": ("brand.jpg", "image/jpeg")}
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}")


class Console:
    def __init__(self, workspace):
        self.workspace = Path(workspace).expanduser().resolve()
        self.configurations = self.workspace / "configurations"
        self.configurations.mkdir(parents=True, exist_ok=True)
        self.jobs = Jobs(self.workspace / "jobs")
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def directory(self, name):
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("Name must be 1-64 letters, digits, underscores or hyphens")
        path = self.configurations / name
        if path.is_symlink() or path.resolve().parent != self.configurations:
            raise ValueError("Configuration must stay inside the workspace")
        return path

    def bootstrap(self):
        return {"workspace": str(self.workspace), "catalog": build_catalog(),
                "live_supported": os.name == "posix",
                "defaults": {item["id"]: dict(zip(("training", "deployment"),
                              defaults(item["id"], self.configurations / "experiment"))) for item in PRESETS},
                "configurations": self.list_configurations(), "jobs": self.jobs.list()}

    def list_configurations(self):
        result = []
        for path in sorted(self.configurations.iterdir()):
            if path.is_dir() and not path.is_symlink() and NAME.fullmatch(path.name) and (path / "setup-manifest.json").is_file():
                result.append({"id": path.name, "path": str(path)})
        return result

    def preview(self, body):
        output = self.directory(body["name"])
        bundle = build_config_bundle(body["answers"], output_directory=output)
        if body.get("check_files") is True:
            bundle["validation"] = validate_mappings(yaml.safe_load(bundle["files"]["config.yaml"]),
                yaml.safe_load(bundle["files"]["deployment.yaml"]), config_path=output / "config.yaml", check_files=True)
        return bundle

    def save(self, body):
        with self.lock:
            output = self.directory(body["name"])
            bundle = build_config_bundle(body["answers"], output_directory=output)
            publish_bundle(bundle, output)
            return {"id": output.name, "path": str(output), "validation": bundle["validation"]}

    def configuration(self, identifier):
        root = self.directory(identifier)
        manifest = json.loads((root / "setup-manifest.json").read_text(encoding="utf-8"))
        answers = {"answers_version": 1, "preset": manifest["preset"],
                   "training": read_mapping(root / "config.yaml"), "deployment": read_mapping(root / "deployment.yaml")}
        return {"id": identifier, "answers": answers,
                "files": {name: (root / name).read_text(encoding="utf-8") for name in
                          ("config.yaml", "deployment.yaml", "setup-manifest.json")}}

    def start(self, body):
        mode = body["mode"]
        if mode not in {"dry-run", "doctor", "train"}:
            raise ValueError("Unknown run mode")
        if mode != "dry-run" and (body.get("confirmed") is not True or os.name != "posix"):
            raise ValueError("Live jobs require explicit confirmation and a Linux/POSIX controller")
        with self.lock:
            identifier = body["configuration"]
            root = self.directory(identifier)
            config = self.configuration(identifier)
            result = validate_mappings(yaml.safe_load(config["files"]["config.yaml"]),
                yaml.safe_load(config["files"]["deployment.yaml"]), config_path=root / "config.yaml", check_files=True)
            if result["status"] == "invalid":
                raise ValueError("; ".join(row["message"] for row in result["checks"] if row["status"] == "failed"))
            return self.jobs.start(identifier, config, mode)


def make_server(console, port=0):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_args):
            pass

        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            for name, value in {"Content-Type": content_type, "Content-Length": str(len(data)),
                                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self, method):
            try:
                host = self.headers.get("Host", "")
                if urlsplit("http://" + host).hostname not in {"127.0.0.1", "localhost"}:
                    return self.respond(403, {"error": "Loopback Host required"})
                path = urlsplit(self.path).path
                if method == "GET" and path in ASSETS:
                    name, mime = ASSETS[path]
                    return self.respond(200, files("harborrl.webui").joinpath("static", name).read_bytes(), mime)
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + console.token):
                    return self.respond(403, {"error": "Console session expired or not authorized. Reopen the CLI URL."})
                if method == "GET":
                    if path == "/api/bootstrap":
                        return self.respond(200, console.bootstrap())
                    if path == "/api/jobs":
                        return self.respond(200, console.jobs.list())
                    if path.startswith("/api/jobs/"):
                        return self.respond(200, console.jobs.log(path.removeprefix("/api/jobs/")))
                    if path.startswith("/api/configurations/"):
                        return self.respond(200, console.configuration(path.removeprefix("/api/configurations/")))
                if method == "POST":
                    if self.headers.get("Origin") != "http://" + host:
                        return self.respond(403, {"error": "Same-origin request required"})
                    if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        return self.respond(415, {"error": "JSON body required"})
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1048576:
                        return self.respond(413, {"error": "Request must be between 1 byte and 1 MiB"})
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise TypeError("Request must be an object")
                    if path == "/api/preview":
                        return self.respond(200, console.preview(body))
                    if path == "/api/configurations":
                        return self.respond(201, console.save(body))
                    if path == "/api/jobs":
                        return self.respond(201, console.start(body))
                    if path == "/api/stop":
                        return self.respond(200, console.jobs.stop(body["id"]))
                    if path == "/api/reconcile":
                        if body.get("confirmed") is not True:
                            raise ValueError("Confirm target cleanup before reconciling")
                        return self.respond(200, console.jobs.reconcile(body["id"]))
                    if path == "/api/import":
                        answers = yaml.safe_load(body["text"])
                        if (not isinstance(answers, dict) or type(answers.get("answers_version")) is not int
                                or answers.get("answers_version") != 1):
                            raise ValueError("Import an answers_version: 1 YAML/JSON mapping")
                        if set(answers) - {"answers_version", "preset", "training", "deployment", "sandbox_config"}:
                            raise ValueError("Unknown answer fields")
                        if any(not isinstance(answers.get(key, {}), dict) for key in ("training", "deployment")):
                            raise ValueError("training and deployment answers must be objects")
                        if answers.get("preset") not in {p["id"] for p in PRESETS}:
                            raise ValueError("Unknown deployment preset")
                        return self.respond(200, {"answers": answers})
                self.respond(404, {"error": "Not found"})
            except (ValueError, TypeError, KeyError, OSError, AttributeError, yaml.YAMLError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
