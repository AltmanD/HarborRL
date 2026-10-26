"""One foreground CLI job at a time, with immutable inputs and durable logs."""
import json
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from harborrl.options.wizard import build_config_bundle, publish_bundle


def now():
    return datetime.now(timezone.utc).isoformat()


class Jobs:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.child = None
        self.active = None
        self.closed = False
        self.records = {}
        for path in self.root.glob("*/job.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["state"] in {"running", "stopping"}:
                record.update(state="unknown", message="Console restarted; verify target cleanup before launching again.")
            self.records[record["id"]] = record

    def _save(self, record):
        path = self.root / record["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _refresh(self):
        if self.child is not None and self.child.poll() is not None:
            record = self.records[self.active]
            code = self.child.returncode
            record.update(state="stopped" if record["state"] == "stopping" else "succeeded" if code == 0 else "failed",
                          exit_code=code, finished_at=now())
            self._save(record)
            self.child = self.active = None

    def _watch(self, child):
        child.wait()
        with self.lock:
            if self.child is child:
                self._refresh()

    def list(self):
        with self.lock:
            self._refresh()
            return [dict(record) for record in sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)]

    def start(self, configuration, bundle, mode):
        if mode not in {"dry-run", "doctor", "train"}:
            raise ValueError("Unknown run mode")
        with self.lock:
            self._refresh()
            if self.closed or self.child is not None:
                raise ValueError("A job is already active or the console is closing")
            if any(record["state"] == "unknown" for record in self.records.values()):
                raise ValueError("An interrupted job has unknown cleanup status. Reconcile it before launching again.")
            identifier = uuid4().hex
            root = self.root / identifier
            root.mkdir(mode=0o700)
            training = yaml.safe_load(bundle["files"]["config.yaml"])
            deployment = yaml.safe_load(bundle["files"]["deployment.yaml"])
            manifest = json.loads(bundle["files"]["setup-manifest.json"])
            output = str(Path(training["output"]["root"]) / "web" / identifier)
            training["output"]["root"] = deployment["storage"]["run_root"] = output
            snapshot = build_config_bundle({"answers_version": 1, "preset": manifest["preset"],
                                           "training": training, "deployment": deployment}, output_directory=root / "inputs")
            publish_bundle(snapshot, root / "inputs")
            manager = deployment["manager"]
            command = [manager["controller_python"], "-u", "-m", "harborrl.cli", "doctor" if mode == "doctor" else "train",
                       str(root / "inputs/config.yaml"), "--deployment", str(root / "inputs/deployment.yaml")]
            if mode == "dry-run":
                command.append("--dry-run")
            record = {"id": identifier, "configuration": configuration, "mode": mode, "state": "running",
                      "created_at": now(), "finished_at": None, "exit_code": None, "command": command,
                      "log_path": str(root / "console.log"), "snapshot": str(root / "inputs"), "output_root": output}
            self._save(record)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [manager["source_dir"], env.get("PYTHONPATH")]))
            try:
                with (root / "console.log").open("wb") as log:
                    self.child = subprocess.Popen(command, cwd=manager["source_dir"], env=env, stdin=subprocess.DEVNULL,
                                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                record.update(state="failed", message=str(exc), finished_at=now())
            else:
                self.active = identifier
                record["pid"] = self.child.pid
            self.records[identifier] = record
            self._save(record)
            if self.child is not None:
                threading.Thread(target=self._watch, args=(self.child,), name="harborrl-web-job", daemon=True).start()
            return dict(record)

    def stop(self, identifier):
        with self.lock:
            self._refresh()
            if identifier != self.active or self.child is None:
                raise ValueError("This job is not active")
            record = self.records[identifier]
            if record["state"] != "stopping":
                # The existing CLI forwards signals to its local/SSH training target.
                try:
                    self.child.send_signal(signal.SIGINT) if os.name == "posix" else self.child.terminate()
                except ProcessLookupError:
                    self._refresh()
                    return dict(record)
                record["state"] = "stopping"
                self._save(record)
            return dict(record)

    def reconcile(self, identifier):
        with self.lock:
            record = self.records[identifier]
            if record["state"] != "unknown":
                raise ValueError("Only an interrupted job can be reconciled")
            record.update(state="interrupted", finished_at=now(), message="Target cleanup confirmed by operator.")
            self._save(record)
            return dict(record)

    def log(self, identifier):
        with self.lock:
            self._refresh()
            record = self.records[identifier]
            path = self.root / identifier / "console.log"
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 65536))
                tail = stream.read(65536).decode("utf-8", errors="replace")
            result = {"job": dict(record), "text": tail, "bytes": size, "truncated": size > 65536,
                      "training_text": "", "training_log_path": None, "training_truncated": False}
            output = Path(record["output_root"])
            candidates = sorted(output.glob("training/*/train-stdout.log"))
            if candidates:
                training_log = candidates[-1]
                if training_log.resolve().is_relative_to(output.resolve()):
                    with training_log.open("rb") as stream:
                        length = stream.seek(0, os.SEEK_END)
                        stream.seek(max(0, length - 65536))
                        result.update(training_text=stream.read(65536).decode("utf-8", errors="replace"),
                                      training_log_path=str(training_log), training_truncated=length > 65536)
            return result

    def close(self):
        with self.lock:
            self.closed = True
            self._refresh()
            child = self.child
            if child:
                self.stop(self.active)
        if child:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with self.lock:
                    record = self.records[self.active]
                    record.update(state="unknown", message="Graceful shutdown timed out; target cleanup must be checked.")
                    self._save(record)
            else:
                with self.lock:
                    self._refresh()
