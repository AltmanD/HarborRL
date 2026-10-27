"""Native Harbor task discovery, locking, and prompt materialization."""
from __future__ import annotations

import re
from pathlib import Path

from harborrl.data.harbor.native_inspector import inspect_native
from harborrl.trajectories.native import RewardProfile, read_json


class NativeTaskProvider:
    provider_id = "native_harbor"

    def discover(self, tasks_dir):
        return sorted(path.parent for path in Path(tasks_dir).rglob("task.toml"))

    def inspect(self, path, *, expected_digest=None):
        return inspect_native(path, expected_digest=expected_digest)

    def catalog(self, config):
        path = Path(config["tasks"]["catalog"])
        entries = read_json(path)
        if not isinstance(entries, list) or not entries:
            raise ValueError("catalog must be a nonempty JSON list")
        seen = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry)
                != {"id", "revision", "path", "task_digest", "reward_profile"}
                or not isinstance(entry["id"], str)
                or not entry["id"].strip()
                or not isinstance(entry["revision"], str)
                or not entry["revision"].strip()
                or entry["id"] in seen
            ):
                raise ValueError("invalid or duplicate catalog identity")
            if (
                not isinstance(entry["path"], str)
                or not entry["path"].strip()
                or not isinstance(entry["task_digest"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["task_digest"])
                or not isinstance(entry["reward_profile"], dict)
            ):
                raise ValueError(f'invalid native task lock: {entry["id"]}')
            seen.add(entry["id"])
            entry["path"] = str((path.parent / entry["path"]).resolve())
            report = self.inspect(entry["path"], expected_digest=entry["task_digest"])
            if report["status"] != "NEEDS_PROBE":
                raise ValueError(
                    f'task preflight failed: {entry["id"]}: {report["reasons"]}'
                )
            RewardProfile(**entry["reward_profile"])
        selected = set(getattr(config, "task_ids", ()))
        if selected:
            missing = selected - {entry["id"] for entry in entries}
            if missing:
                raise ValueError(f"task selector not found in catalog: {min(missing)}")
            entries = [entry for entry in entries if entry["id"] in selected]
        return entries

    def prompt_rows(self, entries):
        rows = []
        for entry in entries:
            instruction = (Path(entry["path"]) / "instruction.md").read_text()
            rows.append(
                {
                    "task": instruction,
                    "metadata": {"harborrl_native": True, "native_task": entry},
                }
            )
        return rows
