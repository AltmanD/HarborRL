#!/usr/bin/env python3
"""Collect service-mode run evidence for the controller-placement validation.

Reads one run directory and reports: exit code, policy versions, checkpoints,
rollout groups/slots (from the RolloutService artifacts), RL_READY counts,
rewards, trace seals, integrity errors, and cleanup records. Machine-readable
JSON goes to stdout; exit 0 only when the mandatory gates hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def main(run_dir):
    run_dir = Path(run_dir)
    report = {"run_dir": str(run_dir), "gates": {}}

    exit_code = int((run_dir / "exit-code").read_text().strip())
    report["exit_code"] = exit_code
    report["gates"]["exit_code"] = exit_code == 0

    # Policy versions: the manager's activation publishes policy-pool.json
    # (snapshot) and each completed group records its admitted version.
    versions = []
    snapshot = run_dir / "policy-pool.json"
    final_version = None
    if snapshot.is_file():
        final_version = read_json(snapshot).get("weight_version")
    for result in sorted(run_dir.glob("rollout-service/groups/*/group-*/result.json")):
        versions.append(read_json(result)["policy_version"])
    report["policy_versions"] = {"groups": versions, "final_pool": final_version}
    report["gates"]["policy_advanced"] = len(set(versions)) >= 2 and (
        final_version is not None and str(final_version) == str(versions[-1]))

    checkpoints = sorted(p.name for p in run_dir.glob("checkpoints/*/iter_*")) \
        if run_dir.joinpath("checkpoints").is_dir() else []
    report["checkpoints"] = checkpoints
    report["gates"]["checkpoints_written"] = len(checkpoints) >= 2

    # Service artifacts: groups executed by the manager.
    groups_root = run_dir / "rollout-service" / "groups"
    groups = {}
    if groups_root.is_dir():
        for run_id_dir in groups_root.iterdir():
            for group_dir in run_id_dir.iterdir():
                summary = {"group_id": group_dir.name, "states": [],
                           "has_result": (group_dir / "result.json").is_file()}
                for state in ("running", "result", "cancelled", "failed"):
                    if (group_dir / f"{state}.json").is_file():
                        summary["states"].append(state)
                groups[group_dir.name] = summary
    report["service_groups"] = groups
    report["gates"]["service_groups_complete"] = bool(groups) and all(
        g["states"] and "result" in g["states"] and g["has_result"]
        for g in groups.values())

    # Native rollouts + traces.
    attempts = list((run_dir / "native-rollouts").rglob("trajectory.v2.json")) \
        if (run_dir / "native-rollouts").is_dir() else []
    readiness, rewards, integrity, seals = {}, [], 0, {"sealed": 0, "unsealed": 0}
    turns_total = 0
    for path in attempts:
        ir = read_json(path)
        readiness[ir["readiness"]] = readiness.get(ir["readiness"], 0) + 1
        integrity += len(ir.get("integrity_errors", []))
        rewards.append(ir["evaluation"]["training_reward"])
        turns_total += len(ir["turns"])
        if ir["trace_seal"]["status"] == "sealed" and not ir["trace_seal"]["errors"]:
            seals["sealed"] += 1
        else:
            seals["unsealed"] += 1
    report["trajectories"] = {
        "count": len(attempts), "readiness": readiness, "rewards": rewards,
        "turns_total": turns_total, "integrity_errors": integrity, "seals": seals}
    report["gates"]["all_rl_ready"] = len(attempts) >= 6 and readiness.get("RL_READY") == len(attempts)
    report["gates"]["zero_integrity_errors"] = integrity == 0
    report["gates"]["zero_unsealed"] = seals["unsealed"] == 0

    cleanup = [p.name for p in run_dir.rglob("*cleanup-required.json")]
    unsealed = [p.name for p in run_dir.rglob("*trace-unsealed.json")]
    failed = [p.name for p in run_dir.rglob("*failure.json")]
    report["cleanup_records"] = {"cleanup_required": cleanup, "trace_unsealed": unsealed,
                                 "failures": failed}
    report["gates"]["clean_run"] = not (cleanup or unsealed or failed)

    passed = int(exit_code) == 0 and all(report["gates"].values())
    report["summary"] = ("PASS" if passed else "FAIL") + \
        f" gates={report['gates']}"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect_service_evidence.py RUN_DIR")
    raise SystemExit(main(sys.argv[1]))
