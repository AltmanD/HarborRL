"""Collect errors before raising so every participating rank makes the same decision."""
from pathlib import Path

from harborrl.trajectories.native import publish

from .config import violations
from .metrics import merge, statistics


def assess(rows, config, *, metadata=None):
    try:
        return {"statistics": statistics(rows, threshold=config["bad_token_delta"], worst=config["worst_tokens"]),
                "error": None, "metadata": metadata or {}}
    except (ValueError, TypeError, OverflowError) as exc:
        return {"statistics": None, "error": f"{type(exc).__name__}: {exc}", "metadata": metadata or {}}


def decide(parts, config, *, phase):
    structural = [p["error"] for p in parts if p["error"]]
    contributing = [p for p in parts if p.get("contribute", True)]
    metrics = None if structural else merge([p["statistics"] for p in contributing])
    numeric = violations(metrics, config) if metrics else []
    return {"schema_version": "harborrl-consistency-v1", "phase": phase, "mode": config["mode"],
            "status": "failed" if structural or (numeric and config["mode"] == "enforce") else "warning" if numeric else "passed",
            "structural_errors": structural, "numeric_violations": numeric, "metrics": metrics,
            "config": config, "ranks": [{"metadata": p["metadata"], "error": p["error"],
                "contribute": p.get("contribute", True),
                "worst": p["statistics"]["worst"] if p["statistics"] else []} for p in parts]}


def distributed_gate(part, config, path, *, phase, group=None):
    import torch.distributed as dist
    active = dist.is_available() and dist.is_initialized()
    parts = [part]
    if active:
        parts = [None] * dist.get_world_size(group)
        dist.all_gather_object(parts, part, group=group)
    report = decide(parts, config, phase=phase)
    write_error = None
    if not active or dist.get_rank(group) == 0:
        try:
            publish(Path(path), report)
        except (OSError, ValueError) as exc:
            write_error = str(exc)
    if active:
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, write_error, group=group)
        write_error = next((error for error in errors if error), None)
    if write_error or report["status"] == "failed":
        raise ValueError(f"Consistency {phase} rejected: {write_error or report['structural_errors'] or report['numeric_violations']}")
    return report
