"""Hook assembly and dry-run support for the Slime v0.3.2 adapter."""
from __future__ import annotations

import sys

from harborrl.backends.slime_v032.versions import BACKEND_CONTRACT
from harborrl.math_registry import MATH_REGISTRY


def _hooks(contract=BACKEND_CONTRACT):
    if str(contract) != BACKEND_CONTRACT:
        raise ValueError(f"unsupported backend contract: {contract!r}")
    return MATH_REGISTRY.algorithm_for_contract(str(contract)).backend_hooks


def hook_arguments(contract=BACKEND_CONTRACT):
    hooks = _hooks(contract)
    return [
        "--rollout-function-path", hooks["rollout"],
        "--custom-convert-samples-to-train-data-path", hooks["converter"],
        "--custom-advantage-function-path", hooks["advantage"],
        "--loss-type", "custom_loss",
        "--custom-loss-function-path", hooks["loss"],
        "--rollout-data-postprocess-path", hooks["postprocess"],
    ]


def validate_hook_assignment(assignment, *, contract=BACKEND_CONTRACT):
    hooks = _hooks(contract)
    observed = {key: value for key, value in assignment.items() if key in hooks}
    if observed != hooks:
        raise ValueError(f"Slime hook assignment must exactly match {BACKEND_CONTRACT}")
    return observed


def hook_modules():
    return {name: path.rsplit(".", 1)[0] for name, path in _hooks().items()}


def dry_run_command(entrypoint, base_args=(), *, contract=BACKEND_CONTRACT, python=sys.executable):
    return [
        str(python), "-u", str(entrypoint),
        *[str(arg) for arg in base_args],
        *hook_arguments(contract),
    ]
