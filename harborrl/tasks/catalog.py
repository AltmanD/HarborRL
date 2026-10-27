"""Compatibility entry point for the selected Harbor task provider."""
from __future__ import annotations

from harborrl.math_registry import select_math


def catalog(config):
    return select_math(config).task.adapter.catalog(config)
