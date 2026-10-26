"""Explicit deployment-specific tolerances; no universal numeric threshold."""
import json
import os

from harborrl.trajectories.native import finite

DEFAULTS = {"mode": "off", "max_abs_delta": None, "max_mean_abs_delta": None,
            "bad_token_delta": None, "max_bad_fraction": 0.0, "worst_tokens": 8}


def validate(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown consistency configuration fields")
    result = {**DEFAULTS, **value}
    if result["mode"] not in ("off", "audit", "enforce"):
        raise ValueError("consistency.mode must be off, audit or enforce")
    for key in ("max_abs_delta", "max_mean_abs_delta", "bad_token_delta"):
        if result[key] is not None and (not finite(result[key]) or result[key] < 0):
            raise ValueError(f"consistency.{key} must be nonnegative and finite")
    if result["mode"] == "enforce" and result["max_abs_delta"] is None:
        raise ValueError("enforce requires an explicit max_abs_delta for this deployment")
    if not finite(result["max_bad_fraction"]) or not 0 <= result["max_bad_fraction"] <= 1:
        raise ValueError("max_bad_fraction must be in [0, 1]")
    if type(result["worst_tokens"]) is not int or not 0 <= result["worst_tokens"] <= 32:
        raise ValueError("worst_tokens must be an integer in [0, 32]")
    return result


def environment():
    return validate(json.loads(os.environ.get("HARBORRL_CONSISTENCY", "{}")))


def require_raw_sampling(sampling):
    allowed = {"temperature", "top_p", "top_k", "max_new_tokens", "stop", "raw_logprobs"}
    if (not isinstance(sampling, dict) or set(sampling) - allowed
            or sampling.get("temperature") != 1 or sampling.get("top_p") != 1
            or sampling.get("top_k", -1) not in (-1, 0)
            or sampling.get("raw_logprobs", True) is not True):
        raise ValueError("Consistency v1 requires untransformed raw-policy sampling")


def violations(metrics, config):
    errors = []
    for option, field in (("max_abs_delta", "max_abs_delta"), ("max_mean_abs_delta", "mean_abs_delta")):
        if config[option] is not None and metrics[field] > config[option]:
            errors.append(f"{field}={metrics[field]:.8g} exceeds {config[option]}")
    if config["bad_token_delta"] is not None and metrics["bad_fraction"] > config["max_bad_fraction"]:
        errors.append(f"bad_fraction={metrics['bad_fraction']:.8g} exceeds {config['max_bad_fraction']}")
    return errors
