"""Integration using the locked Slime precomputed log_probs and postprocess hook."""
import os
import struct
from pathlib import Path

from harborrl.trajectories.native import digest, publish, read_json

from .config import environment, require_raw_sampling
from .report import assess, distributed_gate


def evidence_key(tokens, logprobs, mask):
    q = [struct.unpack("!f", struct.pack("!f", value))[0] for value in logprobs]
    return digest({"tokens": tokens, "q": q, "mask": mask})


def record_turn(turn, tokens, logprobs, mask):
    config = environment()
    if config["mode"] == "off":
        return
    require_raw_sampling(turn["sampling"])
    if turn["logprob_semantics"] != "raw_model" or turn["evidence_kind"] != "serving":
        raise ValueError("Consistency requires observed raw serving evidence")
    key = evidence_key(tokens, logprobs, mask)
    version = str(turn["policy_version"])
    if not version.isdecimal():
        raise ValueError("Invalid serving version")
    payload = {"key": key, "weight_version": version, "sampling": turn["sampling"],
               "tokenizer_digest": turn["tokenizer_digest"], "template_digest": turn["template_digest"],
               "logprob_semantics": "raw_model_equals_sampling", "evidence_kind": "serving"}
    publish(Path(os.environ["RUN_DIR"]) / "consistency" / "turns" / version / f"{key}.json", payload)


def identity_tis(*args, **kwargs):
    raise RuntimeError("Consistency is metrics-only; invoking a TIS correction would change the Native objective")


def check_training_batch(args, rollout_id, data, validator):
    import torch.distributed as dist
    from megatron.core import mpu
    from slime.utils.distributed_utils import get_gloo_group

    from harborrl.backends.slime_v032.versions import read_locked_policy_version

    config = environment()
    root = Path(os.environ["RUN_DIR"])
    metadata = {"rank": dist.get_rank(), "rollout_id": rollout_id, "backend": "megatron",
                "scope": "pre-update actual packed rollout batch"}
    try:
        validator(args, rollout_id, data)
        if "log_probs" not in data or len(data["log_probs"]) != len(data["tokens"]):
            raise ValueError("Missing pre-update Megatron log_probs; --get-mismatch-metrics is required")
        version = read_locked_policy_version(root)
        metadata["weight_version"] = version
        cases = []
        for tokens, q, mask, p0 in zip(data["tokens"], data["rollout_log_probs"], data["loss_masks"], data["log_probs"], strict=True):
            tokens, q, mask, p0 = (x.detach().cpu().tolist() if hasattr(x, "detach") else list(x) for x in (tokens, q, mask, p0))
            # Slime stores rollout probabilities as float32. Normalize the lookup identically.
            key = evidence_key(tokens, q, mask)
            evidence = read_json(root / "consistency" / "turns" / version / f"{key}.json")
            if evidence["key"] != key or evidence["weight_version"] != version:
                raise ValueError("Frozen serving evidence does not match the training batch")
            for field, env in (("tokenizer_digest", "HARBORRL_NATIVE_TOKENIZER_DIGEST"),
                               ("template_digest", "HARBORRL_NATIVE_TEMPLATE_DIGEST")):
                if evidence[field] != os.environ[env]:
                    raise ValueError(f"Training batch {field} differs from launch identity")
            require_raw_sampling(evidence["sampling"])
            cases.append((p0, q, mask))
        part = assess(cases, config, metadata=metadata)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        part = {"statistics": None, "error": str(exc), "metadata": metadata}
    part["contribute"] = mpu.get_tensor_model_parallel_rank() == 0
    distributed_gate(part, config, root / "consistency" / f"batch-{rollout_id}.json",
                     phase="pre-update-backend-gap", group=get_gloo_group())
