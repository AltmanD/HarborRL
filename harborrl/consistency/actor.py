"""Small Slime actor extension: snapshot trained-model canaries before weight sync."""
import os
from pathlib import Path

import torch
import torch.distributed as dist
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.data import DataIterator
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.backends.megatron_utils.model import forward_only
from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.reloadable_process_group import (
    destroy_process_groups,
    reload_process_groups,
)

from harborrl.trajectories.native import digest, publish


class ConsistencyActor(MegatronTrainRayActor):
    def update_weights(self):
        if self.args.offload_train:
            self.wake_up()
        self._switch_model("actor")
        from harborrl.rollout.runtime import tokenizer_fingerprint
        error = None
        try:
            actual = tokenizer_fingerprint(self.tokenizer)
            expected = (os.environ['HARBORRL_NATIVE_TOKENIZER_DIGEST'], os.environ['HARBORRL_NATIVE_TEMPLATE_DIGEST'])
            if actual != expected:
                raise ValueError('Trainer tokenizer/template differs from rollout launch identity')
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=get_gloo_group())
        if any(errors):
            raise ValueError(f'Invalid trainer identity: {errors}')
        cases = []
        # Literal token fixtures intentionally bypass all serving-side text templating.
        for prompt, answer in (("Compute 1 + 1 =", " 2.\n"), ("def square(x):\n", "    return x * x\n")):
            p = self.tokenizer.encode(prompt, add_special_tokens=True)
            r = self.tokenizer.encode(answer, add_special_tokens=False)
            if not p or not r:
                raise ValueError("Tokenizer produced an empty canary")
            cases.append({"prompt_ids": p, "response_ids": r})
        device = next(self.model[0].parameters()).device
        data = {"tokens": [torch.tensor(c["prompt_ids"] + c["response_ids"], device=device) for c in cases],
                "loss_masks": [torch.ones(len(c["response_ids"]), dtype=torch.int, device=device) for c in cases],
                "total_lengths": [len(c["prompt_ids"]) + len(c["response_ids"]) for c in cases],
                "response_lengths": [len(c["response_ids"]) for c in cases]}
        output = forward_only(get_log_probs_and_entropy, self.args, self.model,
                              [DataIterator(data, [[i] for i in range(len(cases))])], [len(cases)])
        for case, values in zip(cases, output["log_probs"], strict=True):
            case["logprobs"] = values.detach().float().cpu().tolist()
        fingerprints = []
        for name, parameter in named_params_and_buffers(self.args, self.model):
            flat = parameter.detach().reshape(-1)
            indices = torch.linspace(0, max(0, flat.numel() - 1), min(16, flat.numel()), device=flat.device).long()
            fingerprints.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                                 "sample": flat[indices].float().cpu().tolist()})
        local = {"rank": dist.get_rank(), "fingerprint": digest(fingerprints), "tensor_count": len(fingerprints)}
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local, group=get_gloo_group())
        config = self.hf_config.to_dict()
        reference = {"schema_version": "harborrl-sync-reference-v1", "cases": cases,
                     "model_config_digest": digest(config), "model_config": config,
                     "tokenizer_vocabulary_digest": digest(sorted(self.tokenizer.get_vocab().items())),
                     "parameter_shards": shards, "fingerprint_semantics": "16 deterministic values per tensor; not a full checksum"}
        if self.args.offload_train:
            self.sleep()
        super().update_weights()
        if self.args.offload_train:
            reload_process_groups()
        reference["weight_version"] = str(self.weight_updater.weight_version)
        reference["digest"] = digest(reference)
        error = None
        if dist.get_rank() == 0:
            try:
                publish(Path(os.environ["RUN_DIR"]) / "consistency" / f"sync-{reference['weight_version']}.json", reference)
            except (OSError, ValueError) as exc:
                error = str(exc)
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=get_gloo_group())
        if self.args.offload_train:
            destroy_process_groups()
        if any(errors):
            raise ValueError(f"Failed to publish synchronized model evidence: {errors}")
