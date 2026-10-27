"""Run the locked upstream training loop with its supported actor-class factory."""
import functools
import importlib.util
import json
import os
from pathlib import Path

from .config import environment


class CheckedActorGroup:
    """Check every sync, including the final one, before returning to upstream."""
    def __init__(self, actor, manager, args):
        self.actor, self.manager, self.args = actor, manager, args

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def update_weights(self):
        import ray

        from harborrl.backends.slime_v032.rollout_service_client import (
            fetch_pool_states,
        )

        from .sglang import admit_pool
        result = self.actor.update_weights()
        if self.args.offload_rollout:
            ray.get(self.manager.onload_kv.remote())
        endpoint = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        admit_pool(fetch_pool_states(endpoint), os.environ['RUN_DIR'])
        return result


def main():
    import ray
    from slime.ray.placement_group import create_training_models
    from slime.utils.arguments import parse_args

    from .actor import ConsistencyActor

    args = parse_args()
    for name, expected in (("pipeline_model_parallel_size", 1), ("context_parallel_size", 1),
                           ("update_weights_interval", 1)):
        if getattr(args, name) != expected:
            raise ValueError(f"Consistency v1 requires {name}={expected}")
    if any(getattr(args, key, False) for key in ('use_critic', 'keep_old_actor', 'use_tis', 'release_train', 'use_rollout_routing_replay', 'num_experts')):
        raise ValueError("Consistency v1 supports synchronous Native actor training without routing replay")
    if args.update_weight_mode != "full" or args.update_weight_transport != "nccl":
        raise ValueError("Consistency v1 requires full tensor/NCCL weight synchronization")
    settings = environment()
    rollout_env = {key: value for key, value in os.environ.items()
                   if key.startswith('HARBORRL_') or key in ('RUN_DIR', 'RUN_ID', 'MODEL_TAG', 'MODEL_ARGS_FILE',
                                                           'ROLLOUT_MAX_RESPONSE_LEN', 'ROLLOUT_MAX_CONTEXT_LEN', 'PYTHONPATH')}
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': rollout_env})
    args.get_mismatch_metrics = True
    args.custom_tis_function_path = "harborrl.consistency.slime.identity_tis"
    args.train_env_vars = dict(args.train_env_vars or {})
    args.train_env_vars.update({"HARBORRL_CONSISTENCY": json.dumps(settings), "RUN_DIR": os.environ["RUN_DIR"],
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": os.environ["HARBORRL_NATIVE_TOKENIZER_DIGEST"],
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": os.environ["HARBORRL_NATIVE_TEMPLATE_DIGEST"]})
    spec = importlib.util.spec_from_file_location("harborrl_upstream_train", Path(os.environ["SLIME_DIR"]) / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the factory in this private entrypoint module is replaced; upstream files are untouched.
    factory = functools.partial(create_training_models, actor_cls=ConsistencyActor)
    def checked_factory(args, pgs, manager):
        actor, critic = factory(args, pgs, manager)
        return CheckedActorGroup(actor, manager, args), critic
    module.create_training_models = checked_factory
    module.train(args)


if __name__ == "__main__":
    main()
