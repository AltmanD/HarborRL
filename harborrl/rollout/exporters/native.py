"""Compatibility record view of the canonical trajectory-weighted GRPO batch.

No independent advantage calculation or secondary whitening occurs here.
"""
from harborrl.export.native import export_training_batch
from harborrl.export.validate import token_weight
from harborrl.trajectories.native import Identity


def export_group(trajectories, group_size, *, epsilon=1e-6, allow_synthetic=False):
    if type(group_size) is not int or group_size < 2 or len(trajectories) != group_size:
        raise ValueError("a complete group of at least two trajectories is required")
    batch = export_training_batch(
        [trajectories], group_size=group_size, epsilon=epsilon,
        allow_synthetic=allow_synthetic,
    )
    group = batch.groups[0]
    records = []
    for ir, unit in zip(trajectories, group.trajectories):
        ident = Identity(**ir["identity"])
        weight = token_weight(group_size, unit.token_count)
        for span in unit.turns:
            n = span.response_length
            records.append({
                "identity": ident.to_dict(), "turn_index": span.turn_index,
                "tokens": list(span.tokens), "prompt_length": span.prompt_length,
                "response_length": n, "loss_mask": [int(flag) for flag in span.loss_mask],
                "old_logprobs": list(span.old_logprobs), "advantages": [unit.advantage] * n,
                "token_weights": [weight] * n, "training_reward": ir["evaluation"]["training_reward"],
                "padding": False,
            })
    return {"export_version": "native-grpo-v2", "group_key": group.group_key,
            "trajectory_count": group_size, "mean": group.reward_mean, "std": group.reward_std,
            "advantages": [unit.advantage for unit in group.trajectories], "records": records,
            "synthetic": allow_synthetic,
            "reduction": "sum_token_weighted_loss; no secondary whitening"}
