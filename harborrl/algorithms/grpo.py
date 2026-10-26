"""GRPO semantics backed by the existing CPU-verified implementations."""
from __future__ import annotations

from harborrl.backends.slime_v032.advantage import compute_advantages_and_returns
from harborrl.backends.slime_v032.loss import clipped_loss, loss_function


class GRPOAlgorithmAdapter:
    algorithm_id = "grpo"

    def compute_advantages_and_returns(self, args, rollout_data):
        return compute_advantages_and_returns(args, rollout_data)

    def clipped_loss(self, *args, **kwargs):
        return clipped_loss(*args, **kwargs)

    def backend_loss_function(self, *args, **kwargs):
        return loss_function(*args, **kwargs)
