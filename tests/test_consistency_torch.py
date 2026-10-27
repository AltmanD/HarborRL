"""Distributed rejection tests; optional Torch dependency."""
from pathlib import Path

import pytest

from harborrl.consistency.config import validate
from harborrl.consistency.report import assess, distributed_gate

torch = pytest.importorskip('torch')


def rank_failure(rank, rendezvous, root):
    from datetime import timedelta

    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        settings = validate({'mode': 'enforce', 'max_abs_delta': .01})
        part = assess([([-1.], [-1. if rank == 0 else float('nan')], [1])], settings)
        with pytest.raises(ValueError, match='rejected'):
            distributed_gate(part, settings, Path(root) / 'failure.json', phase='injected-rank-failure')
        (Path(root) / f'rank{rank}.ok').write_text('rejected')
    finally:
        dist.destroy_process_group()


def test_one_rank_error_rejects_all_ranks(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(rank_failure, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / 'rank0.ok').exists()
    assert (tmp_path / 'rank1.ok').exists()
