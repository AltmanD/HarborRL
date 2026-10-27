"""A saturated image pool must not reserve unrelated global capacity."""
import asyncio
from contextlib import suppress

from harborrl.rollout.harbor_job.sandbox_config import SandboxAdmission


def test_waiting_binding_does_not_starve_other_environments(tmp_path):
    async def run():
        first = {'image': 'first', 'max_live_instances': 1}
        second = {'image': 'second', 'max_live_instances': 1}
        admission = SandboxAdmission({'max_live_instances': 2, 'bindings': [first, second]})
        entered, release = asyncio.Event(), asyncio.Event()

        async def waiting():
            entered.set()
            async with admission.lease(first, tmp_path / 'waiting'):
                await release.wait()

        async with admission.lease(first, tmp_path / 'active'):
            task = asyncio.create_task(waiting())
            try:
                await entered.wait()
                async def independent():
                    async with admission.lease(second, tmp_path / 'independent'):
                        pass
                await asyncio.wait_for(independent(), 1)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        # Cancellation of a queued request must not leak either admission permit.
        async with admission.lease(first, tmp_path / 'after'), admission.lease(second, tmp_path / 'also-after'):
            pass

    asyncio.run(run())
