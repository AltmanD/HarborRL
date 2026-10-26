"""Delivery acknowledgements must precede native-session consumption audit."""
import threading

import pytest

from harborrl.gateway.trace import TraceRegistry


def test_delivery_barrier_waits_and_wakes(tmp_path):
    registry = TraceRegistry(tmp_path)
    registry.attempts["attempt"] = {"pending": {"request"}, "turns": []}
    entered = threading.Event()
    finished = threading.Event()

    def wait():
        entered.set()
        registry.wait_delivered("attempt", 2)
        finished.set()

    worker = threading.Thread(target=wait)
    worker.start()
    assert entered.wait(1)
    assert not finished.is_set()
    with registry.changed:
        registry.attempts["attempt"]["pending"].clear()
        registry.attempts["attempt"]["turns"].append({"delivery": "sent"})
        registry.changed.notify_all()
    worker.join(3)
    assert finished.is_set()
    assert not worker.is_alive()


def test_delivery_barrier_rejects_pending_and_unknown(tmp_path):
    registry = TraceRegistry(tmp_path)
    registry.attempts["attempt"] = {"pending": {"request"}, "turns": []}
    with pytest.raises(TimeoutError, match="pending"):
        registry.wait_delivered("attempt", 0)
    registry.attempts["attempt"]["pending"].clear()
    registry.attempts["attempt"]["turns"].append({"delivery": "delivery_unknown"})
    with pytest.raises(ValueError, match="not confirmed"):
        registry.wait_delivered("attempt", 0)
