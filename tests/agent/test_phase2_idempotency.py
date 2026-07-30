"""Atomic idempotency claims for Phase 2 mutating tool calls."""

from __future__ import annotations

import threading

from agent import phase2_idempotency


def test_first_mutation_claim_wins_and_duplicate_is_denied(tmp_path):
    store = phase2_idempotency.MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "d" * 64,
        "fence": 1,
    }

    assert store.try_claim(envelope, tool_name="write_file", args={"path": "one"}) is True
    assert store.try_claim(envelope, tool_name="write_file", args={"path": "one"}) is False


def test_mutation_claim_is_atomic_under_concurrency(tmp_path):
    store = phase2_idempotency.MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "e" * 64,
        "fence": 4,
    }
    barrier = threading.Barrier(8)
    outcomes: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        outcome = store.try_claim(envelope, tool_name="patch", args={"path": "one"})
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7


def test_claim_persists_canonical_identity_without_raw_arguments(tmp_path):
    store = phase2_idempotency.MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-secret",
        "idempotency_key": "f" * 64,
        "fence": 2,
    }
    store.try_claim(
        envelope,
        tool_name="write_file",
        args={"path": "/tmp/out", "content": "do-not-store-this"},
    )

    claims = store.read_all()
    assert claims[0]["graph_id"] == "g-1"
    assert claims[0]["node_id"] == "n-1"
    assert claims[0]["attempt_id"] == "a-secret"
    assert claims[0]["tool_name"] == "write_file"
    assert claims[0]["fence"] == 2
    assert len(claims[0]["args_hash"]) == 64
    assert "do-not-store-this" not in repr(claims)
