"""Phase 3 separately sealed delegate child authority.

Contract v2 §9: every delegated child is a separately sealed node with its own
node_id, attempt_id, lease, fence, budget, and receipt chain; parent and child
bind through an immutable graph edge plus the parent's receipt/action id; a
child never inherits the parent's mutation capability or unspent budget; the
parent reserves the child's allocation before dispatch and reconciles on
return.  §11 G5 is the acceptance gate these tests stand in for.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import (
    AuthorityError,
    Phase2AuthorityStore,
    _canonical_hash,
)
from tools import delegate_tool

# The end-to-end delegate_task paths take real wall-clock time (production
# never injects a clock), so leases must be live *now* for those tests to
# exercise anything but lease_expired.
NOW = datetime.now(timezone.utc)
TTL_S = 3600
CHILD_GOAL = "inspect the child"
SECOND_GOAL = "inspect the sibling"


def _hex(seed: str) -> str:
    return (seed * 64)[:64]


def _node(node_id: str, *, surface: str, goal: str | None = None) -> dict:
    is_parent = node_id == "parent"
    return {
        "node_id": node_id,
        "objective": goal or f"execute {node_id}",
        "idempotency_key": _hex("a" if is_parent else "b"),
        "execution_surface": surface,
        "lane": "hermes" if is_parent else "claude_opus",
        "roots": ["/tmp"],
        "permissions": {
            "read": ["/tmp"],
            "write": [],
            "spawn": ["delegate_task:leaf"] if is_parent else [],
        },
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": (
            {"usd_max": 10.0, "tokens_max": 10000, "wall_clock_s_max": 600}
            if is_parent
            else {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60}
        ),
        "deadline_utc": "2030-01-01T00:00:00+00:00",
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["dispatch"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "input_hash": _hex("c" if is_parent else "d"),
    }


def _authority(tmp_path, *, goals: tuple[str, ...] = (CHILD_GOAL,)):
    """Seal a parent with one delegate edge per goal and grant every node."""

    store = Phase2AuthorityStore(tmp_path / "phase2_authority.db")
    child_ids = [f"child{i}" if i else "child" for i in range(len(goals))]
    plan = {
        "contract_version": 2,
        "graph_id": "g-delegate",
        "edges": [
            {
                "parent_node_id": "parent",
                "child_node_id": child_id,
                "dispatch_tool": "delegate_task",
            }
            for child_id in child_ids
        ],
        "nodes": [_node("parent", surface="local_tool")]
        + [
            _node(child_id, surface="a2a", goal=goal)
            for child_id, goal in zip(child_ids, goals)
        ],
    }
    store.seal_plan(plan, policy_hash=_hex("e"))
    parent = store.grant_node(
        "g-delegate", "parent", holder="hermes:parent", ttl_s=TTL_S,
        attempt_id="at-parent", now=NOW,
    )
    children = [
        store.grant_node(
            "g-delegate", child_id, holder=f"delegate:{child_id}", ttl_s=TTL_S,
            attempt_id=f"at-{child_id}", now=NOW,
        )
        for child_id in child_ids
    ]
    return store, parent, children[0] if len(children) == 1 else children, NOW


# --------------------------------------------------------------------------
# Graph-edge resolution and atomic reservation
# --------------------------------------------------------------------------


def test_prepare_child_dispatch_binds_exact_graph_edge_and_reserves_budget(tmp_path):
    store, parent, child, now = _authority(tmp_path)

    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child],
            store=store,
            now=now,
        )

    assert prepared[0].envelope == child
    assert prepared[0].parent_action_id == "g-delegate:parent:at-parent"
    assert prepared[0].child_action_id == "g-delegate:child:at-child"
    assert prepared[0].reservation_id.startswith("res-")
    usage = store.budget_usage("g-delegate", "parent", fence=1)
    assert usage["reserved_tokens"] == 1000
    assert usage["reserved_usd"] == 1.0


def test_prepare_child_dispatch_reserves_every_child_atomically(tmp_path):
    """Either the whole fan-out is reserved on the parent's fence, or none is."""

    store, parent, children, now = _authority(tmp_path, goals=(CHILD_GOAL, SECOND_GOAL))

    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"tasks": [{"goal": CHILD_GOAL}, {"goal": SECOND_GOAL}]},
            child_envelopes=list(children),
            store=store,
            now=now,
        )
        assert [item.child_action_id for item in prepared] == [
            "g-delegate:child:at-child",
            "g-delegate:child1:at-child1",
        ]
        assert len({item.reservation_id for item in prepared}) == 2
        usage = store.budget_usage("g-delegate", "parent", fence=1)
        assert usage["reserved_tokens"] == 2000

        # One unresolvable member rejects the whole batch — no partial reservation.
        with pytest.raises(AuthorityError):
            phase2_enforcement.prepare_delegate_child_authority(
                {"tasks": [{"goal": CHILD_GOAL}, {"goal": "not on any edge"}]},
                store=store,
                now=now,
            )
    assert store.budget_usage("g-delegate", "parent", fence=1)["reserved_tokens"] == 2000


def _regrant_parent_with_budgets(store, parent, budgets: dict) -> dict:
    """Append a chain-valid parent regrant carrying non-finite sealed ceilings.

    ``seal_plan`` refuses to write a non-finite ceiling into the immutable
    tables, so this reproduces the only way one can still be authoritative: a
    store whose grant predates that check. The event is appended exactly as the
    prior base wrote it, so the fan-out path meets a real legacy row rather than
    a monkeypatched loader.
    """

    envelope = json.loads(json.dumps(dict(parent)))
    envelope["budgets"] = budgets
    envelope["fence"] = 2
    envelope["attempt_id"] = "at-parent-legacy"
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        previous = conn.execute(
            "SELECT event_hash FROM authority_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = previous["event_hash"] if previous else None
        payload = {"envelope": envelope, "holder": envelope["lease"]["holder"]}
        row = {
            "event_id": "ev-legacy-grant-parent-2",
            "ts_utc": NOW.isoformat(),
            "kind": "LEASE_GRANTED",
            "graph_id": "g-delegate",
            "node_id": "parent",
            "attempt_id": envelope["attempt_id"],
            "fence": 2,
            "payload": payload,
            "prev_event_hash": prev_hash,
        }
        conn.execute(
            """INSERT INTO authority_events(
                   event_id, ts_utc, kind, graph_id, node_id, attempt_id, fence,
                   payload_json, prev_event_hash, event_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["event_id"],
                row["ts_utc"],
                row["kind"],
                row["graph_id"],
                row["node_id"],
                row["attempt_id"],
                row["fence"],
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                prev_hash,
                _canonical_hash(row),
            ),
        )
    return envelope


@pytest.mark.parametrize("ceiling", [float("inf"), float("nan")])
def test_fan_out_reservation_refuses_a_non_finite_sealed_ceiling(tmp_path, ceiling):
    """The batch path enforces the finite-ceiling bound exactly like the single path.

    A non-finite ``tokens_max`` is unbreachable (``nan > limit`` is always False)
    and ``int(inf)`` aborts accounting with an untyped ``OverflowError``, so both
    reservation paths must refuse it as a typed, auditable failure. Gating the
    fan-out any more weakly would make the delegation seam the cheapest way to
    spend against a ceiling that can never be breached — one call, N children.
    """

    store, parent, child, now = _authority(tmp_path)
    legacy_parent = _regrant_parent_with_budgets(
        store, parent, {"usd_max": 10.0, "tokens_max": ceiling, "wall_clock_s_max": 600}
    )

    with pytest.raises(AuthorityError, match="tokens_max must be a finite number"):
        store.reserve_budget_allocations(
            legacy_parent, [{"tokens": 1, "usd": 0.0}], now=now
        )
    # Parity: the single-reservation path refuses the identical authority.
    with pytest.raises(AuthorityError, match="tokens_max must be a finite number"):
        store.reserve_budget(legacy_parent, tokens=1, usd=0.0, now=now)
    # Fail-closed means nothing was charged or reserved against the legacy fence.
    usage = store.budget_usage("g-delegate", "parent", fence=2)
    assert usage["reserved_tokens"] == 0
    assert usage["charged_tokens"] == 0


def test_seal_plan_refuses_a_non_finite_delegate_ceiling(tmp_path):
    """A delegation graph can never enter the immutable tables unbreachable."""

    store = Phase2AuthorityStore(tmp_path / "phase2_authority.db")
    parent_node = _node("parent", surface="local_tool")
    parent_node["budgets"]["usd_max"] = float("inf")
    plan = {
        "contract_version": 2,
        "graph_id": "g-delegate",
        "edges": [
            {
                "parent_node_id": "parent",
                "child_node_id": "child",
                "dispatch_tool": "delegate_task",
            }
        ],
        "nodes": [parent_node, _node("child", surface="a2a", goal=CHILD_GOAL)],
    }

    with pytest.raises(AuthorityError, match="finite"):
        store.seal_plan(plan, policy_hash=_hex("e"))


def test_prepare_child_dispatch_rejects_missing_foreign_and_ambient_parent_authority(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    foreign = dict(child)
    foreign["graph_id"] = "g-foreign"

    with store.bind_current(parent, now=now):
        with pytest.raises(AuthorityError, match="child envelope"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": CHILD_GOAL, "role": "leaf"},
                child_envelopes=[], store=store, now=now,
            )
        with pytest.raises(AuthorityError, match="child envelope|graph"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": CHILD_GOAL, "role": "leaf"},
                child_envelopes=[foreign], store=store, now=now,
            )
        with pytest.raises(AuthorityError, match="child envelope|node"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": CHILD_GOAL, "role": "leaf"},
                child_envelopes=[parent], store=store, now=now,
            )

    # No ambient parent authority at all: nothing may be reserved or dispatched.
    with pytest.raises(AuthorityError, match="parent authority"):
        phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )


def test_prepare_child_dispatch_rejects_goal_drift_stale_fence_and_batch_shape(tmp_path):
    store, parent, child, now = _authority(tmp_path)

    with store.bind_current(parent, now=now):
        with pytest.raises(AuthorityError, match="objective|match"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": "different goal", "role": "leaf"},
                child_envelopes=[child], store=store, now=now,
            )
        with pytest.raises(AuthorityError, match="batch|count|match"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"tasks": [{"goal": CHILD_GOAL}, {"goal": "other"}]},
                child_envelopes=[child], store=store, now=now,
            )

    later = now + timedelta(seconds=1)
    store.revoke_node(child, now=later)
    current = store.grant_node(
        "g-delegate", "child", holder="delegate:new", ttl_s=TTL_S,
        attempt_id="at-child-2", now=later,
    )
    with store.bind_current(parent, now=later):
        with pytest.raises(AuthorityError, match="stale|invalid current"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": CHILD_GOAL, "role": "leaf"},
                child_envelopes=[child], store=store, now=later,
            )
        assert phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[current], store=store, now=later,
        )[0].envelope == current


# --------------------------------------------------------------------------
# Context isolation, reconciliation, and fail-closed completion
# --------------------------------------------------------------------------


def test_child_context_is_separate_and_reconcile_completes_exact_child(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]
        phase2_enforcement.record_budget_usage(tokens=17, usd=0.5)
        assert phase2_enforcement.current_sealed_envelope() == parent
        with phase2_enforcement.bind_delegate_child_authority(prepared, store=store, now=now):
            assert phase2_enforcement.current_sealed_envelope() == child
            assert phase2_enforcement.current_sealed_envelope() != parent
            # The child gets its own meter, not the parent's partially spent one.
            assert phase2_enforcement.current_budget_usage()["tokens"] == 0
            assert phase2_enforcement.current_authoritative_fence() == 1
        assert phase2_enforcement.current_sealed_envelope() == parent
        assert phase2_enforcement.current_budget_usage()["tokens"] == 17

    phase2_enforcement.reconcile_delegate_child_authority(
        prepared,
        {"status": "completed", "summary": "done"},
        actual_tokens=200,
        actual_usd=0.2,
        store=store,
        now=now + timedelta(seconds=1),
    )
    child_events = store.read_events("g-delegate")
    accepted = [e for e in child_events if e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"]
    assert len(accepted) == 1
    assert store.budget_usage("g-delegate", "parent", fence=1)["charged_tokens"] == 200


def test_reconcile_is_idempotent_for_a_redelivered_child_result(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]

    result = {"status": "completed", "summary": "done"}
    kwargs = dict(actual_tokens=200, actual_usd=0.2, store=store, now=now)
    first = phase2_enforcement.reconcile_delegate_child_authority(prepared, result, **kwargs)
    second = phase2_enforcement.reconcile_delegate_child_authority(prepared, result, **kwargs)

    assert first.get("idempotent") is not True
    assert second.get("idempotent") is True
    accepted = [
        e for e in store.read_events("g-delegate")
        if e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
    ]
    assert len(accepted) == 1
    assert store.budget_usage("g-delegate", "parent", fence=1)["charged_tokens"] == 200


def test_reconcile_rejects_stale_child_completion_and_poison_unknown_cost(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]
    store.revoke_node(child, now=now + timedelta(seconds=1))
    with pytest.raises(AuthorityError, match="revoked|stale"):
        phase2_enforcement.reconcile_delegate_child_authority(
            prepared, {"status": "completed"}, actual_tokens=1, actual_usd=None,
            store=store, now=now + timedelta(seconds=2),
        )
    assert store.budget_usage("g-delegate", "parent", fence=1)["cost_unknown"] is True


def test_close_revokes_child_and_settlement_never_upgrades_to_success(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]

    settlement = phase2_enforcement.DelegateChildSettlement({0: prepared}, store=store)
    closed = settlement.close(0, reason="timeout", actual_tokens=40, actual_usd=0.1)
    assert closed["outcome"] == "revoked"
    assert closed["child_action_id"] == "g-delegate:child:at-child"

    # A late delivery of the abandoned child's work cannot become an acceptance.
    late = settlement.accept(0, {"status": "completed"}, actual_tokens=40, actual_usd=0.1)
    assert late["outcome"] == "already_settled"
    assert late["prior_outcome"] == "revoked"
    events = store.read_events("g-delegate")
    assert any(e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child" for e in events)
    assert not any(e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child" for e in events)
    assert store.budget_usage("g-delegate", "parent", fence=1)["charged_tokens"] == 40


def test_late_completion_of_an_abandoned_child_is_downgraded_not_reported_green(tmp_path):
    """The worker loses the race to the parent's revocation and must not win."""

    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]

    settlement = phase2_enforcement.DelegateChildSettlement({0: prepared}, store=store)
    settlement.close(0, reason="parent_interrupted", actual_tokens=5, actual_usd=None)

    entry = delegate_tool._settle_child_entry(
        {"task_index": 0, "status": "completed", "summary": "done", "api_calls": 1},
        settlement,
        0,
        _StubChild(0),
    )
    assert entry["status"] == "failed"
    assert entry["exit_reason"] == "authority_rejected"
    assert entry["authority_outcome"] == "already_settled"
    assert not any(
        e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
        for e in store.read_events("g-delegate")
    )


def test_settlement_accept_of_a_stale_child_is_reported_not_swallowed(tmp_path):
    store, parent, child, now = _authority(tmp_path)
    with store.bind_current(parent, now=now):
        prepared = phase2_enforcement.prepare_delegate_child_authority(
            {"goal": CHILD_GOAL, "role": "leaf"},
            child_envelopes=[child], store=store, now=now,
        )[0]
    store.revoke_node(child, now=now + timedelta(seconds=1))

    settlement = phase2_enforcement.DelegateChildSettlement({0: prepared}, store=store)
    outcome = settlement.accept(
        0, {"status": "completed"}, actual_tokens=10, actual_usd=0.01,
        now=now + timedelta(seconds=2),
    )
    assert outcome["outcome"] == "rejected"
    assert outcome["detail"]
    assert not any(
        e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
        for e in store.read_events("g-delegate")
    )


# --------------------------------------------------------------------------
# End-to-end delegate_task dispatch
# --------------------------------------------------------------------------


class _StubChild:
    """Minimal child agent double that records the authority it ran under."""

    def __init__(self, index: int, *, summary: str = "done", interrupted: bool = False,
                 raises: BaseException | None = None, tokens=(60, 40),
                 cost: float | None = 0.05, gate: threading.Event | None = None,
                 started: threading.Event | None = None):
        self._summary = summary
        self._interrupted = interrupted
        self._raises = raises
        self._gate = gate
        self.model = "stub-model"
        self.session_id = f"child-{index}"
        self.session_prompt_tokens, self.session_completion_tokens = tokens
        self.session_estimated_cost_usd = cost
        self.session_reasoning_tokens = 0
        self.tool_progress_callback = None
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self._parent_subagent_id = None
        self._credential_pool = None
        self._interrupt_requested = False
        self.observed_envelope = None
        self.observed_fence = None
        self.started = started or threading.Event()
        self.finished = threading.Event()
        self.closed = False

    def get_activity_summary(self):
        return {
            "current_tool": None,
            "api_call_count": 1,
            "max_iterations": 5,
            "last_activity_desc": "",
        }

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        self.observed_envelope = phase2_enforcement.current_sealed_envelope()
        self.observed_fence = phase2_enforcement.current_authoritative_fence()
        self.started.set()
        try:
            if self._gate is not None:
                self._gate.wait(timeout=10)
            if self._raises is not None:
                raise self._raises
            return {
                "final_response": self._summary,
                "completed": not self._interrupted,
                "interrupted": self._interrupted,
                "api_calls": 1,
                "messages": [],
            }
        finally:
            self.finished.set()

    def interrupt(self, *_args):
        self._interrupt_requested = True

    def close(self):
        self.closed = True


class _StubParent:
    _delegate_depth = 0
    _interrupt_requested = False
    session_id = "parent-session"
    _current_turn_id = "turn-1"
    session_estimated_cost_usd = 0.0
    model = "parent-model"
    provider = "stub"
    base_url = None
    api_mode = None


def _wire_delegate_tool(monkeypatch, tmp_path, store, children, *, enforce=True):
    """Stub every non-authority dependency of delegate_task."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if enforce:
        monkeypatch.setenv("HERMES_ENVELOPE_ENFORCE", "1")
    else:
        monkeypatch.delenv("HERMES_ENVELOPE_ENFORCE", raising=False)
    monkeypatch.setattr(
        phase2_enforcement, "_authority_store", lambda supplied=None: supplied or store
    )
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_route_config", lambda cfg, name: {}
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda route_cfg, parent: {
            "model": "stub-model",
            "provider": "stub",
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        },
    )

    from tools import delegation_live_log

    monkeypatch.setattr(
        delegation_live_log,
        "create_live_transcripts",
        lambda task_list, context: ("deleg-test", [None] * len(task_list), []),
    )
    monkeypatch.setattr(
        delegation_live_log, "update_manifest_statuses", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        delegate_tool, "_build_child_agent", lambda **kw: children[kw["task_index"]]
    )
    return _StubParent()


def test_synchronous_single_child_runs_under_its_own_sealed_authority(monkeypatch, tmp_path):
    store, parent, child, now = _authority(tmp_path)
    stub = _StubChild(0)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(goal=CHILD_GOAL, parent_agent=parent_agent)
        )

    entry = payload["results"][0]
    assert entry["status"] == "completed"
    # Receipt linkage: the entry names the exact parent and child action ids.
    assert entry["parent_action_id"] == "g-delegate:parent:at-parent"
    assert entry["child_action_id"] == "g-delegate:child:at-child"
    assert entry["authority_outcome"] == "accepted"
    # Context isolation: the child ran under its own envelope, never the parent's.
    assert stub.observed_envelope == child
    assert stub.observed_envelope != parent
    assert stub.observed_fence == 1

    events = store.read_events("g-delegate")
    accepted = [e for e in events if e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"]
    assert len(accepted) == 1
    usage = store.budget_usage("g-delegate", "parent", fence=1)
    assert usage["charged_tokens"] == 100
    assert usage["charged_usd"] == pytest.approx(0.05)
    assert usage["reserved_tokens"] == 0
    assert usage["cost_unknown"] is False


def test_synchronous_batch_settles_each_child_against_its_own_node(monkeypatch, tmp_path):
    store, parent, children, now = _authority(tmp_path, goals=(CHILD_GOAL, SECOND_GOAL))
    gate = threading.Event()
    stubs = [
        _StubChild(0, gate=gate, summary="first"),
        _StubChild(1, gate=gate, summary="second"),
    ]
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, stubs)

    def _release():
        # Both children must be inside run_conversation before either returns,
        # so the two sealed contexts are provably live at the same time.
        for stub in stubs:
            assert stub.started.wait(timeout=10)
        gate.set()

    releaser = threading.Thread(target=_release, daemon=True)
    releaser.start()
    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(
                tasks=[{"goal": CHILD_GOAL}, {"goal": SECOND_GOAL}],
                parent_agent=parent_agent,
            )
        )
    releaser.join(timeout=10)

    assert [e["status"] for e in payload["results"]] == ["completed", "completed"]
    assert [e["child_action_id"] for e in payload["results"]] == [
        "g-delegate:child:at-child",
        "g-delegate:child1:at-child1",
    ]
    # Concurrent children each saw their own node, never a shared one.
    assert stubs[0].observed_envelope["node_id"] == "child"
    assert stubs[1].observed_envelope["node_id"] == "child1"
    events = store.read_events("g-delegate")
    for node_id in ("child", "child1"):
        assert len(
            [e for e in events if e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == node_id]
        ) == 1
    usage = store.budget_usage("g-delegate", "parent", fence=1)
    assert usage["charged_tokens"] == 200
    assert usage["reserved_tokens"] == 0


def test_failed_child_is_revoked_without_acceptance(monkeypatch, tmp_path):
    store, parent, child, now = _authority(tmp_path)
    stub = _StubChild(0, raises=RuntimeError("provider exploded"))
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(goal=CHILD_GOAL, parent_agent=parent_agent)
        )

    entry = payload["results"][0]
    assert entry["status"] in {"error", "failed"}
    assert entry["authority_outcome"] == "revoked"
    events = store.read_events("g-delegate")
    assert any(e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child" for e in events)
    assert not any(e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child" for e in events)


def test_interrupted_child_is_revoked_without_acceptance(monkeypatch, tmp_path):
    store, parent, child, now = _authority(tmp_path)
    stub = _StubChild(0, summary="", interrupted=True)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(goal=CHILD_GOAL, parent_agent=parent_agent)
        )

    entry = payload["results"][0]
    assert entry["status"] == "interrupted"
    assert entry["authority_outcome"] == "revoked"
    assert not any(
        e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
        for e in store.read_events("g-delegate")
    )
    # Spend the interrupted child may have incurred is unaccountable → fail closed.
    assert store.budget_usage("g-delegate", "parent", fence=1)["cost_unknown"] is False


def test_parent_interrupt_revokes_the_child_it_abandons(monkeypatch, tmp_path):
    """The abandoned worker keeps running; its lease must already be closed."""

    store, parent, _children, now = _authority(tmp_path, goals=(CHILD_GOAL, SECOND_GOAL))
    gate = threading.Event()
    stubs = [_StubChild(0, summary="first"), _StubChild(1, gate=gate, summary="second")]
    _wire_delegate_tool(monkeypatch, tmp_path, store, stubs)

    trigger = threading.Event()

    class _InterruptingParent(_StubParent):
        @property
        def _interrupt_requested(self):
            return trigger.is_set()

    parent_agent = _InterruptingParent()

    def _revoked_child1():
        return any(
            e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child1"
            for e in store.read_events("g-delegate")
        )

    def _accepted_child1():
        return any(
            e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child1"
            for e in store.read_events("g-delegate")
        )

    def _arm():
        # Interrupt only after the second child is provably in-flight, so the
        # poll loop abandons a *pending* future rather than a finished one.
        assert stubs[0].finished.wait(timeout=10)
        assert stubs[1].started.wait(timeout=10)
        trigger.set()
        # Release the abandoned worker only once its lease is durably revoked:
        # that orders "parent gave up" strictly before "child produced work",
        # which is exactly the sequence the contract has to survive.  (The
        # batch executor still joins abandoned workers on scope exit, so
        # holding the gate any longer would just stall the parent.)
        for _ in range(1000):
            if _revoked_child1():
                break
            time.sleep(0.01)
        gate.set()

    armer = threading.Thread(target=_arm, daemon=True)
    armer.start()
    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(
                tasks=[{"goal": CHILD_GOAL}, {"goal": SECOND_GOAL}],
                parent_agent=parent_agent,
            )
        )
    armer.join(timeout=10)

    abandoned = [e for e in payload["results"] if e["task_index"] == 1][0]
    assert abandoned["status"] == "interrupted"
    assert abandoned["child_action_id"] == "g-delegate:child1:at-child1"
    assert abandoned["authority_outcome"] == "revoked"

    assert _revoked_child1()
    # The abandoned worker ran to completion after the revocation and still
    # never landed an acceptance.
    assert stubs[1].finished.wait(timeout=10)
    for _ in range(100):
        if _accepted_child1():
            break
        time.sleep(0.02)
    assert not _accepted_child1()
    # Every child settled exactly once, whichever way the race resolved.
    final = store.read_events("g-delegate")
    for node_id in ("child", "child1"):
        terminal = [
            e for e in final
            if e["node_id"] == node_id and e["kind"] in {"RESULT_ACCEPTED", "LEASE_REVOKED"}
        ]
        assert len(terminal) == 1, (node_id, [e["kind"] for e in terminal])


def test_unknown_child_cost_poisons_the_parent_budget(monkeypatch, tmp_path):
    store, parent, _children, now = _authority(tmp_path, goals=(CHILD_GOAL, SECOND_GOAL))
    stub = _StubChild(0, cost=None)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(goal=CHILD_GOAL, parent_agent=parent_agent)
        )

    assert payload["results"][0]["authority_outcome"] == "accepted"
    usage = store.budget_usage("g-delegate", "parent", fence=1)
    assert usage["charged_tokens"] == 100
    assert usage["charged_usd"] == 0.0
    assert usage["cost_unknown"] is True
    # A fence that absorbed an unaccountable spend cannot fund another child.
    with store.bind_current(parent, now=now):
        with pytest.raises(AuthorityError, match="cost is unknown"):
            phase2_enforcement.prepare_delegate_child_authority(
                {"goal": SECOND_GOAL, "role": "leaf"}, store=store, now=now,
            )
    reserved_for = [
        (e["payload"].get("metadata") or {}).get("child_action_id")
        for e in store.read_events("g-delegate")
        if e["kind"] == "BUDGET_RESERVED"
    ]
    assert reserved_for == ["g-delegate:child:at-child"]


def test_delegation_aborted_after_reservation_closes_every_child(monkeypatch, tmp_path):
    """An early exit between reservation and dispatch must not leak a reservation."""

    store, parent, child, now = _authority(tmp_path)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [])

    def _explode(**_kwargs):
        raise RuntimeError("child build failed")

    monkeypatch.setattr(delegate_tool, "_build_child_agent", _explode)

    with store.bind_current(parent, now=now):
        with pytest.raises(RuntimeError, match="child build failed"):
            delegate_tool.delegate_task(goal=CHILD_GOAL, parent_agent=parent_agent)

    events = store.read_events("g-delegate")
    assert any(e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child" for e in events)
    assert store.budget_usage("g-delegate", "parent", fence=1)["reserved_tokens"] == 0


# --------------------------------------------------------------------------
# Asynchronous (background) dispatch
# --------------------------------------------------------------------------


def _capture_async_dispatch(monkeypatch):
    captured: dict = {}

    def _dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": "deleg-test"}

    from tools import async_delegation

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", _dispatch)
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    return captured


def test_async_dispatch_carries_sealed_identity_into_background_finalization(
    monkeypatch, tmp_path
):
    store, parent, child, now = _authority(tmp_path)
    stub = _StubChild(0)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])
    captured = _capture_async_dispatch(monkeypatch)

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(
                goal=CHILD_GOAL, background=True, parent_agent=parent_agent
            )
        )

    assert payload["status"] == "dispatched"
    # The parent turn has returned; the child is still live and unsettled.
    events = store.read_events("g-delegate")
    assert not any(e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child" for e in events)
    assert store.budget_usage("g-delegate", "parent", fence=1)["reserved_tokens"] == 1000

    combined = captured["runner"]()

    entry = combined["results"][0]
    assert entry["status"] == "completed"
    assert entry["child_action_id"] == "g-delegate:child:at-child"
    assert entry["parent_action_id"] == "g-delegate:parent:at-parent"
    assert stub.observed_envelope == child
    accepted = [
        e for e in store.read_events("g-delegate")
        if e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
    ]
    assert len(accepted) == 1
    assert store.budget_usage("g-delegate", "parent", fence=1)["reserved_tokens"] == 0


def test_async_cancellation_revokes_children_before_any_late_acceptance(
    monkeypatch, tmp_path
):
    store, parent, child, now = _authority(tmp_path)
    gate = threading.Event()
    stub = _StubChild(0, gate=gate)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])
    captured = _capture_async_dispatch(monkeypatch)

    with store.bind_current(parent, now=now):
        delegate_tool.delegate_task(
            goal=CHILD_GOAL, background=True, parent_agent=parent_agent
        )

    runner = threading.Thread(target=captured["runner"], daemon=True)
    runner.start()
    assert stub.started.wait(timeout=10)

    captured["interrupt_fn"]()
    events = store.read_events("g-delegate")
    assert any(e["kind"] == "LEASE_REVOKED" and e["node_id"] == "child" for e in events)

    gate.set()
    runner.join(timeout=10)
    # The child finished after cancellation — it must never have been accepted.
    assert not any(
        e["kind"] == "RESULT_ACCEPTED" and e["node_id"] == "child"
        for e in store.read_events("g-delegate")
    )


# --------------------------------------------------------------------------
# Model-visible surface, dispatch-path parity, and flags-off parity
# --------------------------------------------------------------------------


def test_delegate_tool_flags_off_parity_does_not_touch_authority(monkeypatch, tmp_path):
    """Flags off: identical control flow, and not one byte of authority state."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ENVELOPE_ENFORCE", raising=False)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    called = {"build": 0}

    def _build(**kwargs):
        called["build"] += 1
        raise RuntimeError("stop after parity seam")

    def _no_store(supplied=None):
        raise AssertionError("authority store must not be opened with flags off")

    monkeypatch.setattr(phase2_enforcement, "_authority_store", _no_store)
    monkeypatch.setattr(delegate_tool, "_build_child_agent", _build)
    parent = type("Parent", (), {"_delegate_depth": 0})()
    db = tmp_path / "phase2_authority.db"

    assert phase2_enforcement.is_enabled() is False
    with pytest.raises(RuntimeError, match="stop after parity seam"):
        delegate_tool.delegate_task(goal="x", parent_agent=parent)

    # The child was still built and the exception still escaped unchanged:
    # the enforcement seam is inert, not an extra failure mode.
    assert called["build"] == 1
    assert not db.exists()


def test_delegate_tool_rejects_enforced_dispatch_without_child_envelopes(monkeypatch, tmp_path):
    store, parent, _child, now = _authority(tmp_path)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [])

    with store.bind_current(parent, now=now):
        result = json.loads(
            delegate_tool.delegate_task(goal="not on any edge", parent_agent=parent_agent)
        )

    assert result["status"] == "blocked"
    assert result["code"] == "missing_child_authority"
    assert store.budget_usage("g-delegate", "parent", fence=1)["reserved_tokens"] == 0


def test_delegate_tool_internal_child_authority_fields_are_not_model_visible():
    hidden = delegate_tool._strip_model_hidden_task_fields(
        [{
            "goal": "x",
            "child_envelope": {"secret": "authority"},
            "parent_action_id": "p",
            "budget_reservation_id": "r",
        }]
    )
    assert hidden == [{"goal": "x"}]


def test_model_supplied_authority_fields_cannot_select_child_authority(monkeypatch, tmp_path):
    """Authority-shaped arguments are dropped, not honoured, even off-registry."""

    store, parent, child, now = _authority(tmp_path)
    stub = _StubChild(0)
    parent_agent = _wire_delegate_tool(monkeypatch, tmp_path, store, [stub])
    forged = dict(child)
    forged["budgets"] = {"usd_max": 999.0, "tokens_max": 999999, "wall_clock_s_max": 60}

    with store.bind_current(parent, now=now):
        payload = json.loads(
            delegate_tool.delegate_task(
                tasks=[{
                    "goal": CHILD_GOAL,
                    "child_envelope": forged,
                    "parent_action_id": "g-forged:parent:at-forged",
                    "budget_reservation_id": "res-forged",
                }],
                parent_agent=parent_agent,
            )
        )

    entry = payload["results"][0]
    assert entry["parent_action_id"] == "g-delegate:parent:at-parent"
    assert entry["child_action_id"] == "g-delegate:child:at-child"
    # The sealed child budget governed the reservation, not the forged one.
    assert stub.observed_envelope["budgets"]["tokens_max"] == 1000


def test_tool_executor_and_run_agent_delegate_paths_enforce_equivalently(monkeypatch):
    """Both model-facing dispatch seams reach one enforced delegate_task.

    ``agent.tool_executor`` (concurrent/segmented) and
    ``agent.agent_runtime_helpers`` (sequential/inline) both funnel into
    ``AIAgent._dispatch_delegate_task``; the registry handler is the fallback
    for a bypassed intercept.  Enforcement therefore has to be identical across
    exactly two kwarg constructors, which is what this asserts.
    """

    import inspect

    from agent import agent_runtime_helpers, tool_executor
    from run_agent import AIAgent
    from tools.registry import registry

    for module in (tool_executor, agent_runtime_helpers):
        assert "_dispatch_delegate_task" in inspect.getsource(module), (
            f"{module.__name__} no longer funnels delegation through "
            "_dispatch_delegate_task — enforcement parity is not guaranteed"
        )

    seen: list[dict] = []

    def _recording_delegate_task(**kwargs):
        seen.append(kwargs)
        return "{}"

    monkeypatch.setattr(delegate_tool, "delegate_task", _recording_delegate_task)

    args = {
        "goal": None,
        "tasks": [{"goal": "g", "child_envelope": {"x": 1}, "parent_action_id": "p"}],
        "role": "leaf",
    }
    parent = type("Parent", (), {"_delegate_depth": 0})()

    registry.dispatch("delegate_task", dict(args), parent_agent=parent)
    AIAgent._dispatch_delegate_task(parent, dict(args))

    assert len(seen) == 2
    for call in seen:
        assert call["tasks"] == [{"goal": "g"}]
        assert call["parent_agent"] is parent
        assert call["background"] is True
    assert seen[0]["goal"] == seen[1]["goal"]
    assert seen[0]["role"] == seen[1]["role"]
