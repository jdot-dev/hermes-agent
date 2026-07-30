"""Durable Hermes-owned authority for sealed Phase 2 graph nodes.

The store is deliberately append-only at the lifecycle layer. Plans and node
specifications are immutable after sealing; lease/fence and budget state are
derived from hash-chained events written under ``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from agent.phase2_enforcement import bind_sealed_envelope, validate_sealed_envelope
from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()
_HASH_KEYS = ("idempotency_key", "input_hash")
_ALLOWED_SURFACES = {"direct_model", "combo", "acp_worker", "a2a", "cloud", "local_tool"}
_SHA256_RE_LEN = 64

# Lifecycle event kinds. LEASE_GRANTED carries the sole sealed envelope, which
# stays immutable forever. LEASE_RENEWED carries bounded external metadata only
# (never an envelope); effective expiry is derived from grant + latest renewal.
_EVENT_KINDS = (
    "PLAN_SEALED",
    "LEASE_GRANTED",
    "LEASE_RENEWED",
    "LEASE_REVOKED",
    "RESULT_ACCEPTED",
    "RESULT_REJECTED",
    "NODE_COMPLETED",  # legacy alias folded identically to RESULT_ACCEPTED
    "BUDGET_RESERVED",
    "BUDGET_RECONCILED",
)
_ACCEPTED_KINDS = ("RESULT_ACCEPTED", "NODE_COMPLETED")


class AuthorityError(RuntimeError):
    """A requested authority transition is invalid or unavailable."""


class ResultRejection(AuthorityError):
    """A completion was rejected; exactly one RESULT_REJECTED event was appended.

    The rejection is recorded durably in the same decision transaction, so this
    exception carries the bounded machine-readable ``reason`` and the
    ``result_hash`` that was refused.
    """

    def __init__(self, reason: str, *, result_hash: str | None = None) -> None:
        super().__init__(f"result rejected: {reason}")
        self.reason = reason
        self.result_hash = result_hash


class _Reject(Exception):
    """Internal control-flow signal: one typed rejection must be recorded."""

    def __init__(self, reason: str, *, fence_hint: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fence_hint = fence_hint


def default_db_path() -> Path:
    return get_hermes_home() / "phase2_authority.db"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise AuthorityError("authority timestamps must be timezone-aware")
    return current.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuthorityError("invalid authority timestamp") from exc
    if parsed.tzinfo is None:
        raise AuthorityError("authority timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise AuthorityError(f"{field} must be a non-negative number")
    return float(value)


def _valid_sha256(value: Any, field: str) -> str:
    """Return a validated lowercase SHA-256 hex digest or fail closed."""

    if not isinstance(value, str) or len(value) != _SHA256_RE_LEN:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest") from exc
    if value.lower() != value:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest")
    return value


class Phase2AuthorityStore:
    """Seal plans and issue node-scoped fences, leases, and budget reservations."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        self._ensure_schema(conn)
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sealed_plans (
                graph_id TEXT PRIMARY KEY,
                contract_version INTEGER NOT NULL,
                planner_hash TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                sealed_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sealed_nodes (
                graph_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_json TEXT NOT NULL,
                PRIMARY KEY(graph_id, node_id),
                FOREIGN KEY(graph_id) REFERENCES sealed_plans(graph_id)
            );
            CREATE TABLE IF NOT EXISTS authority_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                ts_utc TEXT NOT NULL,
                kind TEXT NOT NULL,
                graph_id TEXT NOT NULL,
                node_id TEXT,
                attempt_id TEXT,
                fence INTEGER,
                payload_json TEXT NOT NULL,
                prev_event_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS sealed_plans_no_update
            BEFORE UPDATE ON sealed_plans BEGIN
                SELECT RAISE(ABORT, 'sealed_plans is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_plans_no_delete
            BEFORE DELETE ON sealed_plans BEGIN
                SELECT RAISE(ABORT, 'sealed_plans is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_nodes_no_update
            BEFORE UPDATE ON sealed_nodes BEGIN
                SELECT RAISE(ABORT, 'sealed_nodes is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_nodes_no_delete
            BEFORE DELETE ON sealed_nodes BEGIN
                SELECT RAISE(ABORT, 'sealed_nodes is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS authority_events_no_update
            BEFORE UPDATE ON authority_events BEGIN
                SELECT RAISE(ABORT, 'authority_events is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS authority_events_no_delete
            BEFORE DELETE ON authority_events BEGIN
                SELECT RAISE(ABORT, 'authority_events is append-only');
            END;
            CREATE UNIQUE INDEX IF NOT EXISTS one_accepted_result_per_node
            ON authority_events(graph_id, node_id)
            WHERE kind = 'RESULT_ACCEPTED';
            CREATE UNIQUE INDEX IF NOT EXISTS one_terminal_result_per_node
            ON authority_events(graph_id, node_id)
            WHERE kind IN ('RESULT_ACCEPTED', 'NODE_COMPLETED');
            CREATE UNIQUE INDEX IF NOT EXISTS one_grant_per_attempt
            ON authority_events(attempt_id)
            WHERE kind = 'LEASE_GRANTED';
            """
        )

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        kind: str,
        graph_id: str,
        node_id: str | None,
        attempt_id: str | None,
        fence: int | None,
        payload: Mapping[str, Any],
        now: datetime,
        event_id: str | None = None,
    ) -> str:
        event_id = event_id or f"ev-{uuid.uuid4().hex}"
        previous = conn.execute(
            "SELECT event_hash FROM authority_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = previous["event_hash"] if previous else None
        row = {
            "event_id": event_id,
            "ts_utc": _iso(now),
            "kind": kind,
            "graph_id": graph_id,
            "node_id": node_id,
            "attempt_id": attempt_id,
            "fence": fence,
            "payload": dict(payload),
            "prev_event_hash": prev_hash,
        }
        event_hash = _canonical_hash(row)
        conn.execute(
            """INSERT INTO authority_events(
                   event_id, ts_utc, kind, graph_id, node_id, attempt_id, fence,
                   payload_json, prev_event_hash, event_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                row["ts_utc"],
                kind,
                graph_id,
                node_id,
                attempt_id,
                fence,
                _canonical_json(payload),
                prev_hash,
                event_hash,
            ),
        )
        return event_id

    @staticmethod
    def _validate_plan(plan: Mapping[str, Any], policy_hash: str) -> tuple[str, list[dict[str, Any]]]:
        if plan.get("contract_version") != 2:
            raise AuthorityError("contract_version must be 2")
        graph_id = plan.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise AuthorityError("graph_id is required")
        if not isinstance(policy_hash, str) or len(policy_hash) != 64:
            raise AuthorityError("policy_hash must be a SHA-256 hex digest")
        try:
            int(policy_hash, 16)
        except ValueError as exc:
            raise AuthorityError("policy_hash must be a SHA-256 hex digest") from exc
        nodes = plan.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise AuthorityError("plan nodes must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in nodes:
            if not isinstance(raw, Mapping):
                raise AuthorityError("each plan node must be an object")
            node = json.loads(_canonical_json(raw))
            node_id = node.get("node_id")
            if not isinstance(node_id, str) or not node_id.strip():
                raise AuthorityError("node_id is required")
            if node_id in seen:
                raise AuthorityError(f"duplicate node_id: {node_id}")
            seen.add(node_id)
            surface = node.get("execution_surface")
            if surface not in _ALLOWED_SURFACES:
                raise AuthorityError(f"invalid execution_surface for {node_id}")
            if any(node.get(key) is None for key in _HASH_KEYS):
                raise AuthorityError(f"missing sealed node hash for {node_id}")
            normalized.append(node)
        return graph_id, normalized

    def seal_plan(
        self,
        plan: Mapping[str, Any],
        *,
        policy_hash: str,
        now: datetime | None = None,
    ) -> str:
        """Persist an immutable plan and its nodes, returning the planner hash."""

        graph_id, nodes = self._validate_plan(plan, policy_hash)
        sealed_at = _utc(now)
        canonical_plan = json.loads(_canonical_json(plan))
        planner_hash = _canonical_hash(canonical_plan)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute(
                    "SELECT 1 FROM sealed_plans WHERE graph_id = ?", (graph_id,)
                ).fetchone():
                    raise AuthorityError(f"graph {graph_id} is already sealed")
                conn.execute(
                    """INSERT INTO sealed_plans(
                           graph_id, contract_version, planner_hash, policy_hash,
                           plan_json, sealed_utc
                       ) VALUES (?, 2, ?, ?, ?, ?)""",
                    (graph_id, planner_hash, policy_hash, _canonical_json(canonical_plan), _iso(sealed_at)),
                )
                conn.executemany(
                    "INSERT INTO sealed_nodes(graph_id, node_id, node_json) VALUES (?, ?, ?)",
                    [(graph_id, node["node_id"], _canonical_json(node)) for node in nodes],
                )
                self._append_event(
                    conn,
                    kind="PLAN_SEALED",
                    graph_id=graph_id,
                    node_id=None,
                    attempt_id=None,
                    fence=None,
                    payload={"planner_hash": planner_hash, "policy_hash": policy_hash},
                    now=sealed_at,
                )
                conn.execute("COMMIT")
                return planner_hash
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    @staticmethod
    def _load_plan_node(
        conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> tuple[sqlite3.Row, dict[str, Any]]:
        plan = conn.execute(
            "SELECT * FROM sealed_plans WHERE graph_id = ?", (graph_id,)
        ).fetchone()
        row = conn.execute(
            "SELECT node_json FROM sealed_nodes WHERE graph_id = ? AND node_id = ?",
            (graph_id, node_id),
        ).fetchone()
        if plan is None or row is None:
            raise AuthorityError("unknown sealed graph node")
        return plan, json.loads(row["node_json"])

    @staticmethod
    def _latest_grant(
        conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM authority_events
               WHERE graph_id = ? AND node_id = ? AND kind = 'LEASE_GRANTED'
               ORDER BY id DESC LIMIT 1""",
            (graph_id, node_id),
        ).fetchone()

    @classmethod
    def _node_authority(
        cls, conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> dict[str, Any] | None:
        """Fold the current fence's lifecycle events into one authority view.

        The sealed ``LEASE_GRANTED`` envelope is immutable: it is returned
        unchanged regardless of renewals. The current *effective* expiry is
        derived externally from the grant plus the latest valid ``LEASE_RENEWED``
        event, never by rewriting envelope bytes. A superseded fence is not
        represented here — only the newest ``LEASE_GRANTED`` establishes the
        current fence, and any renewal, revocation, or completion is scoped to
        that fence.
        """

        grant = cls._latest_grant(conn, graph_id, node_id)
        if grant is None:
            return None
        fence = int(grant["fence"])
        envelope = json.loads(grant["payload_json"])["envelope"]
        rows = conn.execute(
            """SELECT kind, payload_json FROM authority_events
               WHERE graph_id = ? AND node_id = ? AND fence = ?
                 AND kind IN ('LEASE_RENEWED', 'LEASE_REVOKED',
                              'RESULT_ACCEPTED', 'NODE_COMPLETED')
               ORDER BY id""",
            (graph_id, node_id, fence),
        ).fetchall()
        revoked = False
        completed = False
        accepted_result_hash: str | None = None
        new_expires_utc: str | None = None
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["kind"] == "LEASE_RENEWED":
                # Bounded external metadata only; never an envelope copy.
                new_expires_utc = payload["new_expires_utc"]
            elif row["kind"] == "LEASE_REVOKED":
                revoked = True
            elif row["kind"] in _ACCEPTED_KINDS:
                completed = True
                accepted_result_hash = payload.get("result_hash")
        base_expiry = cls._lease_expiry(envelope)
        effective_expiry = (
            _parse_time(new_expires_utc) if new_expires_utc is not None else base_expiry
        )
        return {
            "fence": fence,
            "envelope": envelope,
            "granted_expires": base_expiry,
            "effective_expires": effective_expiry,
            "revoked": revoked,
            "completed": completed,
            "accepted_result_hash": accepted_result_hash,
        }

    @staticmethod
    def _lease_expiry(envelope: Mapping[str, Any]) -> datetime:
        lease = envelope["lease"]
        return _parse_time(lease["granted_utc"]) + timedelta(seconds=float(lease["ttl_s"]))

    @staticmethod
    def _terminal_completed(conn: sqlite3.Connection, graph_id: str, node_id: str) -> bool:
        """True if the node has any terminal acceptance across ALL fences."""

        return (
            conn.execute(
                """SELECT 1 FROM authority_events
                   WHERE graph_id = ? AND node_id = ?
                     AND kind IN ('RESULT_ACCEPTED', 'NODE_COMPLETED')
                   LIMIT 1""",
                (graph_id, node_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _attempt_seen(conn: sqlite3.Connection, attempt_id: str) -> bool:
        """True if the attempt_id was ever granted anywhere in this authority store.

        ``attempt_id`` identifies one execution attempt, not one node-local
        sequence slot. Contract v2 §4 requires it to be new per attempt and
        never reused, including on another graph or node.
        """

        return (
            conn.execute(
                """SELECT 1 FROM authority_events
                   WHERE kind = 'LEASE_GRANTED' AND attempt_id = ?
                   LIMIT 1""",
                (attempt_id,),
            ).fetchone()
            is not None
        )

    def grant_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        holder: str,
        ttl_s: float,
        attempt_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically grant the next monotonic fence and return its sealed envelope.

        A node closed by terminal completion can never be regranted. Attempt IDs
        are single-use across the whole graph/node history and are never reused,
        including after revocation or natural expiry.
        """

        granted_at = _utc(now)
        if not isinstance(holder, str) or not holder.strip():
            raise AuthorityError("lease holder is required")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise AuthorityError("attempt_id is required")
        ttl = _nonnegative_number(ttl_s, "ttl_s")
        if ttl == 0:
            raise AuthorityError("ttl_s must be greater than zero")
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                plan, node = self._load_plan_node(conn, graph_id, node_id)
                if self._terminal_completed(conn, graph_id, node_id):
                    raise AuthorityError("node is terminal and cannot be regranted")
                if self._attempt_seen(conn, attempt_id):
                    raise AuthorityError(f"attempt_id {attempt_id} was already used")
                authority = self._node_authority(conn, graph_id, node_id)
                if authority is not None:
                    if not authority["revoked"] and not authority["completed"]:
                        if granted_at < authority["effective_expires"]:
                            raise AuthorityError("node already has an active lease")
                    fence = authority["fence"] + 1
                else:
                    fence = 1
                envelope = {
                    "envelope_version": 2,
                    "graph_id": graph_id,
                    **node,
                    "attempt_id": attempt_id,
                    "planner_hash": plan["planner_hash"],
                    "policy_hash": plan["policy_hash"],
                    "lease": {
                        "holder": holder,
                        "granted_utc": _iso(granted_at),
                        "ttl_s": ttl,
                        "renewable": True,
                    },
                    "fence": fence,
                }
                errors = validate_sealed_envelope(envelope)
                if errors:
                    raise AuthorityError("invalid sealed node envelope: " + ", ".join(errors))
                self._append_event(
                    conn,
                    kind="LEASE_GRANTED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=attempt_id,
                    fence=fence,
                    payload={"envelope": envelope},
                    now=granted_at,
                )
                conn.execute("COMMIT")
                return envelope
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def _authority_gate(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        *,
        action: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Shared exact-authority gate binding holder and canonical envelope identity.

        Returns ``(graph_id, node_id, authority)`` for the current fence when the
        supplied envelope IS the byte-identical authoritative sealed envelope
        (correct fence, attempt, holder, and canonical identity). Fails closed
        otherwise. This is the common identity gate for renew/revoke/complete
        (contract §9); it does NOT check time, revocation, or terminal state —
        those are decided per action.
        """

        graph_id = str(envelope.get("graph_id"))
        node_id = str(envelope.get("node_id"))
        authority = self._node_authority(conn, graph_id, node_id)
        if authority is None or envelope.get("fence") != authority["fence"]:
            raise AuthorityError(f"stale fence cannot {action}")
        authoritative = authority["envelope"]
        if authoritative.get("attempt_id") != envelope.get("attempt_id"):
            raise AuthorityError(f"stale attempt cannot {action}")
        env_lease = envelope.get("lease")
        if not isinstance(env_lease, Mapping) or env_lease.get("holder") != authoritative[
            "lease"
        ].get("holder"):
            raise AuthorityError(f"lease holder cannot {action}")
        if _canonical_json(dict(envelope)) != _canonical_json(authoritative):
            raise AuthorityError(f"non-authoritative envelope cannot {action}")
        return graph_id, node_id, authority

    def _load_live_authority(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        current: datetime,
        *,
        action: str,
    ) -> tuple[str, str, int, dict[str, Any]]:
        """Return the authoritative envelope for a lifecycle transition or fail.

        A transition is only legal for the holder/attempt/fence that currently
        owns the node under an unexpired, un-revoked, non-terminal lease. This is
        the shared fail-closed gate for renew/complete (contract §9). The
        ``authority`` mapping derives the effective expiry externally from the
        grant plus the latest valid renewal; the envelope itself is immutable.
        """

        graph_id, node_id, authority = self._authority_gate(conn, envelope, action=action)
        if authority["completed"]:
            raise AuthorityError("node already completed")
        if authority["revoked"]:
            raise AuthorityError(f"revoked lease cannot {action}")
        if current >= authority["effective_expires"]:
            raise AuthorityError(f"expired lease cannot {action}")
        if current >= _parse_time(authority["envelope"]["deadline_utc"]):
            raise AuthorityError(f"expired deadline cannot {action}")
        return graph_id, node_id, authority["fence"], authority["envelope"]

    def renew_node(
        self,
        envelope: Mapping[str, Any],
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically extend the active lease via external bounded metadata.

        The sealed ``LEASE_GRANTED`` envelope is immutable forever; it is never
        re-sealed, rewritten, or stored inside the renewal event. ``ttl_s`` is a
        duration measured from the renewal moment, and the new effective expiry
        (``now + ttl_s``) must not reach or exceed the node deadline. Renewal is
        bound to graph/node/holder/attempt/fence and must occur strictly before
        the current effective expiry. It never revives an expired, revoked, or
        terminal lease (contract §9). Returns a current-authority record whose
        ``envelope`` is the original sealed envelope; effective expiry derives
        from grant plus this latest valid renewal.
        """

        renewed_at = _utc(now)
        ttl = _nonnegative_number(ttl_s, "ttl_s")
        if ttl == 0:
            raise AuthorityError("ttl_s must be greater than zero")
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                # Identity gate first (graph/node/holder/attempt/fence/canonical
                # envelope), without the time checks so the deadline cap and the
                # "never revive" rules are evaluated explicitly and in order.
                graph_id, node_id, authority = self._authority_gate(
                    conn, envelope, action="renew"
                )
                if authority["completed"]:
                    raise AuthorityError("node already completed")
                if authority["revoked"]:
                    raise AuthorityError("revoked lease cannot renew")
                authoritative = authority["envelope"]
                deadline = _parse_time(authoritative["deadline_utc"])
                new_expires = renewed_at + timedelta(seconds=ttl)
                # A renewal is a heartbeat extension. It may never shorten the
                # authority window or append a no-op renewal event.
                if new_expires <= authority["effective_expires"]:
                    raise AuthorityError("renewal must extend the effective lease expiry")
                # The new effective expiry must not reach or exceed the deadline.
                if new_expires >= deadline:
                    raise AuthorityError(
                        "renewal effective expiry must not exceed the node deadline"
                    )
                # Renewal must occur strictly before the current effective expiry
                # (now >= expires is already expired) and before the deadline.
                if renewed_at >= authority["effective_expires"]:
                    raise AuthorityError("expired lease cannot renew")
                if renewed_at >= deadline:
                    raise AuthorityError("expired deadline cannot renew")
                fence = authority["fence"]
                self._append_event(
                    conn,
                    kind="LEASE_RENEWED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(envelope["attempt_id"]),
                    fence=fence,
                    payload={
                        "holder": authoritative["lease"]["holder"],
                        "prior_expires_utc": _iso(authority["effective_expires"]),
                        "new_expires_utc": _iso(new_expires),
                    },
                    now=renewed_at,
                )
                conn.execute("COMMIT")
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(envelope["attempt_id"]),
                    "fence": fence,
                    "holder": authoritative["lease"]["holder"],
                    "envelope": authoritative,
                    "prior_expires_utc": _iso(authority["effective_expires"]),
                    "new_expires_utc": _iso(new_expires),
                    "effective_expires_utc": _iso(new_expires),
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def revoke_node(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Cancel the current lease so the node can be immediately regranted.

        Revocation uses the shared exact authority gate: the supplied envelope
        must be the byte-identical authoritative sealed envelope (current fence,
        attempt, holder, and canonical identity) — a forged holder or tampered
        envelope fails closed. Exact-expiry revoke is legal (an unexpired lease
        may be cancelled at any live moment). After revocation the old envelope
        validates ``lease_revoked``/``stale_fence`` and can never authorize
        execution; a fresh ``grant_node`` issues the next monotonic fence
        without waiting for the original TTL to elapse.
        """

        revoked_at = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id, node_id, authority = self._authority_gate(
                    conn, envelope, action="revoke"
                )
                if authority["completed"]:
                    raise AuthorityError("node already completed")
                if authority["revoked"]:
                    raise AuthorityError("lease already revoked")
                authoritative = authority["envelope"]
                self._append_event(
                    conn,
                    kind="LEASE_REVOKED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(authoritative["attempt_id"]),
                    fence=authority["fence"],
                    payload={"holder": authoritative["lease"]["holder"]},
                    now=revoked_at,
                )
                conn.execute("COMMIT")
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(authoritative["attempt_id"]),
                    "fence": authority["fence"],
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def complete_node(
        self,
        envelope: Mapping[str, Any],
        *,
        result_hash: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Accept the current unexpired holder/attempt/fence exactly once.

        Result acceptance requires the caller's ``result_hash`` (a SHA-256 over
        the canonical result; raw results are never stored). Acceptance is the
        sole authoritative terminal transition: it folds identity, expiry,
        deadline, revocation, and terminal state in one immediate transaction
        and appends exactly one ``RESULT_ACCEPTED`` event carrying the hash.

        A redelivery of the SAME accepted ``result_hash`` is an idempotent
        lookup returning the prior acceptance with no new event. A stale,
        expired, revoked, superseded, conflicting-duplicate, or forged
        completion appends exactly one typed ``RESULT_REJECTED`` event (bounded
        reason + result_hash) in the same decision transaction, then raises
        ``ResultRejection``. At most one accepted result per graph/node is
        backstopped by a partial unique index (contract §5, §9).
        """

        result_hash = _valid_sha256(result_hash, "result_hash")
        completed_at = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id = str(envelope.get("graph_id"))
                node_id = str(envelope.get("node_id"))
                try:
                    record = self._accept_result(
                        conn, envelope, result_hash, completed_at
                    )
                except _Reject as reject:
                    self._append_rejection(
                        conn,
                        envelope,
                        reject.reason,
                        result_hash,
                        completed_at,
                        fence_hint=reject.fence_hint,
                    )
                    conn.execute("COMMIT")
                    raise ResultRejection(reject.reason, result_hash=result_hash) from None
                conn.execute("COMMIT")
                return record
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def _accept_result(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        result_hash: str,
        current: datetime,
    ) -> dict[str, Any]:
        """Fold the acceptance decision; append RESULT_ACCEPTED or raise _Reject.

        Identity (graph/node/holder/attempt/fence/canonical envelope) is folded
        in deterministic order so a forgery that is also stale reports the more
        fundamental staleness reason first.
        """

        graph_id = str(envelope.get("graph_id"))
        node_id = str(envelope.get("node_id"))
        self._load_plan_node(conn, graph_id, node_id)  # unknown node -> AuthorityError
        authority = self._node_authority(conn, graph_id, node_id)

        def reject(reason: str, *, fence_hint: int | None = None) -> None:
            raise _Reject(reason, fence_hint=fence_hint)

        if authority is None:
            reject("no_grant")
        assert authority is not None
        if envelope.get("fence") != authority["fence"]:
            reject("stale_fence", fence_hint=authority["fence"])
        authoritative = authority["envelope"]
        if authoritative.get("attempt_id") != envelope.get("attempt_id"):
            reject("stale_attempt", fence_hint=authority["fence"])
        env_lease = envelope.get("lease")
        if not isinstance(env_lease, Mapping) or env_lease.get("holder") != authoritative[
            "lease"
        ].get("holder"):
            reject("holder_mismatch", fence_hint=authority["fence"])
        if _canonical_json(dict(envelope)) != _canonical_json(authoritative):
            reject("envelope_not_authoritative", fence_hint=authority["fence"])
        if authority["completed"]:
            if authority["accepted_result_hash"] == result_hash:
                # Idempotent redelivery of the same accepted result.
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(authoritative["attempt_id"]),
                    "fence": authority["fence"],
                    "holder": authoritative["lease"]["holder"],
                    "result_hash": result_hash,
                    "idempotent": True,
                }
            reject("node_completed", fence_hint=authority["fence"])
        if authority["revoked"]:
            reject("lease_revoked", fence_hint=authority["fence"])
        if current >= authority["effective_expires"]:
            reject("lease_expired", fence_hint=authority["fence"])
        if current >= _parse_time(authoritative["deadline_utc"]):
            reject("deadline_expired", fence_hint=authority["fence"])

        self._append_event(
            conn,
            kind="RESULT_ACCEPTED",
            graph_id=graph_id,
            node_id=node_id,
            attempt_id=str(authoritative["attempt_id"]),
            fence=authority["fence"],
            payload={
                "holder": authoritative["lease"]["holder"],
                "result_hash": result_hash,
            },
            now=current,
        )
        return {
            "graph_id": graph_id,
            "node_id": node_id,
            "attempt_id": str(authoritative["attempt_id"]),
            "fence": authority["fence"],
            "holder": authoritative["lease"]["holder"],
            "result_hash": result_hash,
        }

    def _append_rejection(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        reason: str,
        result_hash: str | None,
        now: datetime,
        *,
        fence_hint: int | None = None,
    ) -> None:
        """Append one typed RESULT_REJECTED audit event in the decision transaction.

        The event is stamped with the caller's claimed fence from the envelope
        (not the current live fence), so the audit trail faithfully records which
        fence the rejected caller claimed to hold.
        """

        # Use the presenting envelope's fence value so the audit record faithfully
        # captures what the caller claimed. fence_hint is the authoritative
        # current fence and is recorded in the payload for reconciliation.
        event_fence = envelope.get("fence")
        self._append_event(
            conn,
            kind="RESULT_REJECTED",
            graph_id=str(envelope.get("graph_id")),
            node_id=str(envelope.get("node_id")),
            attempt_id=str(envelope.get("attempt_id")) if envelope.get("attempt_id") else None,
            fence=event_fence,
            payload={"reason": reason, "result_hash": result_hash, "current_fence": fence_hint},
            now=now,
        )

    def current_fence(self, graph_id: str, node_id: str) -> int | None:
        conn = self._connect()
        try:
            row = self._latest_grant(conn, graph_id, node_id)
            return int(row["fence"]) if row is not None else None
        finally:
            conn.close()

    def current_authority(self, graph_id: str, node_id: str) -> dict[str, Any] | None:
        """Return the folded current authority view for a node, or ``None``.

        The ``envelope`` is the immutable sealed grant envelope. Effective
        expiry is derived externally from grant plus the latest valid renewal.
        """

        conn = self._connect()
        try:
            authority = self._node_authority(conn, graph_id, node_id)
            if authority is None:
                return None
            return {
                "fence": authority["fence"],
                "envelope": authority["envelope"],
                "granted_expires_utc": _iso(authority["granted_expires"]),
                "effective_expires_utc": _iso(authority["effective_expires"]),
                "revoked": authority["revoked"],
                "completed": authority["completed"],
                "accepted_result_hash": authority["accepted_result_hash"],
            }
        finally:
            conn.close()

    def validate_current(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> list[str]:
        errors = list(validate_sealed_envelope(envelope))
        current = _utc(now)
        conn = self._connect()
        try:
            try:
                plan, _ = self._load_plan_node(
                    conn, str(envelope.get("graph_id")), str(envelope.get("node_id"))
                )
            except AuthorityError:
                return sorted(set(errors + ["unknown_graph_node"]))
            authority = self._node_authority(
                conn, str(envelope.get("graph_id")), str(envelope.get("node_id"))
            )
            if authority is None or envelope.get("fence") != authority["fence"]:
                errors.append("stale_fence")
            if authority is not None:
                try:
                    authoritative = authority["envelope"]
                    if _canonical_json(dict(envelope)) != _canonical_json(authoritative):
                        errors.append("envelope_not_authoritative")
                except (KeyError, TypeError, ValueError):
                    errors.append("authoritative_envelope_invalid")
                if authority["revoked"]:
                    errors.append("lease_revoked")
                if authority["completed"]:
                    errors.append("node_terminal")
                # Effective expiry derives from grant + latest valid renewal, so
                # the original sealed envelope stays authoritative through a
                # renewal-extended lease.
                if current >= authority["effective_expires"]:
                    errors.append("lease_expired")
                if current >= _parse_time(authority["envelope"]["deadline_utc"]):
                    errors.append("deadline_expired")
            else:
                try:
                    granted = _parse_time(envelope.get("lease", {}).get("granted_utc"))
                    expires = granted + timedelta(
                        seconds=float(envelope.get("lease", {}).get("ttl_s"))
                    )
                    if current >= expires:
                        errors.append("lease_expired")
                    if current >= _parse_time(envelope.get("deadline_utc")):
                        errors.append("deadline_expired")
                except (AuthorityError, TypeError, ValueError):
                    errors.append("lease_or_deadline_invalid")
            if envelope.get("planner_hash") != plan["planner_hash"]:
                errors.append("planner_hash")
            if envelope.get("policy_hash") != plan["policy_hash"]:
                errors.append("policy_hash")
            return sorted(set(errors))
        finally:
            conn.close()

    @contextmanager
    def bind_current(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> Iterator[None]:
        """Publish one durable current node as the scoped execution authority."""

        errors = self.validate_current(envelope, now=now)
        if errors:
            raise AuthorityError("invalid current node authority: " + ", ".join(errors))
        fence = self.current_fence(str(envelope["graph_id"]), str(envelope["node_id"]))
        if fence is None:
            raise AuthorityError("missing authoritative current fence")
        with bind_sealed_envelope(envelope, current_fence=fence):
            yield

    @staticmethod
    def _budget_state(
        conn: sqlite3.Connection, graph_id: str, node_id: str, fence: int
    ) -> dict[str, Any]:
        rows = conn.execute(
            """SELECT kind, event_id, payload_json FROM authority_events
               WHERE graph_id = ? AND node_id = ? AND fence = ?
                 AND kind IN ('BUDGET_RESERVED', 'BUDGET_RECONCILED')
               ORDER BY id""",
            (graph_id, node_id, fence),
        ).fetchall()
        reservations: dict[str, dict[str, Any]] = {}
        reconciled: set[str] = set()
        charged_tokens = 0
        charged_usd = 0.0
        cost_unknown = False
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["kind"] == "BUDGET_RESERVED":
                reservations[row["event_id"]] = payload
            else:
                reservation_id = payload["reservation_id"]
                reconciled.add(reservation_id)
                charged_tokens += int(payload["actual_tokens"])
                if payload["actual_usd"] is None:
                    cost_unknown = True
                else:
                    charged_usd += float(payload["actual_usd"])
        reserved_tokens = sum(
            int(payload["tokens"])
            for event_id, payload in reservations.items()
            if event_id not in reconciled
        )
        reserved_usd = sum(
            float(payload["usd"])
            for event_id, payload in reservations.items()
            if event_id not in reconciled
        )
        return {
            "reserved_tokens": reserved_tokens,
            "reserved_usd": round(reserved_usd, 12),
            "charged_tokens": charged_tokens,
            "charged_usd": round(charged_usd, 12),
            "cost_unknown": cost_unknown,
        }

    def reserve_budget(
        self,
        envelope: Mapping[str, Any],
        *,
        tokens: int,
        usd: float,
        now: datetime | None = None,
    ) -> str:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise AuthorityError("tokens must be a non-negative integer")
        usd_value = _nonnegative_number(usd, "usd")
        current = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id = str(envelope.get("graph_id"))
                node_id = str(envelope.get("node_id"))
                authority = self._node_authority(conn, graph_id, node_id)
                if authority is None or envelope.get("fence") != authority["fence"]:
                    raise AuthorityError("stale fence cannot reserve budget")
                if authority["revoked"]:
                    raise AuthorityError("revoked lease cannot reserve budget")
                if authority["completed"]:
                    raise AuthorityError("completed node cannot reserve budget")
                latest_envelope = authority["envelope"]
                if latest_envelope.get("attempt_id") != envelope.get("attempt_id"):
                    raise AuthorityError("stale attempt cannot reserve budget")
                if current >= authority["effective_expires"]:
                    raise AuthorityError("expired lease cannot reserve budget")
                state = self._budget_state(conn, graph_id, node_id, int(envelope["fence"]))
                if state["cost_unknown"]:
                    raise AuthorityError("budget cost is unknown")
                limits = latest_envelope["budgets"]
                if state["charged_tokens"] + state["reserved_tokens"] + tokens > int(
                    limits["tokens_max"]
                ):
                    raise AuthorityError("token budget reservation exceeds node budget")
                if state["charged_usd"] + state["reserved_usd"] + usd_value > float(
                    limits["usd_max"]
                ):
                    raise AuthorityError("USD budget reservation exceeds node budget")
                reservation_id = f"res-{uuid.uuid4().hex}"
                self._append_event(
                    conn,
                    kind="BUDGET_RESERVED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(envelope["attempt_id"]),
                    fence=int(envelope["fence"]),
                    payload={"tokens": tokens, "usd": usd_value},
                    now=current,
                    event_id=reservation_id,
                )
                conn.execute("COMMIT")
                return reservation_id
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def reconcile_budget(
        self,
        reservation_id: str,
        *,
        actual_tokens: int,
        actual_usd: float | None,
        now: datetime | None = None,
    ) -> None:
        if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens < 0:
            raise AuthorityError("actual_tokens must be a non-negative integer")
        if actual_usd is not None:
            actual_usd = _nonnegative_number(actual_usd, "actual_usd")
        current = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                reservation = conn.execute(
                    "SELECT * FROM authority_events WHERE event_id = ? AND kind = 'BUDGET_RESERVED'",
                    (reservation_id,),
                ).fetchone()
                if reservation is None:
                    raise AuthorityError("unknown budget reservation")
                duplicate = conn.execute(
                    """SELECT 1 FROM authority_events
                       WHERE kind = 'BUDGET_RECONCILED'
                         AND json_extract(payload_json, '$.reservation_id') = ?""",
                    (reservation_id,),
                ).fetchone()
                if duplicate:
                    raise AuthorityError("budget reservation already reconciled")
                self._append_event(
                    conn,
                    kind="BUDGET_RECONCILED",
                    graph_id=reservation["graph_id"],
                    node_id=reservation["node_id"],
                    attempt_id=reservation["attempt_id"],
                    fence=reservation["fence"],
                    payload={
                        "reservation_id": reservation_id,
                        "actual_tokens": actual_tokens,
                        "actual_usd": actual_usd,
                    },
                    now=current,
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def budget_usage(self, graph_id: str, node_id: str, *, fence: int) -> dict[str, Any]:
        conn = self._connect()
        try:
            return self._budget_state(conn, graph_id, node_id, fence)
        finally:
            conn.close()

    def recover(self, *, now: datetime | None = None) -> dict[str, int]:
        """Verify the durable chain and poison expired, unreconciled reservations.

        Startup recovery verifies every lifecycle event's hash chain, then folds
        grant, renewal, revocation, terminal acceptance, and rejection events to
        derive current lease/terminal state. The sealed grant envelope stays
        immutable; effective expiry derives externally from the grant plus the
        latest valid renewal. A revoked or terminally accepted lease is dead;
        a renewed lease's expiry is honored. Any malformed payload or hash-chain
        tamper fails closed. Monotonic fences are preserved (grants are only
        ever read, never rewritten).

        A crash after reservation but before provider accounting leaves spend
        unknowable. Recovery reconciles that reservation with ``actual_usd=None``
        once its owning lease is dead, which preserves fail-closed budget behavior.
        """

        current = _utc(now)
        reconciled_count = 0
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute("SELECT * FROM authority_events ORDER BY id").fetchall()
                previous_hash: str | None = None
                reservations: dict[str, sqlite3.Row] = {}
                reconciled: set[str] = set()
                grants: dict[tuple[str, str, int], dict[str, Any]] = {}
                renewals: dict[tuple[str, str, int], dict[str, Any]] = {}
                dead_leases: set[tuple[str, str, int]] = set()
                for row in rows:
                    try:
                        payload = json.loads(row["payload_json"])
                    except (TypeError, ValueError) as exc:
                        raise AuthorityError("authority event hash chain is invalid") from exc
                    expected = _canonical_hash(
                        {
                            "event_id": row["event_id"],
                            "ts_utc": row["ts_utc"],
                            "kind": row["kind"],
                            "graph_id": row["graph_id"],
                            "node_id": row["node_id"],
                            "attempt_id": row["attempt_id"],
                            "fence": row["fence"],
                            "payload": payload,
                            "prev_event_hash": row["prev_event_hash"],
                        }
                    )
                    if row["prev_event_hash"] != previous_hash or row["event_hash"] != expected:
                        raise AuthorityError("authority event hash chain is invalid")
                    previous_hash = row["event_hash"]
                    if row["fence"] is None:
                        continue
                    key = (row["graph_id"], row["node_id"], int(row["fence"]))
                    if row["kind"] == "LEASE_GRANTED":
                        grants[key] = payload
                    elif row["kind"] == "LEASE_RENEWED":
                        renewals[key] = payload
                    elif row["kind"] in ("LEASE_REVOKED",) + _ACCEPTED_KINDS:
                        # Revocation and terminal acceptance close the lease.
                        dead_leases.add(key)
                    elif row["kind"] == "BUDGET_RESERVED":
                        reservations[row["event_id"]] = row
                    elif row["kind"] == "BUDGET_RECONCILED":
                        reconciled.add(str(payload.get("reservation_id")))
                    # RESULT_REJECTED is audit-only; it never closes a lease.

                for reservation_id, reservation in reservations.items():
                    if reservation_id in reconciled:
                        continue
                    key = (
                        reservation["graph_id"],
                        reservation["node_id"],
                        int(reservation["fence"]),
                    )
                    grant = grants.get(key)
                    if grant is None:
                        raise AuthorityError("orphaned reservation has no authoritative lease")
                    # A revoked or terminally accepted lease is dead now; renewal
                    # extends the effective expiry derived from the immutable
                    # grant envelope. Reconcile only once the live lease is dead.
                    if key not in dead_leases:
                        try:
                            envelope = grant["envelope"]
                            lease = envelope["lease"]
                            base_expires = _parse_time(lease["granted_utc"]) + timedelta(
                                seconds=float(lease["ttl_s"])
                            )
                            renewal = renewals.get(key)
                            expires = (
                                _parse_time(renewal["new_expires_utc"])
                                if renewal is not None
                                else base_expires
                            )
                        except (KeyError, TypeError, ValueError, AuthorityError) as exc:
                            raise AuthorityError("orphaned reservation lease is invalid") from exc
                        if current < expires:
                            continue
                    self._append_event(
                        conn,
                        kind="BUDGET_RECONCILED",
                        graph_id=reservation["graph_id"],
                        node_id=reservation["node_id"],
                        attempt_id=reservation["attempt_id"],
                        fence=reservation["fence"],
                        payload={
                            "reservation_id": reservation_id,
                            "actual_tokens": 0,
                            "actual_usd": None,
                            "recovery": "expired_lease_unknown_spend",
                        },
                        now=current,
                    )
                    reconciled_count += 1
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()
        return {"orphaned_reservations_reconciled": reconciled_count}

    def get_planner_hash(self, graph_id: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT planner_hash FROM sealed_plans WHERE graph_id = ?", (graph_id,)
            ).fetchone()
            return row["planner_hash"] if row else None
        finally:
            conn.close()

    def read_events(self, graph_id: str) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM authority_events WHERE graph_id = ? ORDER BY id", (graph_id,)
            )
            result = []
            for row in rows:
                record = dict(row)
                # Deserialize payload_json -> payload so callers can inspect
                # structured event payloads without an extra json.loads().
                try:
                    record["payload"] = json.loads(record["payload_json"])
                except (TypeError, ValueError, KeyError):
                    record["payload"] = None
                result.append(record)
            return result
        finally:
            conn.close()


__all__ = ["AuthorityError", "Phase2AuthorityStore", "ResultRejection", "default_db_path"]
