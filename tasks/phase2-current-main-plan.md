# Phase 2 current-main execution plan

**Baseline:** `4d9401f0896d1640126320ad8f6f83d447e834e8`
**Acceptance-test restoration commit:** `d644f5c070635910073f5c4bfa718b2f92d0b584`
**Worktree:** `/home/jack/worktrees/hermes-phase2-current-main`
**Production enforcement:** unset/off; this plan does not authorize activation.

## Resource envelope

- Hermes flat-child cap: 4, spawn depth: 1.
- Local Qwythos: 2 real slots × 65,536 tokens; additional local children queue.
- Codex quota: 89% remaining from a snapshot 5 minutes old at audit time.
- Claude quota: 58% weekly remaining from a snapshot over 10 hours old; do not spend it as if current.
- OmniRoute observed global operating ceiling remains 6 concurrent requests. Use at most 2 local + 2 hosted children in a batch and no recursive fan-out.
- Hosted route is reserved for bounded adversarial review and exact-current verification. Implementation stays in the parent or isolated coding lane; no blind multi-agent writes.

## Gates

1. **Pre-flight — contract:** Fable authors a narrow contract-v2 delta; local and hosted reviewers inspect it. Enforcement stays off.
2. **Pre-flight — baseline:** restored Phase 0/1 tests must pass before Phase 2 source moves.
3. **Revision — port:** copy candidate tests first, observe RED where current main lacks Phase 2, then port smallest source slices. Max 3 review cycles per slice.
4. **Abort — authority drift:** stop if production enforcement becomes enabled, the implementation worktree gains unrelated files, or a claimed authority is only context-local/advisory.
5. **Revision — readiness:** positive and negative readiness controls must both behave correctly; a timed-out/missing reviewer verdict is not approval.
6. **Pre-flight — seal:** stage an explicit file allowlist; check cached diff, static diagnostics, and selected/broad tests.
7. **Independent verification:** detached worktree at exact commit plus hosted adversarial review.
8. **GO/NO-GO:** only the final evidence artifact decides. Source progress does not imply activation readiness.

## Dependency order

1. Seal `target-contract-v2.md` and versioned readiness-gate semantics.
2. Port default-off envelope validation and seam blocking with existing unsafe transports still blocked.
3. Build a Hermes-owned immutable graph/node envelope producer and canonical node/fence store.
4. Add transactional budget reservation/reconciliation and claim lifecycle `CLAIMED -> STARTED -> TERMINAL`.
5. Add canonical rejection/execution receipts.
6. Add safe executor slices only where the transport itself can enforce policy:
   - structured no-shell terminal argv;
   - descriptor-relative filesystem access;
   - redirect/DNS/final-IP network adapter;
   - child envelope derivation, reservation, receipt propagation, and reconciliation.
7. Keep all unsupported surfaces blocked while enforcement is enabled.
8. Run representative flags-on shadow, full fail-closed matrix, broad suite, detached verification, and readiness gate.

## Verification baseline

`52 passed` for restored Phase 0/1 acceptance tests on current main before Phase 2 changes.
