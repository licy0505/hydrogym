# QUALITY-1 merge dependency plan

## Graph

```
main 002120f3
  ├─ PR #23 SEC-1 (94f16bf)  uv.lock + evidence/sec1   OPEN
  │     Quality: lock PASS, audit PASS, lint FAIL; format/isort/codespell SKIPPED
  ├─ PR #22 L1A-2s (19da2a4)  diagnostic-only          OPEN, independent
  └─ QUALITY-1  (this evidence)  cannot green Quality without Tier P/H edits
```

## Why quality-only on old main cannot go green

Quality job on `main` still fails `uv audit --locked` first; lint never runs. A quality-only PR against old lock inherits that deadlock.

## Preferred owner path (not executed)

1. Merge **PR #23 security-only** after hosted audit PASS (already true) **only if** branch protection allows merging with Quality still red because of **pre-existing lint**. That is a **human policy** decision (`MERGE_POLICY_BLOCKED` until owner says so). Do not force-merge.
2. Recreate QUALITY-1 on updated main **after** owner chooses:
   - **Option S:** authorized style reseal of named two-phase files + new source-hash protocol.
   - **Option R:** narrow ratchet for exact E501/F841 counts on named frozen paths, non-increasing, no global ignore.
3. Re-run full hosted Quality (all four style steps actually execute).
4. Only then rebase **PR #22** and re-run *its* CI. Do not start L1A-2t.

## This Arena session

Pinned to `arena/7e891333-hydrogym` (PR #23). Cannot legally open a second quality-only branch. Therefore QUALITY-1 here is **diagnostic evidence only**. It must not silently absorb style patches into the security PR.

## PR #22

Keep isolated. Head `19da2a4` unchanged. Quality still fails audit until #23 is in its base.

## Approval request (bounded)

Remaining Quality failures are 100% under `examples/two_phase/` (12 ruff files / 27 format / 28 isort / 3 codespell). Zero Tier-U files exist. Agent will not autoformat, isort, or F841-delete on P/H files without explicit Option S or R approval.
