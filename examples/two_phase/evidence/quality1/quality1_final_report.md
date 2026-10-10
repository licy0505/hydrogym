# QUALITY-1 final report

**Decision: `QUALITY_POLICY_APPROVAL_REQUIRED` (remaining issues are `PROTECTED_SOURCE_BLOCKED`).**

Not `QUALITY1_GREEN`. No style/solver patch was applied.

## Preflight (2026-10-10T06:35:22Z)

- main: `002120f3a051e638a7e85f9db4022107dbe45780`
- PR #23 OPEN `94f16bf` — hosted Quality FAIL at lint; audit reached (PASS); format/isort/codespell SKIPPED; imports 3.10–3.14 PASS; two-phase E2E IN_PROGRESS at query time.
- PR #22 OPEN `19da2a4` — Quality FAIL (audit); two-phase E2E SUCCESS historically.
- Workspace: `arena/7e891333-hydrogym`, clean at start, uv 0.12.5.

## Inventories (SEC-1 tree = same Python as main)

| Tool | Exit | Count |
|---|---:|---|
| ruff | 1 | 171 findings (165 E501, 6 F841) / 12 files |
| format | 1 | 27 files |
| isort | 1 | 28 files |
| codespell | 65 | 3 lines |

All hits are `examples/two_phase/**`. `chns_nonstationarity_audit.py` has 132/171 ruff findings. `phasefield.py` is Tier P (E501+F841+format+isort).

## Fixes performed

None on production/audit/test Python. No bulk format/isort. No CI weakening.

## Owner fork (do not auto-choose)

- **S** reseal named frozen sources after AST+hash+regression protocol.
- **R** path-and-rule ratchet with non-increasing counts; no global ignore.

## Flags

`SOLVER_CONTRACT_VERSION=12` `L1A_STATUS=BLOCKED` `L1B_DATA_NOT_READY` `L1A-2t=NOT_RUN`

## Next action

Obtain repository-owner approval for Option S or Option R on the named two-phase files; do not merge PR #22; do not start L1A-2t.
