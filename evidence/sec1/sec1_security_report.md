# SEC-1 locked dependency security remediation

- **Workspace:** `/home/user/hydrogym` on Arena session branch `arena/7e891333-hydrogym` (not PR #22 `arena/3d47375a-hydrogym`).
- **Clean base:** `git status --porcelain` empty before edits; `HEAD == origin/main == 002120f3a051e638a7e85f9db4022107dbe45780`.
- **UTC capture:** 2026-10-10T06:19:33Z (baseline) / 2026-10-10T06:21:01Z (quality re-run).
- **Tools:** `uv 0.12.5`, CPython 3.11.2 (`/usr/bin/python3`), `UV_PYTHON_DOWNLOADS=never`.
- **Remote:** `https://github.com/licy0505/hydrogym.git`

## Decision

`SECURITY_FIXED_BUT_QUALITY_BLOCKED`

Hosted `uv audit --locked` remains the authoritative vulnerability verdict. Local OSV (`https://api.osv.dev/v1/querybatch`) was **NETWORK_UNAVAILABLE** (tls handshake eof). Quality lint/format/isort/codespell were executed locally and **failed** with pre-existing two-phase debt. Those steps were SKIPPED on main/PR #22 because audit failed first. SEC-1 does **not** patch solver/quality files.

## Lock remediation

Command (only `uv.lock` changed; `pyproject.toml` unchanged):

```
uv lock --upgrade-package jupyterlab \
        --upgrade-package notebook \
        --upgrade-package tornado \
        --upgrade-package urllib3 \
        --upgrade-package werkzeug
```

Resolver: 262 packages; **five** version bumps, no unrelated scientific/JAX upgrades.

| Package | Old | New | Advisory floor (2026-10-09 hosted) | Role |
|---|---|---|---|---|
| jupyterlab | 4.6.3 | 4.6.4 | >=4.6.4 | direct `interactive` extra |
| notebook | 7.6.2 | 7.6.3 | >=7.6.3 | direct `interactive` extra |
| tornado | 6.5.8 | 6.5.10 | >=6.5.9 | transitive (Jupyter stack) |
| urllib3 | 2.7.0 | 2.8.0 | >=2.8.0 | transitive |
| werkzeug | 3.1.8 | 3.1.9 | >=3.1.9 | transitive |

`uv lock --check` local: **PASS** (exit 0).

`uv audit --locked` local: **NOT_RUN** / `NETWORK_UNAVAILABLE` (cannot reach OSV; not claimed PASS).

Historical hosted job: https://github.com/licy0505/hydrogym/actions/runs/37904603600/job/113734922481 (18 records / 5 releases).

No `--ignore-vuln`, no workflow edits.

## Quality (local, exact CI commands)

| Step | Exit | Verdict |
|---|---:|---|
| uv sync --locked --only-group dev --no-install-project | 0 | PASS |
| uv lock --check | 0 | PASS |
| uv audit --locked | 2 (OSV unreachable) | NOT_RUN |
| ruff check . | 1 | FAIL (pre-existing; includes `phasefield.py` F841/E501) |
| ruff format --check --diff . | 1 | FAIL (27 files would reformat) |
| isort . --check-only --diff | 1 | FAIL (two_phase import order) |
| codespell | 65 | FAIL (pre-existing false-positives in production audits) |

Classification after audit repair: **SECURITY_FIXED_BUT_QUALITY_BLOCKED**. Do not treat CI quality as green.

## Python / solver

- Core `uv sync --locked --no-default-groups` + `import hydrogym` on **3.11.2**: PASS.
- Python 3.10, 3.12, 3.13, 3.14: **NOT_RUN** (interpreters absent).
- Two-phase E2E / pytest: **NOT_RUN** locally (JAX extra not fully exercised here).
- Hosted PR #22 two-phase smoke on *old* lock: SUCCESS (does not prove this head).
- Frozen files unchanged vs main: `phasefield.py`, `generate_dataset.py`, `cases.py`, `.github/workflows/python-ci.yml`, `pyproject.toml`.

## PR #22

- OPEN, head `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf`, branch `arena/3d47375a-hydrogym`.
- Quality still FAIL on audit; **no post-SEC-1 CI**.
- This session did not edit or push that branch.
- After SEC-1 merge: `PR22_RECHECK_REQUIRED`.

## Readiness flags (unchanged)

- `SOLVER_CONTRACT_VERSION=12`
- `L1A_STATUS=BLOCKED`
- `L1B_DATA_NOT_READY`
- `L1A-2t=NOT_RUN`

## Next action

(c) After hosted `uv audit --locked` is green on this PR, **secure merge authorization is still blocked** until lint/format/isort/codespell are handled in a **separate** non-solver-quality PR — or an authorized owner accepts that those failures predate SEC-1. Smallest unblock for *security* is hosted audit PASS then authorized merge of the lockfile; quality remains a separate blocker for a fully green `Python CI` workflow.
