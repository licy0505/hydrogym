# L1A-2s CI / frozen-fixture closure

**Status:** `diagnostic_only`. No production solver, physics, geometry, timestep, acceptance threshold, data contract, or lineage default was changed. This closure does not promote contract 13, establish L1B readiness, or create training data. The hosted-check snapshot below is for the pre-closure PR head `78fc8d7`; the test/workflow changes require a new hosted run, whose result is reported separately rather than assumed here.

## 1. Base-SHA reproduction

Reproduced the five failures against PR #22's exact base, `002120f3a051e638a7e85f9db4022107dbe45780`, in a detached, temporary worktree. The command used Python 3.11.2 from the repository's locked environment, with `JAX_ENABLE_X64=1`, `PYTHONPATH=.`, `pytest -vv --tb=long --full-trace --showlocals`, and only the five named nodes. All five failed on the base. Complete pytest output and full tracebacks are preserved in `base_sha_pytest_full_tracebacks.txt` (13,590 lines; 1,343,610 bytes).

| Failing test | Base-SHA evidence | Classification |
|---|---|---|
| `test_no_production_phi_semantics_change` | Actual `phasefield.py` SHA-256 prefix `024665742dce67bc`; accepted prefixes were `4790c6235dd763db` and `ebb249a22fa2065f`. | Stale whole-file digest allow-list. The historical L1A-2l manifest assertion remains valid and unchanged. |
| `test_phasefield_source_unchanged` | Same actual whole-file digest, rejected by the same two older prefixes. | Stale whole-file digest allow-list. |
| `test_no_production_phase_rate_change` | Same actual full SHA-256, `024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c`, rejected before its later per-source comparisons. | Stale whole-file digest allow-list; not evidence of a phase-rate operator change. |
| `test_no_cutcell_geometry_change` | Its `sdf_cutcell_fv_v1` transport and `sdf_cutcell_v1` wall-measure assertions passed; the subsequent whole-file digest assertion failed. | Stale whole-file digest allow-list; not evidence of a geometry change. |
| `test_w_contact_angle_remains_open` | Pinned L1A-2m merge SHA is `ee15098ab0600b2bdf450162a3c89ec7ce2950d1`; `git log` returned `unavailable` in the depth-1 base checkout. GitHub's commit API confirms that SHA is the L1A-2m PR #16 merge. | Shallow-checkout fixture assumption, not a production or blocker-state change. |

## 2. Semantic-change check

The `phasefield.py` digest changed from `ebb249a22fa2065f…` at commit `165608c` to `024665742dce67bc…` at `7cf11e7`; the latter digest is also recorded in the existing L1A-2q and L1A-2r manifests. The diff from `165608c` to the base adds only the contract-12 lineage/policy documentation and metadata constants (`impact_phase_cap_dx2_v1`, its effective-dt rule, and the operator/trajectory semantics declaration). It does not edit a production operator function. Contract 12's timestep-policy change remains the previously sanctioned trajectory-semantic change; it is not a new L1A-2s physics change.

Independent protections were retained. The 20 function-source hash cases in `test_frozen_production_operators_unchanged`, the stencil/wetting/capillary/property checks, and the related threshold checks passed on the base: **25 passed** (`base_operator_hash_regression.txt`). No operator-level hash assertion was removed or relaxed. PR #22 still changes no `phasefield.py` source.

## 3. Minimal repairs

- Added the already-recorded L1A-2q/r whole-file seal `024665742dce67bc…` to the three stale phasefield digest allow-lists. The older frozen and contract-12-promotion seals remain accepted, and the independent operator-level assertions remain intact.
- Kept the L1A-2m merge SHA pinned. The fixture now accepts `unavailable` only when `git rev-parse --is-shallow-repository` confirms a shallow checkout; a full-history checkout must resolve the exact SHA. The new hosted closure step runs this test under Actions' default depth-1 checkout.
- Added a short dedicated CI step for the five repaired assertions plus the independent operator/hash guards. It runs from `examples/two_phase` and does not launch any L1A-2s forensic simulation.

No solver or dependency files were changed.

## 4. Validation

- Repaired closure tests plus operator guards: **28 passed in 1.71s** (`fixed_anchor_targeted_tests.txt`).
- Full local non-slow regression suite: **228 passed in 274.68s** (`full_non_slow_regression.txt`). This used Python 3.11.2 because the sandbox could not download the pinned Python 3.12 build; all packages were installed from `uv.lock`.
- Ruff on the three edited Python test files: passed. `py_compile`, YAML parsing, `git diff --check`, and `uv lock --check` against a temporary Python-3.11 copy of the same project metadata/lock: passed.
- Repo-wide `ruff check .` is **not clean**: 171 diagnostics (6 F841, 165 E501) on both base and PR worktrees, so none were introduced by this closure. The repo-wide import-order and spelling commands also returned nonzero; this closure did not broaden into those unrelated style fixes, and the hosted Quality job failed earlier at dependency audit and skipped lint, formatting, import-order, and spelling. The scoped test-file formatting check flags the same pre-existing hunk on base.

## 5. Hosted `uv audit --locked` finding

The hosted job's `Check lockfile` step passed, then `Audit locked dependencies` exited 1. Its GitHub annotation exposes only “Process completed with exit code 1”; the signed Actions log archive could not be fetched from this sandbox, so the exact job stdout is not available. The same audit step failed on the base commit and on PR #21. This PR does not change `uv.lock`, `pyproject.toml`, or the workflow.

A scan of all **262** locked package/version records against PyPI's per-release vulnerability metadata (source `osv`; raw results in `pypi_locked_vulnerability_scan.json`) identifies five vulnerable locked releases, corroborated against GitHub Advisory records. `uv audit` audits all extras and groups by default, so optional dependency groups are included:

| Locked release | OSV/GHSA/CVE records | Patched versions listed by the advisories |
|---|---|---|
| `jupyterlab==4.6.3` | CVE-2026-102830, CVE-2026-102831, CVE-2026-102904 | `4.6.4` (or `4.5.11` on the 4.5 line) |
| `notebook==7.6.2` | CVE-2026-102831 | `7.6.3` |
| `tornado==6.5.8` | GHSA-3hv7-mjh2-fv65, GHSA-chx6-46f5-w4vp, GHSA-c2m8-h5v5-343r | `6.5.9` |
| `urllib3==2.7.0` | CVE-2026-97687, CVE-2026-97688, CVE-2026-97689 | `2.8.0` |
| `werkzeug==3.1.8` | CVE-2026-102598 | `3.1.9` |

The local `uv audit` reproduction could resolve the same 262-package lock but could not reach `https://api.osv.dev/v1/querybatch` because outbound access is restricted; that local network error is **not** the hosted failure. The vulnerable locked versions and advisory records are the concrete pre-existing cause evidenced here. No lockfile/dependency update was included in this test-anchor-only closure.

## 6. Merge/readiness disposition

Do not merge as part of this closure. The frozen-fixture failures are repaired and the local suite is green, but the hosted dependency audit remains a genuine security failure; hosted lint was skipped, not passed, and repo-wide local Ruff already reproduces baseline violations. PR #22 should remain open for separate dependency/quality resolution. Merging later must not be described as L1B data readiness; all historical readiness/blocker statuses remain as in the L1A-2s report.
