# Hosted CI follow-up after the frozen-fixture closure commit

**Scope:** PR #22 at head `7b87dd31354817b942502923b18d956db477dd68`; this is an appended hosted observation, not a change to the earlier pre-closure snapshot in `ci_frozen_fixture_closure.md`.

## Observed checks

- Python CI runs `37902686932` and `37902693265` completed on the same head. The two-phase unit-test step passed. The newly added **L1A-2s frozen-fixture closure** step failed with pytest exit code `4`; the subsequent two-phase audits were skipped. Core imports, package-distribution checks, deployment smoke, CodeQL, and analysis checks succeeded.
- The Quality job passed the lockfile check, then `Audit locked dependencies` failed with exit code `4`. Lint, format, import-order, and spelling steps were skipped, not passes. The hosted annotation contains only the exit code; retrieving the signed Actions log archive from this sandbox returned EOF, so no hosted stdout is claimed. The locked vulnerable releases and their advisory records remain listed in `ci_frozen_fixture_closure.md`; `uv.lock` is unchanged.

## Demonstrated closure-step cause and correction

The hosted annotation identified the closure-test step, but the raw step log was inaccessible. I reproduced the exact selected-node pytest command locally from `examples/two_phase`:

- Without a `PYTHONPATH`, the tests under `tests/` failed collection with `ModuleNotFoundError` for the sibling `phasefield` and `cases` modules; pytest returned exit code `4` (“no collectors”).
- With `PYTHONPATH=.` set for that working directory, the same selected node list passed: **28 passed in 1.74s**.

The workflow follow-up adds `PYTHONPATH: .` specifically to the closure-test step. A fresh hosted run is required to verify that correction; its outcome must not be presumed from the local reproduction. PR #22 remains open and unmerged.
