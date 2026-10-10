# QUALITY-3B Owner-Approved Exact-SHA Reseal and Final PR #22 Integration

执行策略：fail closed。本报告只把已实际执行的检查记录为 PASS；`NOT_RUN`、`PENDING`、`ENVIRONMENT_BLOCKED` 和旧 PR head 的结果不升级。QUALITY-3B 保持 `SOLVER_CONTRACT_VERSION=12`、冻结 CFD 求解器、历史质量基线和 L1B 阶段状态不变。

## Current pre-hosted state

- Repo：`licy0505/hydrogym`。
- PR #23：`CLOSED + MERGED`，实际 merge SHA `039f6a0d9a14ff601c76a58f5a66ce0900677de5`；最新 `main` 已重新核验为同一 SHA。
- PR #22：`OPEN`，当前实际 head `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf`，base `002120f3a051e638a7e85f9db4022107dbe45780`。没有更新 PR #22 分支，也没有 merge/close。
- QUALITY-3 integration reference `39dfdab0ef384eabb6693833dc33a4b08b378d8e` 保留双 parent：PR #23 merged-main 与 PR #22 exact head；它不是 main merge。
- 当前代码候选提交（evidence publication 前）为 `86125cfaf8656af0de691cae6398ea58bc95eeae`。最终 hosted run 尚未完成，`hosted_ci_matrix.json` 的 `PENDING` 是有意的 fail-closed 状态。
- QUALITY-3B evidence manifest：`examples/two_phase/evidence/quality3b/manifest.json`，SHA-256 `bd09469fd7b24c05249138779845b9605179ceffaec4ce438674e4f7e18fc39c`。

## Verified GitHub owner attestation

仓库所有者在 PR #22 留下了可由 GitHub API 重查的 comment：

- actor：`licy0505`，GitHub user id `156068376`，`author_association=OWNER`
- comment id：`6098141190`
- URL：<https://github.com/licy0505/hydrogym/pull/22#issuecomment-6098141190>
- created/updated：`2026-10-10T13:46:35Z` / `2026-10-10T13:46:35Z`
- comment body SHA-256：`450ddc8ff92b4cec0e32a0f8ee1751fa3c63d94d18603f8f9982fcde045a0dc1`
- 两阶段 owner attestation commit：`6237cbb9614588b976384ca27e6a7c5d05aa3348`
- commit URL：<https://github.com/licy0505/hydrogym/commit/6237cbb9614588b976384ca27e6a7c5d05aa3348>
- GitHub API 显示该 commit 的 author/committer 均为 `licy0505`，且只新增 `quality/approvals/quality3b_owner_attestation.json`。GitHub API 没有报告 cryptographic signature；本报告不把 unsigned commit 写成签名验证。
- 独立 verifier 已实时查询 comment 和 commit API，并返回 `owner_approval_verified=true`。详细 provenance 在 `owner_approval_provenance.json`。

该 owner approval 只覆盖四项 exact old-to-new SHA transitions、PR #22 原 head、evidence hash `2187be0a33e2a68d49feb3cb4075883df3bd506e365b6562cd2b161e3aacff5d` 和 baseline hash `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`。它不授权 merge/close PR #22、生产物理修改、额外文件/诊断/豁免、L1A-2t、正式 L1B 数据、T=8 CFD 或 ML training。

## Exact migration and quality gate

`quality/approved_migrations.json` 的四项均绑定 `approval_commit=6237cbb9614588b976384ca27e6a7c5d05aa3348`、`approved_by=licy0505` 和 PR #22 head `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf`。每个 current source SHA 与 approved new SHA 完整相等；baseline 未改写。

四项 exact pairs 与 source/diagnostic mapping 见 `authorized_transitions.json`。四路质量输出为：

| tool | raw exit | historical | new | ratchet |
|---|---:|---:|---:|---|
| Ruff | 1 | 171 | 0 | `PASS_BASELINE_ONLY` |
| Ruff format | 1 | 27 | 0 | `PASS_BASELINE_ONLY` |
| isort | 1 | 28 | 0 | `PASS_BASELINE_ONLY` |
| codespell | 65 | 3 | 0 | `PASS_BASELINE_ONLY` |

这表示历史工具债务仍被原始工具报告，但没有新增诊断；不是声称 raw tools clean。QUALITY-3B verifier 加上既有 gate 对抗测试共 `44 passed`，其中 QUALITY-3B 新增 13 tests、12 个 fail-closed negative cases。

## Regression and identity

- L1A-2s focused module：`49 passed`。
- Frozen fixture closure：`28 passed`。
- `examples/two_phase/tests -m 'not slow'`：`228 passed`。
- 原有 two-phase component regression：solver core `93 passed`；wall `46 passed`；cut-cell `25 passed, 2 deselected`；remaining audits `84 passed`；合计 `248 passed, 2 deselected`。
- `capture_fields` numerical identity：`phi,u,v` exact equal；common ledger exact equal；唯一 capture-only key 为 `captured_fields`；RHS recomposition bitwise equal；fast-kernel dtype tolerance PASS。
- `phasefield.py`、`generate_dataset.py`、`cases.py` 和 `production/timestep_policy.py` 与最新 main 字节一致；contract 12、`impact_phase_cap_dx2_v1`、CHNS/wetting/cut-cell operators 和 physical thresholds 未改变。
- 本地 `uv lock --check --python 3.11` PASS；exact local `uv lock --check` 与 `uv audit --locked` 因 Python 3.12 standalone download TLS `UnknownIssuer` 环境阻塞，均没有写成 local PASS。Final hosted Quality 必须独立给出 authoritative `uv audit --locked` 结果。

## Hosted checks and merge boundary

`hosted_ci_matrix.json` 当前仍为 `QUALITY3B_HOSTED_CHECKS_PENDING`。旧 PR #22 head 的历史 Quality checks 不是本候选提交的结果，且当前 PR #22 branch 仍未更新。即使固定 session branch 的 final hosted run 变绿，也不能把独立 Arena 分支的绿色解释为 PR #22 当前 head 已满足 required checks。

由于本 session 只能写固定分支 `arena/9fa6b0dd-hydrogym`，不能无授权把 candidate 推入 PR #22 的 `arena/3d47375a-hydrogym`，本轮不能产生 `MERGE_ELIGIBLE_AWAITING_OWNER`。若 final candidate hosted CI 全绿，阶段判决最多为 `QUALITY3B_VALIDATED_PR22_HEAD_NOT_UPDATED`；若任一 required check 失败，则为 `QUALITY3B_CHECKS_FAILED`，不得 merge。

## Frozen milestone

```text
SOLVER_CONTRACT_VERSION=12
L1A_STATUS=BLOCKED
L1B_DATA_NOT_READY
L1A-2t=NOT_RUN
```

<!-- MACHINE-PARSEABLE BLOCK -->
```text
QUALITY-3B FINAL
main_sha / PR23_merge_sha: 039f6a0d9a14ff601c76a58f5a66ce0900677de5 / 039f6a0d9a14ff601c76a58f5a66ce0900677de5
PR22_original_sha / PR22_final_head_sha / PR22_state: 19da2a4d695e6cceabd0e6c3dcc505d09ba84daf / 19da2a4d695e6cceabd0e6c3dcc505d09ba84daf / OPEN
owner_approval_actor / GitHub_record_url / approval_commit: licy0505 / https://github.com/licy0505/hydrogym/pull/22#issuecomment-6098141190 / 6237cbb9614588b976384ca27e6a7c5d05aa3348
owner_approval_verified: TRUE
four_exact_transitions: 4 full pairs; actual source SHA matches all new SHA; approval=APPROVED
quality_baseline_sha / unchanged: 52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b / TRUE
raw_ruff / raw_format / raw_isort / raw_codespell: RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(65)
ratchet_ruff / format / isort / codespell: PASS_BASELINE_ONLY / PASS_BASELINE_ONLY / PASS_BASELINE_ONLY / PASS_BASELINE_ONLY
historical_ruff / format / isort / codespell: 171 / 27 / 28 / 3
new_diagnostics: 0
adversarial_tests / focused_49 / fixture_28 / nonslow_228 / component_248: 44 PASS / 49 PASS / 28 PASS / 228 PASS / 248 PASS + 2 deselected
operator_identity / numerical_equivalence / frozen_source_checks: PASS / PASS / PASS
evidence_manifest_path / SHA: examples/two_phase/evidence/quality3b/manifest.json / bd09469fd7b24c05249138779845b9605179ceffaec4ce438674e4f7e18fc39c
uv_lock_check / hosted_uv_audit / core_imports_3p10_to_3p14 / hosted_two_phase_e2e: local-compatible-PASS; exact-local-ENVIRONMENT_BLOCKED / PENDING / PENDING / PENDING
hosted_run_id / required_checks / workflow_sha / PR22_head_sha_match: PENDING / PENDING / PENDING / FALSE
GitHub_merge_eligibility / actual_merges_performed: NOT_YET_ELIGIBLE_PR22_HEAD_NOT_UPDATED / NONE
SOLVER_CONTRACT_VERSION=12
L1A_STATUS=BLOCKED
L1B_DATA_NOT_READY
L1A-2t=NOT_RUN
DECISION: QUALITY3B_HOSTED_CHECKS_PENDING
EXACTLY_ONE_NEXT_ACTION: Run and record required hosted CI on the final candidate; if green, retain PR #22 OPEN because its head is not updated; do not merge or close.
```
