# QUALITY-3 Frozen Quality-Baseline Reconciliation & Integration Validation

执行策略：fail closed。本文档只报告已实际执行的检查；未执行项不填为 PASS。QUALITY-3 不重新计算 L1A-2s，不改变 HydroGym two-phase 生产算子、物理参数、solver contract 或历史 L1A-2s 结论。

## Provenance

- 查询时间窗口：2026-10-10 UTC；最新 main 为 `039f6a0d9a14ff601c76a58f5a66ce0900677de5`。
- PR #23 已由仓库所有者 `licy0505` 合并，实际 merge SHA 是 `039f6a0d9a14ff601c76a58f5a66ce0900677de5`，head SHA 是 `ea00347c8a6cebf6cfbff36f744d39ccd60784d1`。没有把 PR #23 head 当作 merge SHA。
- PR #22 仍 OPEN，head 是 `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf`，original `base_main_sha` 是 `002120f3a051e638a7e85f9db4022107dbe45780`，未执行 PR merge。
- 实际集成预检无冲突；保留 ancestry 的本地 integration merge 是 `39dfdab0ef384eabb6693833dc33a4b08b378d8e`，父提交为 merged-main `039f6a0...` 和 PR #22 exact head `19da2a4...`，merge-base 为 `002120f...`，预检 tree 为 `ac6ac3f0651f7bcf211f2e3775fe5891a5360061`。
- `git diff --check` 返回 0。预合并工作区 `git status --porcelain` 为空；当前工作区包含 QUALITY-3 证据和修复，尚未执行任何 merge/close 操作。
- 四个且仅四个 QUALITY-2 pinned 文件与 PR #22 重叠：
  1. `examples/two_phase/production/impact_impulse_projection_audit.py`
  2. `examples/two_phase/tests/test_inactive_phase_coupling_audit.py`
  3. `examples/two_phase/tests/test_l1a_data_readiness_exit_audit.py`
  4. `examples/two_phase/tests/test_stationarity_metric_domain_audit.py`

## Quality and migration result

四路原始工具都真实执行，仍保留原历史 multiset，没有发现新增诊断；原始工具非零并不等于 RAW_CLEAN：

| tool | raw exit | historical | added | ratchet result |
|---|---:|---:|---:|---|
| Ruff | 1 | 171 | 0 | FAIL：4 个 protected SHA transition 待 owner reseal |
| Ruff format | 1 | 27 | 0 | FAIL：4 个 protected SHA transition 待 owner reseal |
| isort | 1 | 28 | 0 | FAIL：4 个 protected SHA transition 待 owner reseal |
| codespell | 65 | 3 | 0 | FAIL：4 个 protected SHA transition 待 owner reseal |

质量 gate 输出同时记录了原始 return code 和 ratchet 判定。新增的三个 PR #22 Python 文件已逐文件执行 Ruff、format、isort、codespell，四项均 exit 0；它们没有继承旧债务豁免。L1A-2s 原有两个新增脚本的 import disorder 已作局部、可审计修复，未对冻结生产算子做格式化。

`quality/approved_migrations.json` 有 4 个 exact old-to-new SHA 条目，全部为 `LEGACY_SHIFTED_APPROVED` 的 `PENDING_OWNER`；没有任何 `APPROVED`。无摘要重算、无计数净抵消、无 wildcard/prefix SHA。baseline JSON 未更新，原 baseline SHA-256 为 `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`。

四个迁移的 old/new SHA、PR #22 commit、源码 diff fingerprint、诊断映射及 evidence hash 见 `diagnostic_ancestry_mapping.json` 与 `authorized_sha_transition_manifest.json`。没有 `LEGACY_REMOVED_APPROVED`，因此没有降低历史数量；没有 `NEW_QUALITY_DEBT`。

## Frozen source and numerical identity

- `SOLVER_CONTRACT_VERSION=12`。
- `impact_phase_cap_dx2_v1` 未改变。
- `phasefield.py`、`generate_dataset.py`、`cases.py`、`timestep_policy.py` 以及 CHNS、wetting、cut-cell、质量/物理阈值配置与 merged main 字节一致；各 SHA 见 `frozen_source_and_numerical_identity.json`。
- `C2` 保持 diagnostic-only；没有写入 `pf.step` 或生产 `rhs` 返回的求解状态。官方数据集、L1B readiness 和生产默认 initializer 未改变。
- 同一冻结输入下，`capture_fields=False` 与 `capture_fields=True` 的 `phi,u,v` 为逐数组 exact equal，共有 ledger 值 exact equal，唯一新增 ledger key 是 `captured_fields`，结果 PASS。
- fast kernel 与 `pf.step` 的既有 dtype tolerance、RHS bitwise recomposition、Poisson/CG 及物理观测验证 PASS；详情和实际数值见 `raw/ledger_validation.json`。
- L1A-2s compatibility focused test：49 passed；fixture closure：28 passed。历史阈值、反篡改和 contract-12 assertions 未弱化。

## Regression and hosted checks

- QUALITY-3 adversarial suite：31 passed，其中新增 QUALITY-3 测试 12 项，覆盖新文件 E501/F841、SHA drift、等数量替换、无证据行移位、批准删除后重新引入、baseline 篡改、跨目录诊断、拼写/import disorder、tool/parse/missing-stage、exact overlap pair 和 frozen physics source。
- `examples/two_phase/tests -m 'not slow'`（在 `examples/two_phase` cwd）：228 passed。
- 原有 two-phase unit regression 分组件完成：solver core 93 passed；wall 46 passed；cut-cell transport 25 passed、2 deselected；其余 audit 84 passed、合计 248 passed、2 deselected。一个聚合调用超过本地 30 分钟预算，因此只记录 TIMEOUT；组件结果没有被伪装成聚合调用 PASS。
- merged main `039f6a0...` 的 hosted Python CI Quality、Two-phase unit + E2E smoke、五种 core import、package、CodeQL、部署 build 均有对应记录；Python CI Quality job `114168222944` 与 E2E job `114168223007` 为 SUCCESS。GitHub Pages deploy job `114168485062` 为 FAILURE；branch protection API 返回 403，无法独立证明其 required/non-required 属性。
- PR #22 exact head 的 hosted Two-phase、imports、package、CodeQL、deployment 和分析 jobs 为 SUCCESS，但 Quality job `113734922481` FAILURE：`Audit locked dependencies` FAILURE，Lint/format/isort/codespell 随后 SKIPPED。因此 PR #22 当前不能称为 required checks 全绿。
- 本地 exact `uv lock --check` 和 exact `uv audit --locked` 因环境没有 Python 3.12 返回环境失败；`uv lock --check --python 3.11` PASS。不能把本地 audit 写成 PASS。merged main hosted authoritative `uv audit --locked` 为 SUCCESS；PR #22 head 的 audit step 为 FAILURE。

## Frozen milestone status

本轮没有启动 L1A-2t 正式实验，没有生成正式 L1B 训练数据，没有默认 C2，没有 T=8 重算，也没有 ML training。历史 L1A-2s status 原样保留：

```text
SOLVER_CONTRACT_VERSION=12
L1A_STATUS=BLOCKED
L1B_DATA_NOT_READY
L1A-2t=NOT_RUN
```

<!-- MACHINE-PARSEABLE BLOCK -->
```text
QUALITY-3 FINAL
main_sha / latest_integrated_main_sha: 039f6a0d9a14ff601c76a58f5a66ce0900677de5 / 039f6a0d9a14ff601c76a58f5a66ce0900677de5
PR23 state / merge_sha / head / required_checks: CLOSED+MERGED / 039f6a0d9a14ff601c76a58f5a66ce0900677de5 / ea00347c8a6cebf6cfbff36f744d39ccd60784d1 / Python-CI-and-CodeQL-PASS; Pages-FAIL; required-classification-UNOBSERVABLE-HTTP403
PR22 state / head / original_base_sha / integrated_tree_sha: OPEN / 19da2a4d695e6cceabd0e6c3dcc505d09ba84daf / 002120f3a051e638a7e85f9db4022107dbe45780 / ac6ac3f0651f7bcf211f2e3775fe5891a5360061
workspace_branch / clean_at_start / permission: arena/9fa6b0dd-hydrogym / true / session-branch-write; owner-approval-not-observed
conflicted_pinned_files: 4 (exact paths in pr22_quality_overlap_matrix.json)
raw_ruff / raw_format / raw_isort / raw_codespell: RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(65)
ratchet_historical / added / removed / moved / approved: 171+27+28+3 / 0 / 0 / 0 / 0; pending_owner=4
provenance_and_sha_checks: PASS for frozen sources; PENDING_OWNER_RESEAL for exact four overlap pairs
operator_identity / frozen_source_tests: owner=licy0505; agent=Arena session / focused=49 PASS; fixture=28 PASS; numerical=PASS
quality_gate / uv_audit / two_phase_e2e / non_slow_pytest: BLOCKED_OWNER_RESEAL / hosted-main-PASS; local-exact-ENVIRONMENT_BLOCKED / hosted-main-PASS / 228 PASS
new_tests / adversarial_tests: 12 new QUALITY-3 tests / 31 total PASS
 evidence_manifest_path / SHA: examples/two_phase/evidence/quality3/manifest.json / 499bc947840b2613ab2d525d5d33d4f1c14fde4cc44fa0c1b7ab85e64da564cd
PR23_merge_eligibility / PR22_merge_eligibility: ALREADY_MERGED_BY_OWNER_WITH_PAGES_FAILURE_UNCLASSIFIED / NOT_ELIGIBLE
SOLVER_CONTRACT_VERSION=12
L1A_STATUS=BLOCKED
L1B_DATA_NOT_READY
L1A-2t=NOT_RUN
DECISION: OWNER_RESEAL_APPROVAL_REQUIRED
EXACTLY_ONE_NEXT_ACTION: Repository owner reviews and explicitly approves the four named exact SHA transitions, then reruns PR #22 required hosted checks on the exact integrated commit; no merge is executed by this agent.
```

## Decision

主 verdict 是 `OWNER_RESEAL_APPROVAL_REQUIRED`。这是 fail-closed 阻塞，不是 `QUALITY3_INTEGRATION_VALIDATED`：源码、冻结物理身份、局部数值等价性和本地回归证据已记录，但四个受保护文件的 exact SHA transition 没有仓库所有者批准，PR #22 hosted Quality 也没有全绿。下一步只能由仓库所有者审查并批准上述四个 named transitions，随后在 exact integrated commit 上重新运行 required checks；本轮不 merge PR #22。
