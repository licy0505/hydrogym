# QUALITY-3 Frozen Quality-Baseline Reconciliation & Integration Validation

执行策略：fail closed。本文档只报告已实际执行的检查；未执行项不填为 PASS。当前报告记录四项 exact SHA migration 已获仓库所有者批准，并已在批准后的 exact integrated commit 上完成 hosted required checks；这不等同于授权合并 PR #22。QUALITY-3 不重新计算 L1A-2s，不改变 HydroGym two-phase 生产算子、物理参数、solver contract 或历史 L1A-2s 结论。

## Provenance

- 查询时间窗口：2026-10-10 UTC；merged main 为 `039f6a0d9a14ff601c76a58f5a66ce0900677de5`。
- PR #23 已由仓库所有者 `licy0505` 合并，实际 merge SHA 是 `039f6a0d9a14ff601c76a58f5a66ce0900677de5`，head SHA 是 `ea00347c8a6cebf6cfbff36f744d39ccd60784d1`。没有把 PR #23 head 当作 merge SHA。
- PR #22 仍 OPEN，head 是 `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf`，original `base_main_sha` 是 `002120f3a051e638a7e85f9db4022107dbe45780`，未执行 PR merge。
- 实际集成预检无冲突；保留 ancestry 的本地 integration merge 是 `39dfdab0ef384eabb6693833dc33a4b08b378d8e`，父提交为 merged-main `039f6a0...` 和 PR #22 exact head `19da2a4...`，merge-base 为 `002120f...`，预检 tree 为 `ac6ac3f0651f7bcf211f2e3775fe5891a5360061`。
- `git diff --check` 返回 0。没有执行 PR #22 merge/close 操作。`quality/owner_approval_quality3_sha_transitions_2026-10-10.json` 和 `owner_approval_reseal.json` 保存本轮显式 owner approval 的范围与 exact pair。
- 四个且仅四个 QUALITY-2 pinned 文件与 PR #22 重叠：
  1. `examples/two_phase/production/impact_impulse_projection_audit.py`
  2. `examples/two_phase/tests/test_inactive_phase_coupling_audit.py`
  3. `examples/two_phase/tests/test_l1a_data_readiness_exit_audit.py`
  4. `examples/two_phase/tests/test_stationarity_metric_domain_audit.py`

## Quality and migration result

四路原始工具已重新执行，仍保留原历史 multiset，没有新增诊断。raw tool 的非零返回码表示历史债务仍被工具报告；ratchet gate 已按 owner-approved exact transitions 放行：

| tool | raw exit | historical | added | ratchet result |
|---|---:|---:|---:|---|
| Ruff | 1 | 171 | 0 | PASS_BASELINE_ONLY；4 个 exact SHA transitions approved |
| Ruff format | 1 | 27 | 0 | PASS_BASELINE_ONLY；4 个 exact SHA transitions approved |
| isort | 1 | 28 | 0 | PASS_BASELINE_ONLY；4 个 exact SHA transitions approved |
| codespell | 65 | 3 | 0 | PASS_BASELINE_ONLY；4 个 exact SHA transitions approved |

`quality/approved_migrations.json` 的 4 个条目均为 `LEGACY_SHIFTED_APPROVED`、`approval_status=APPROVED`，批准人为 `licy0505`，approval marker commit 为 `0909dd69fa847c51e75dad7528b0f066823baebd`。source manifest SHA-256 为 `b5f79986e0358fe128265d88863a25a5a6007b10565006514571d8caaa484e8a`；baseline JSON 未更新，baseline SHA-256 仍为 `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`。

| path | old SHA-256 | new SHA-256 | PR #22 commit | mapped debt |
|---|---|---|---|---|
| `production/impact_impulse_projection_audit.py` | `b3da6608e06704c56bc7e9ed7f4d0c7d7eed54769635c0cf83c424e182690691` | `5a8a4d476e0daa7d9249da1eb0a536011021e68e6ff535703d76488fd122a9b9` | `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf` | format + isort |
| `tests/test_inactive_phase_coupling_audit.py` | `869c53b6050aa1ecee9b5da611415fbae15868a658b1746ae604ce5e73f381f7` | `1c97db79f830d5387ffede08a4a4f5b4e65139eda68fa407bce8cb2ebf7b5c29` | `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf` | isort |
| `tests/test_l1a_data_readiness_exit_audit.py` | `ec4e780169391b96f0af67416dab015fb47254893a1c8e05a8c9c3ea655f379c` | `e93a05cd5d9a9140e1294f36c788cf54540f8a9476b2eb93c7ac4a8c8083a94f` | `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf` | isort |
| `tests/test_stationarity_metric_domain_audit.py` | `808f52d749797af44ded20889086d919dc071ec604f0d107797baf936fa60b76` | `2418f24262dcb6593586cd2fad4506113ad64878a72360aba8583ba31ae6a215` | `19da2a4d695e6cceabd0e6c3dcc505d09ba84daf` | format |

没有 `LEGACY_REMOVED_APPROVED`，因此没有降低历史数量；没有 `NEW_QUALITY_DEBT`，没有摘要重算、计数净抵消或 wildcard/prefix SHA。旧的 pre-approval evidence 仍保留在 `authorized_sha_transition_manifest.json`、`baseline_reconciliation_matrix.json`、`diagnostic_ancestry_mapping.json` 和对应 raw artifacts 中，没有被改写；本轮新增 post-approval raw output 在 `raw/ratchet_post_approval.txt`。

新增的 PR #22 Python 文件已逐文件执行 Ruff、format、isort、codespell，四项均 exit 0；它们没有继承旧债务豁免。L1A-2s 原有两个新增脚本的 import disorder 已作局部、可审计修复，未对冻结生产算子做格式化。

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
- 本地批准后 gate test：31 passed；ratchet exit 0，且输出 `APPROVED=4`、`sha_transitions_approved=4`、`sha_drift_unapproved=0`。
- 本地 `uv lock --check --python 3.11` exit 0。exact `uv lock --check` 和 exact `uv audit --locked` 仍因环境无法取得 Python 3.12（GitHub standalone download 的 TLS `UnknownIssuer`）而 exit 2；这两项没有写成 local PASS。`uv audit --locked` 必须由 hosted required check 实际复核。
- merged main `039f6a0...` 的 hosted Python CI Quality、Two-phase unit + E2E smoke、五种 core import、package、CodeQL、部署 build 均有对应记录；Python CI Quality job `114168222944` 与 E2E job `114168223007` 为 SUCCESS。GitHub Pages deploy job `114168485062` 为 FAILURE；branch protection API 返回 403，无法独立证明其 required/non-required 属性。
- PR #22 exact head 的 hosted Two-phase、imports、package、CodeQL、deployment 和分析 jobs 为 SUCCESS，但 Quality job `113734922481` FAILURE：`Audit locked dependencies` FAILURE，Lint/format/isort/codespell 随后 SKIPPED。因此 PR #22 当前仍不能称为 required checks 全绿。
- 旧 session-branch run `38045950758`/`38047079721` 是 pre-approval historical runs：前者因 4 个 `PENDING_OWNER` transition 在 Quality gate 阻塞，后者的 strict lint 仍对应 approval 尚未写入的 commit；它们没有被重写成当前 PASS。
- owner approval 后的 exact integrated commit `7a0284c438d37f0015f079a7869c3ea640c6e9cb` 已完成 hosted Python CI run `38053658694`；Quality（含 `uv audit --locked`）、Two-phase unit + E2E、五种 core import 和 package 八个 job 均 SUCCESS。逐 job URL、时间与结论见 `post_approval_hosted_checks.json`。该 PASS 只验证 session-branch integration，不执行也不授权合并 PR #22。

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
workspace_branch / clean_at_start / permission: arena/9fa6b0dd-hydrogym / true / session-branch-write; owner-approval-observed
conflicted_pinned_files: 4 (exact paths in pr22_quality_overlap_matrix.json)
raw_ruff / raw_format / raw_isort / raw_codespell: RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(1) / RAW_TOOL_FAIL(65)
ratchet_historical / added / removed / moved / approved: 171+27+28+3 / 0 / 0 / 0 / 4; pending_owner=0
provenance_and_sha_checks: PASS for frozen sources; APPROVED_EXACT_TRANSITIONS for four overlap pairs
operator_identity / frozen_source_tests: owner=licy0505; agent=Arena session / focused=49 PASS; fixture=28 PASS; numerical=PASS
quality_gate / uv_audit / two_phase_e2e / non_slow_pytest: LOCAL-PASS-BASELINE-ONLY / local-exact-ENVIRONMENT-BLOCKED; hosted-python-ci-PASS / hosted-main-PASS / 228 PASS
new_tests / adversarial_tests: 12 new QUALITY-3 tests / 31 total PASS
evidence_manifest_path / SHA: examples/two_phase/evidence/quality3/manifest.json / 505b161c743a90a418c0c4a840500597924f3ee9f4b6ee8129769e96dd531cae
post_approval_hosted_run / head / conclusion: 38053658694 / 7a0284c438d37f0015f079a7869c3ea640c6e9cb / SUCCESS
PR23_merge_eligibility / PR22_merge_eligibility: ALREADY_MERGED_BY_OWNER_WITH_PAGES_FAILURE_UNCLASSIFIED / NOT_ELIGIBLE; integration-validated-but-PR22-merge-blocked
SOLVER_CONTRACT_VERSION=12
L1A_STATUS=BLOCKED
L1B_DATA_NOT_READY
L1A-2t=NOT_RUN
DECISION: QUALITY3_INTEGRATION_VALIDATED_PR22_MERGE_BLOCKED
EXACTLY_ONE_NEXT_ACTION: No merge action is authorized or executed by this agent; retain PR #22 OPEN pending separate merge authorization and current PR-head policy checks.
```

## Decision

主 verdict 现在是 `QUALITY3_INTEGRATION_VALIDATED_PR22_MERGE_BLOCKED`。四项 named exact SHA transitions 已得到仓库所有者批准，四路 ratchet 已通过且没有新增债务；批准后的 exact integrated commit `7a0284c438d37f0015f079a7869c3ea640c6e9cb` 的 hosted Python CI run `38053658694` 八个 job 全部 SUCCESS，包含 hosted `uv audit --locked`。这验证了 QUALITY-3 integration，但不改变 PR #22 仍 OPEN、其自身观察到的 head Quality 失败、以及本轮不 merge 的决定。L1B 仍为 `L1B_DATA_NOT_READY`。
