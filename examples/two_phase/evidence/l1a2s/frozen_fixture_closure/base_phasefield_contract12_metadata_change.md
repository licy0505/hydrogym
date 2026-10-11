# Base `phasefield.py` digest delta

- Contract-12 promotion source: `165608c463783db483f9ee8f8cdc3b869c55bcb2`, whole-file SHA-256 `ebb249a22fa2065fa3dacb0a166289220ded6c82e4aeea6eaab533ff39e4c03b`.
- Lineage-metadata follow-up: `7cf11e7155d0767c5f54c08ca9ee4b1d1e598567`, whole-file SHA-256 `024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c`.
- PR base: `002120f3a051e638a7e85f9db4022107dbe45780`, same whole-file SHA-256 `024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c`.

The only `phasefield.py` hunk from the contract-12 promotion to the PR base adds the contract-12 lineage/policy documentation and the three `SOLVER_CONTRACT_12_*` metadata constants immediately after `SOLVER_CONTRACT_VERSION = 12`. It records `impact_phase_cap_dx2_v1`, the effective-dt rule, and the declared distinction between unchanged CH/NS operator formulas and changed production trajectory semantics. No function or operator implementation changed in that delta. The independent operator-source hash regression is recorded in `base_operator_hash_regression.txt`.
