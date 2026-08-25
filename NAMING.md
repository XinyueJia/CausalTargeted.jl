# Naming scheme (owned causal stack)

## Packages (keep)

| Package | Role | Why the name |
|---------|------|----------------|
| **CausalDynamics.jl** | Graphs, identification, SCMs / CDMs, `do(·)` | Dynamics = structural + temporal generative layer |
| **CausalTargeted.jl** | Cross-fitted estimation (LMTP, mediation, certificates) | Targeted = targeted learning / TMLE lineage |
| **DAGMakie.jl** | DAG figures only | Makie backend for DAGs; no identification |

Applications (e.g. SheepVaccineCDCS) stay thin: loaders, registry TOML, concordance with R notebooks.

## Estimation engines (Julia vocabulary)

Prefer **method names**, not R package nicknames:

| Engine symbol | Public APIs | Meaning |
|---------------|-------------|---------|
| `:lmtp` | `run_lmtp_grid`, `InterventionalMean` | Longitudinal / continuous modified treatment policies |
| `:discrete_lmtp` | `run_discrete_lmtp`, `DiscreteInterventionalMean` | Categorical-treatment LMTP (classification density ratios) |
| `:mediation` | `run_mediation_grid`, `run_mediation_scalar`, `MediationContrast` | Interventional TE / NDE / NIE under MTP shifts |
| `:sequential_lmtp` | `run_sequential_lmtp` | Multi-time sequential regression (numeric shift or factor `policies`) |
| `:survival_lmtp` | `run_survival_lmtp`, `SurvivalPolicy` | Discrete-time event-time / survival LMTP |
| `:repeated_msm` | `run_repeated_outcome_msm`, `RepeatedOutcomeMSM`, `msm_contrast` | Binary point treatment, repeated outcomes, joint IF ``Σ``; optional `cluster=` sandwich |
| `:parametric_msm` | `run_parametric_repeated_msm`, `ParametricRepeatedOutcomeMSM` | GLS projection onto treatment×time MSM designs; forwards `cluster=` |
| `:scalar` | `ScalarMediation` | Single-contrast mediation helpers |

## Legacy R concordance (`crumble`)

The R package [`crumble`](https://cran.r-project.org/package=crumble) (Liu et al.) inspired the mediation grid. In Julia we **do not** use “crumble” as the primary name:

- Prefer `:mediation`, `run_mediation_grid`, `MediationFoldCache`, `run_mediation_scalar_ppl`, …
- Soft-deprecated aliases (emit `DeprecationWarning`): `:crumble` → `:mediation` via `normalize_engine`; `run_crumble_grid` → `run_mediation_grid`; `run_crumble_scalar` → `run_mediation_scalar`; `run_crumble_scalar_ppl` → `run_mediation_scalar_ppl`; `build_crumble_fold_cache` / `CrumbleFoldCache` → mediation names.
- Sheep TOML may still say `engine = "crumble"` for R-registry parity; loaders normalise on read.

Cite the papers (`liu2024mediation`, `liu2025crumble`, `diaz2020mediation`); do not treat the R package name as the Julia API brand.

## Synthetic DGPs

Exported for book / README examples: `simulate_linear_mtp`, `simulate_mediation`,
`simulate_discrete_survival_mtp`, `simulate_mixed_baseline_mtp`,
`simulate_binomial_mtp`, `simulate_multinomial_outcome`,
`simulate_categorical_treatment_mtp`, `simulate_sequential_factor_mtp`,
`simulate_repeated_outcome_ate`.
Other scenario builders and `run_julia_synthetic_once` remain in-module (qualified as `CausalTargeted.…`) for package tests and `scripts/synthetic_benchmark/`.
