"""Point-treatment repeated-outcome MSM with joint influence-function covariance.

Estimates the unstructured profile

```math
\\tau(t) = E[Y_t \\mid do(A=1)] - E[Y_t \\mid do(A=0)]
```

for binary point treatment and several outcomes measured on the same units
(wide layout). Cross-fitted propensity is shared across times; outcome
regressions are fit per ``Y_t``. Unit-level AIPW influence curves are stacked
into an ``n \\times T`` matrix whose empirical covariance yields
``\\widehat{\\Sigma}`` for joint Wald contrasts (Rosenblum & van der Laan 2010
spirit; Julia-native, not an R `tmleMSM` port).

# References

- Rosenblum, M. & van der Laan, M. J. (2010). Targeted Maximum Likelihood
  Estimation of the Parameter of a Marginal Structural Model.
  *The International Journal of Biostatistics*, 6(2).
- Bang & Robins (2005); Chernozhukov et al. (2018) — doubly robust / cross-fit AIPW
"""

using DataFrames
using LinearAlgebra
using Random
using StableRNGs
using Statistics
using Graphs: AbstractGraph
using CausalDynamics: IdentificationResult, TotalEffectQuery, identify

"""
    RepeatedOutcomeMSM(trt, outcomes, adjustment)

Point-treatment profile estimand: binary `trt` on several outcomes measured on
the same units. Engine [`estimand_engine`](@ref) is `:repeated_msm`.
"""
struct RepeatedOutcomeMSM <: Estimand
    trt::Symbol
    outcomes::Vector{Symbol}
    adjustment::Vector{Symbol}
end

RepeatedOutcomeMSM(trt::Symbol, outcomes::AbstractVector{Symbol}, adjustment::AbstractVector{Symbol}) =
    RepeatedOutcomeMSM(trt, collect(Symbol, outcomes), collect(Symbol, adjustment))

estimand_engine(::RepeatedOutcomeMSM) = :repeated_msm

"""
    identify_repeated_outcomes(g, treatment, outcomes; node_names, missingness)
        -> Vector{IdentificationResult}

Identify `TotalEffectQuery(treatment, y)` for each outcome on the same graph.
Useful when all ``Y_t`` share a backdoor set (static ``A``, baseline ``W``).
"""
function identify_repeated_outcomes(
    g::AbstractGraph,
    treatment,
    outcomes::AbstractVector;
    node_names = nothing,
    missingness = nothing,
)
    isempty(outcomes) && throw(ArgumentError("outcomes must be nonempty"))
    return map(outcomes) do y
        q = TotalEffectQuery(treatment, y)
        if missingness === nothing
            identify(g, q; node_names = node_names)
        else
            identify(g, q; node_names = node_names, missingness = missingness)
        end
    end
end

"""
    _crossfit_propensity_folds(df, treatment, covariates, fold_sets; learners, rng)
        -> Vector{Float64}

Cross-fitted propensity on a **shared** fold partition (same folds as outcome fits).
"""
function _crossfit_propensity_folds(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    fold_sets;
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    n = nrow(df)
    a = Float64.(df[!, treatment])
    preds = zeros(n)
    fitted_schema = fit_covariate_schema(df, covariates)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(fitted_schema, train)
        Xte = design_matrix(fitted_schema, test)
        sl = fit_super_learner(
            Xtr, a[train_idx];
            learners = learners,
            family = :binomial,
            metalearner = :invmse,
            rng = rng,
        )
        raw = predict_super_learner(sl, Xte)
        preds[test_idx] = clamp.(raw, 1e-3, 1 - 1e-3)
    end
    return preds
end

"""
    _aipw_influence_matrix(...) -> (psi, estimates, psi_μ1, μ1, psi_μ0, μ0)

Cross-fitted AIPW influence curves for each outcome (columns), sharing folds and
propensity `g_hat`. Returns uncentred plug-in+weighting curves for ``τ``,
``μ(1)``, and ``μ(0)``, with Hajek means.
"""
function _aipw_influence_matrix(
    df::DataFrame,
    treatment::Symbol,
    outcomes::Vector{Symbol},
    adj_covars::Vector{Symbol},
    fold_sets,
    g_hat::AbstractVector{<:Real},
    ipcw_w::AbstractVector{<:Real};
    learners = DEFAULT_SL_LEARNERS,
    rng = StableRNG(1),
    estimator::Symbol = :tmle,
)
    estimator in (:tmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "estimator must be :tmle or :eif (aliases :aipw, :sdr); got :$estimator",
    ))
    n = nrow(df)
    T = length(outcomes)
    length(g_hat) == n || throw(ArgumentError("g_hat length must match nrow(df)"))
    length(ipcw_w) == n || throw(ArgumentError("ipcw_w length must match nrow(df)"))
    a = Float64.(df[!, treatment])
    all(x -> x == 0.0 || x == 1.0, a) || throw(ArgumentError(
        "run_repeated_outcome_msm requires a binary 0/1 treatment column",
    ))

    g = Float64.(g_hat)
    H = @. a / g - (1 - a) / (1 - g)
    H1 = 1.0 ./ g
    H0 = -1.0 ./ (1.0 .- g)
    covariate_schema = fit_covariate_schema(df, adj_covars)
    psi = zeros(n, T)
    psi_μ1 = zeros(n, T)
    psi_μ0 = zeros(n, T)

    for (j, outcome) in enumerate(outcomes)
        y = Float64.(df[!, outcome])
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            train = df[train_idx, :]
            test = df[test_idx, :]
            sl = _fit_sl_outcome(
                train, adj_covars, y[train_idx];
                treatment = treatment,
                learners = learners,
                rng = rng,
                schema = covariate_schema,
            )
            Q_obs = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = a[test_idx],
            )
            Q1 = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = ones(length(test_idx)),
            )
            Q0 = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = zeros(length(test_idx)),
            )
            Ht = H[test_idx]
            if estimator === :tmle
                den = sum(abs2, Ht)
                ε = den > 1e-12 ? sum(Ht .* (y[test_idx] .- Q_obs)) / den : 0.0
                Q_obs = Q_obs .+ ε .* Ht
                Q1 = Q1 .+ ε .* H1[test_idx]
                Q0 = Q0 .+ ε .* H0[test_idx]
            end
            resid = y[test_idx] .- Q_obs
            psi[test_idx, j] = Ht .* resid .+ (Q1 .- Q0)
            # μ(1): A/g * resid + Q1; μ(0): (1-A)/(1-g) * resid + Q0
            # H1 = 1/g, H0 = -1/(1-g) ⇒ (1-A)/(1-g) = -(1-A)*H0
            psi_μ1[test_idx, j] = (a[test_idx] .* H1[test_idx]) .* resid .+ Q1
            psi_μ0[test_idx, j] = ((1 .- a[test_idx]) .* (-H0[test_idx])) .* resid .+ Q0
        end
    end

    estimates = Vector{Float64}(undef, T)
    μ1 = Vector{Float64}(undef, T)
    μ0 = Vector{Float64}(undef, T)
    for j in 1:T
        estimates[j] = transport_weighted_mean(psi[:, j], ipcw_w)
        μ1[j] = transport_weighted_mean(psi_μ1[:, j], ipcw_w)
        μ0[j] = transport_weighted_mean(psi_μ0[:, j], ipcw_w)
    end
    return psi, estimates, psi_μ1, μ1, psi_μ0, μ0
end

"""
    _msm_covariance(psi, estimates, ipcw_w) -> (Σ, se, ic)

Hajek-centred influence matrix and ``\\widehat{\\Sigma} = n^{-1}\\mathrm{Cov}(D_i)``.
"""
function _msm_covariance(
    psi::AbstractMatrix{<:Real},
    estimates::AbstractVector{<:Real},
    ipcw_w::AbstractVector{<:Real},
)
    n, T = size(psi)
    length(estimates) == T || throw(ArgumentError("estimates length must equal ncol(psi)"))
    length(ipcw_w) == n || throw(ArgumentError("ipcw_w length must match nrow(psi)"))
    w = Float64.(ipcw_w)
    sw = sum(w)
    sw ≈ 0 && throw(ArgumentError("weights sum to zero"))
    w_bar = sw / n
    ic = Matrix{Float64}(undef, n, T)
    for j in 1:T
        ic[:, j] = (w ./ w_bar) .* (Float64.(psi[:, j]) .- estimates[j])
    end
    Σ = (ic' * ic) ./ n^2
    # Symmetrise numerical noise
    Σ = Symmetric(0.5 .* (Σ .+ Σ'))
    se = sqrt.(diag(Σ))
    return Matrix{Float64}(Σ), se, ic
end

"""
    _complete_profile_R(df, outcomes) -> Vector{Float64}

Response indicator for a complete outcome profile (all ``Y_t`` observed).
"""
function _complete_profile_R(df::DataFrame, outcomes::Vector{Symbol})
    n = nrow(df)
    R = ones(n)
    for y in outcomes
        R .*= Float64.(.!ismissing.(df[!, y]))
    end
    return R
end

"""
    _ipcw_weights_from_R(df, R, covariates; learners, rng) -> Vector{Float64}

Stabilised IPCW from a 0/1 response vector `R` (units with `R=0` get weight 0).
"""
function _ipcw_weights_from_R(
    df::DataFrame,
    R::AbstractVector{<:Real},
    covariates::Vector{Symbol};
    learners = (:logistic, :mean),
    rng::AbstractRNG = StableRNG(1),
)
    n = nrow(df)
    length(R) == n || throw(ArgumentError("R length must match nrow(df)"))
    Rv = Float64.(R)
    marginal_p = mean(Rv)
    if marginal_p >= 1.0 - 1e-10 || isempty(covariates)
        w = ones(n)
        w[Rv .== 0.0] .= 0.0
        return w
    end
    X = design_matrix(df, covariates)
    sl = fit_super_learner(
        X, Rv;
        learners = learners, family = :binomial, metalearner = :invmse, rng = rng,
    )
    p_obs = clamp.(predict_super_learner(sl, X), 1e-3, 1.0 - 1e-3)
    w = marginal_p ./ p_obs
    w[Rv .== 0.0] .= 0.0
    return w
end

"""
    _handle_missing_repeated_outcomes(df, treatment, outcomes, baseline, strategy; rng)
        -> MissingDataResult

Observable policy for a **shared** analysis sample across all outcomes.

- `:drop` — complete cases on treatment, baseline, and every ``Y_t``
- `:ipcw` — drop incomplete treatment/baseline; IPCW on the complete-profile
  indicator; keep complete-profile rows
- `:impute` — mean-impute baseline only; drop missing treatment or any ``Y_t``
- `:ipcw_impute` — impute baseline, then profile IPCW

Outcomes are never imputed (fills would enter the IF).
"""
function _handle_missing_repeated_outcomes(
    df::DataFrame,
    treatment::Symbol,
    outcomes::Vector{Symbol},
    baseline::Vector{Symbol},
    strategy::Symbol;
    rng::AbstractRNG = StableRNG(1),
)
    n_in = nrow(df)
    extra_cols = Symbol[]
    baseline = unique(baseline)
    required = unique(vcat([treatment], baseline, outcomes))
    rates = _column_miss_rates(df, required)

    if strategy == :drop
        df_clean = dropmissing(df, required)
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, ones(nrow(df_clean)), extra_cols, meta)
    elseif strategy == :ipcw
        df_cc = dropmissing(df, unique(vcat([treatment], baseline)))
        R = _complete_profile_R(df_cc, outcomes)
        w = _ipcw_weights_from_R(df_cc, R, baseline; rng = rng)
        keep = R .== 1.0
        df_clean = df_cc[keep, :]
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, w[keep], extra_cols, meta)
    elseif strategy == :impute
        df_copy = copy(df)
        extra_cols = impute_covariates_mean!(df_copy, baseline)
        df_clean = dropmissing(df_copy, unique(vcat([treatment], outcomes)))
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, ones(nrow(df_clean)), extra_cols, meta)
    elseif strategy == :ipcw_impute
        df_copy = copy(df)
        extra_cols = impute_covariates_mean!(df_copy, baseline)
        df_cc = dropmissing(df_copy, [treatment])
        R = _complete_profile_R(df_cc, outcomes)
        covars = unique(vcat(baseline, extra_cols))
        w = _ipcw_weights_from_R(df_cc, R, covars; rng = rng)
        keep = R .== 1.0
        df_clean = df_cc[keep, :]
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, w[keep], extra_cols, meta)
    else
        error("Unknown missing-data strategy: $strategy. Use :drop, :ipcw, :impute, or :ipcw_impute.")
    end
end

"""
    unstack_repeated_outcomes(long; id, time, outcome, treatment, covariates, prefix)
        -> DataFrame

Pivot a long table (`id`, `time`, `Y`) to wide `Y1…YT` with unit-constant
`treatment` and `covariates`. Times are sorted; columns are `prefix` * `1:T`.
"""
function unstack_repeated_outcomes(
    long::DataFrame;
    id::Symbol,
    time::Symbol,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol} = Symbol[],
    prefix::AbstractString = "Y",
)
    required = unique(vcat([id, time, outcome, treatment], covariates))
    for c in required
        hasproperty(long, c) || throw(ArgumentError("long table missing column :$c"))
    end
    times = sort(unique(long[!, time]))
    T = length(times)
    T >= 1 || throw(ArgumentError("no time levels in :$time"))
    time_index = Dict(t => i for (i, t) in enumerate(times))
    ynames = [Symbol(prefix, i) for i in 1:T]
    records = Dict{Symbol, Any}[]
    for sub in groupby(long, id)
        nrow(sub) == T || throw(ArgumentError(
            "unit $(sub[1, id]) has $(nrow(sub)) rows; expected $T distinct times",
        ))
        a0 = sub[1, treatment]
        all(==(a0), sub[!, treatment]) || throw(ArgumentError(
            "treatment :$treatment is not constant within unit $(sub[1, id])",
        ))
        rec = Dict{Symbol, Any}(treatment => a0)
        for c in covariates
            v0 = sub[1, c]
            all(==(v0), sub[!, c]) || throw(ArgumentError(
                "covariate :$c is not constant within unit $(sub[1, id])",
            ))
            rec[c] = v0
        end
        y = Vector{Any}(undef, T)
        fill!(y, missing)
        for r in eachrow(sub)
            y[time_index[r[time]]] = r[outcome]
        end
        for i in 1:T
            rec[ynames[i]] = y[i]
        end
        push!(records, rec)
    end
    wide = DataFrame(records)
    order = unique(vcat([treatment], covariates, ynames))
    return select(wide, order)
end

"""
    run_repeated_outcome_msm(df, treatment, outcomes; baseline, folds, learners, rng,
                             handle_missing, estimator) -> NamedTuple

Estimate ``\\tau(t)`` for binary point treatment and several outcomes with a
**joint** influence-function covariance.

Returns `(estimates, se, covariance, ic, outcomes, n, positivity, missingness)`.
Use [`msm_contrast`](@ref) for linear forms ``c^\\top\\tau`` (e.g. ``\\tau(t_3)-\\tau(t_2)``).

Missingness uses a **shared** sample across times (complete-profile ``R``).
`estimator=:tmle` (default) applies a per-fold clever-covariate fluctuation;
`:eif` is the untargeted AIPW one-step.
"""
function run_repeated_outcome_msm(
    df::DataFrame,
    treatment::Symbol,
    outcomes::AbstractVector{Symbol};
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    learners = DEFAULT_SL_LEARNERS,
    learners_trt = (:logistic, :mean),
    rng::AbstractRNG = StableRNG(1),
    handle_missing::Symbol = :drop,
    estimator::Symbol = :tmle,
)
    validate_contrast_learners(learners; context = "run_repeated_outcome_msm")
    outs = collect(Symbol, outcomes)
    isempty(outs) && throw(ArgumentError("outcomes must be nonempty"))
    allunique(outs) || throw(ArgumentError("outcome symbols must be unique"))

    miss = _handle_missing_repeated_outcomes(
        df, treatment, outs, baseline, handle_missing; rng = rng,
    )
    df_clean, ipcw_w, extra_cols = miss
    adj_covars = vcat(baseline, extra_cols)

    a = Float64.(df_clean[!, treatment])
    all(x -> x == 0.0 || x == 1.0, a) || throw(ArgumentError(
        "run_repeated_outcome_msm requires a binary 0/1 treatment column",
    ))

    n = nrow(df_clean)
    n >= folds || throw(ArgumentError(
        "need at least $folds complete-profile rows after handle_missing=:$handle_missing; got $n",
    ))
    fold_sets = crossfit_indices(n, folds, rng)
    g_hat = _crossfit_propensity_folds(
        df_clean, treatment, adj_covars, fold_sets;
        learners = learners_trt, rng = rng,
    )
    psi, estimates, psi_μ1, μ1, psi_μ0, μ0 = _aipw_influence_matrix(
        df_clean, treatment, outs, adj_covars, fold_sets, g_hat, ipcw_w;
        learners = learners, rng = rng, estimator = estimator,
    )
    Σ, se, ic = _msm_covariance(psi, estimates, ipcw_w)
    # Keep μ components available to parametric MSM (not exported on the public NT).
    Σ_μ1, _, ic_μ1 = _msm_covariance(psi_μ1, μ1, ipcw_w)
    Σ_μ0, _, ic_μ0 = _msm_covariance(psi_μ0, μ0, ipcw_w)

    min_g = minimum(g_hat)
    max_g = maximum(g_hat)
    positivity = (
        ok = min_g > 1e-3 && max_g < 1 - 1e-3,
        min_propensity = min_g,
        max_propensity = max_g,
        mean_propensity = mean(g_hat),
    )

    return with_missingness((
        estimates = estimates,
        se = se,
        covariance = Σ,
        ic = ic,
        mu1 = μ1,
        mu0 = μ0,
        covariance_mu1 = Σ_μ1,
        covariance_mu0 = Σ_μ0,
        ic_mu1 = ic_μ1,
        ic_mu0 = ic_μ0,
        outcomes = outs,
        n = n,
        positivity = positivity,
    ), miss.meta)
end

"""
    msm_contrast(result, i, j) -> NamedTuple

Wald inference for ``\\tau_i - \\tau_j`` using the joint covariance in `result`.
"""
function msm_contrast(result::NamedTuple, i::Integer, j::Integer)
    T = length(result.estimates)
    (1 <= i <= T && 1 <= j <= T) || throw(BoundsError(result.estimates, (i, j)))
    c = zeros(T)
    c[i] = 1.0
    c[j] = -1.0
    return msm_contrast(result, c)
end

"""
    msm_contrast(result, c) -> NamedTuple

Wald inference for the linear form ``c^\\top\\tau`` with
``\\mathrm{Var}=c^\\top\\Sigma c``.
"""
function msm_contrast(result::NamedTuple, c::AbstractVector{<:Real})
    θ = haskey(result, :coefficients) ? result.coefficients : result.estimates
    Σ = result.covariance
    length(c) == length(θ) || throw(ArgumentError(
        "contrast length $(length(c)) must match parameter length $(length(θ))",
    ))
    c64 = Float64.(c)
    est = dot(c64, θ)
    var = dot(c64, Σ * c64)
    var < 0 && var > -1e-12 && (var = 0.0)
    var >= 0 || throw(ArgumentError("contrast variance is negative ($(var)); check Σ"))
    se = sqrt(var)
    z = 1.96
    return (
        estimate = est,
        se = se,
        ci_lower = est - z * se,
        ci_upper = est + z * se,
        contrast = c64,
    )
end

export RepeatedOutcomeMSM, run_repeated_outcome_msm, msm_contrast
export identify_repeated_outcomes, unstack_repeated_outcomes
