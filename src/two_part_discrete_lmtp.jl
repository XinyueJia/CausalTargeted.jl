"""Two-part hurdle LMTP: binomial presence + Gaussian intensity among positives."""

"""
    TwoPartInterventionalMean(trt, presence, intensity, adjustment, policy)

Hurdle estimand pairing a binary presence outcome with a continuous intensity
outcome observed only when presence is positive. Presence is estimated with
[`run_discrete_lmtp`](@ref) (`family_outcome=:binomial`); intensity is estimated
on the subsample with observed positive intensity (conditional on ``Y_{\\mathrm{pres}} > 0``).
"""
struct TwoPartInterventionalMean <: Estimand
    trt::Symbol
    presence::Symbol
    intensity::Symbol
    adjustment::Vector{Symbol}
    policy::DiscreteTreatmentPolicy
end

estimand_engine(::TwoPartInterventionalMean) = :two_part_discrete_lmtp

function _positive_intensity_rows(df::DataFrame, presence::Symbol, intensity::Symbol)
    pres = df[!, presence]
    mask = BitVector(undef, nrow(df))
    @inbounds for i in eachindex(pres)
        p = pres[i]
        mask[i] = !ismissing(p) && coalesce(p, 0) > 0
    end
    return df[mask, :]
end

"""
    run_two_part_discrete_lmtp(
        df, trt;
        presence, intensity, policy, baseline, folds, rng, ...
    ) -> NamedTuple

Fit presence and conditional-intensity discrete LMTPs under one
[`DiscreteTreatmentPolicy`](@ref). The intensity part conditions on observed
positive presence (`presence > 0`); the estimand is
``E[Y_{\\mathrm{int}} \\mid Y_{\\mathrm{pres}} > 0, \\mathrm{do}(A)]``.
"""
function run_two_part_discrete_lmtp(
    df::DataFrame,
    trt::Symbol;
    presence::Symbol,
    intensity::Symbol,
    policy::DiscreteTreatmentPolicy,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_presence::Symbol = :binomial,
    family_intensity::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    density_ratio::Symbol = :classification,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    epochs::Int = 3,
    handle_missing::Symbol = :drop,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
)
    shared = (;
        baseline, folds, rng, learners_outcome, learners_trt,
        density_ratio, estimator, trunc, epochs, handle_missing, cluster,
    )
    pres = run_discrete_lmtp(
        df, trt, presence;
        policy = policy,
        family_outcome = family_presence,
        shared...,
    )
    int_df = _positive_intensity_rows(df, presence, intensity)
    intens = run_discrete_lmtp(
        int_df, trt, intensity;
        policy = policy,
        family_outcome = family_intensity,
        shared...,
    )
    return (;
        presence = pres,
        intensity = intens,
        n_presence = nrow(df),
        n_intensity = nrow(int_df),
    )
end

"""
    run_two_part_discrete_lmtp_contrast(
        df, trt;
        presence, intensity, arm_hi, arm_ref, levels, kwargs...
    ) -> NamedTuple

Arm contrast for a two-part hurdle outcome. Returns `.presence` and
`.intensity` summaries (`estimate`, `se`, `lower`, `upper`, `n`). The
intensity contrast is conditional on ``Y_{\\mathrm{pres}} > 0`` among rows
retained after missing-data handling on the full panel.
"""
function run_two_part_discrete_lmtp_contrast(
    df::DataFrame,
    trt::Symbol;
    presence::Symbol,
    intensity::Symbol,
    arm_hi,
    arm_ref,
    levels,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_presence::Symbol = :binomial,
    family_intensity::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    density_ratio::Symbol = :classification,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    epochs::Int = 3,
    handle_missing::Symbol = :drop,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
)
    levels_vec = collect(levels)
    pol_hi = discrete_static_policy(arm_hi; levels = levels_vec)
    pol_ref = discrete_static_policy(arm_ref; levels = levels_vec)
    shared = (;
        baseline, folds, rng, learners_outcome, learners_trt,
        density_ratio, estimator, trunc, epochs, handle_missing, cluster,
    )
    pres_hi = run_discrete_lmtp(
        df, trt, presence;
        policy = pol_hi,
        family_outcome = family_presence,
        shared...,
    )
    pres_ref = run_discrete_lmtp(
        df, trt, presence;
        policy = pol_ref,
        family_outcome = family_presence,
        shared...,
    )
    int_df = _positive_intensity_rows(df, presence, intensity)
    int_hi = run_discrete_lmtp(
        int_df, trt, intensity;
        policy = pol_hi,
        family_outcome = family_intensity,
        shared...,
    )
    int_ref = run_discrete_lmtp(
        int_df, trt, intensity;
        policy = pol_ref,
        family_outcome = family_intensity,
        shared...,
    )
    pres_est = pres_hi.estimate - pres_ref.estimate
    pres_se = sqrt(pres_hi.se^2 + pres_ref.se^2)
    pres_lwr, pres_upr = wald_ci(pres_est, pres_se)
    int_est = int_hi.estimate - int_ref.estimate
    int_se = sqrt(int_hi.se^2 + int_ref.se^2)
    int_lwr, int_upr = wald_ci(int_est, int_se)
    return (;
        presence = (;
            estimate = pres_est,
            se = pres_se,
            lower = pres_lwr,
            upper = pres_upr,
            n = nrow(df),
            contrast = string(arm_hi, "_vs_", arm_ref),
            hi = pres_hi,
            ref = pres_ref,
        ),
        intensity = (;
            estimate = int_est,
            se = int_se,
            lower = int_lwr,
            upper = int_upr,
            n = nrow(int_df),
            contrast = string(arm_hi, "_vs_", arm_ref),
            hi = int_hi,
            ref = int_ref,
            conditional_on = presence,
        ),
        arm_hi = arm_hi,
        arm_ref = arm_ref,
        contrast = string(arm_hi, "_vs_", arm_ref),
    )
end

export TwoPartInterventionalMean
export run_two_part_discrete_lmtp, run_two_part_discrete_lmtp_contrast
