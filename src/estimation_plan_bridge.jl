"""Execute a CausalDynamics `EstimationPlan` on a wide panel."""

"""
    run_estimation_plan(df, plan; arm_hi, arm_ref, levels, policy, kwargs...) -> NamedTuple

Run the CausalTargeted engine named in `plan` using column names from
[`plan_targeted_estimation`](@ref).

For `:discrete_lmtp`, pass `arm_hi`, `arm_ref`, and `levels` to call
[`run_discrete_lmtp_contrast`](@ref), or pass `policy` for a single
[`run_discrete_lmtp`](@ref) fit.

For `:two_part_discrete_lmtp`, pass `arm_hi`, `arm_ref`, and `levels`; uses
`plan.presence_col` and `plan.intensity_col`.

Remaining `kwargs` are forwarded to the runner (e.g. `folds`, `rng`,
`learners_outcome`, `cluster`). When omitted, [`recommend_run_options`](@ref)
supplies defaults from `nrow(df)` and `plan.engine`.
"""
function run_estimation_plan(
    df::DataFrame,
    est_plan::CausalDynamics.EstimationPlan;
    arm_hi = nothing,
    arm_ref = nothing,
    levels = nothing,
    policy = nothing,
    folds = nothing,
    learners_outcome = nothing,
    rng = nothing,
    kwargs...,
)
    if !est_plan.identifiable
        @warn("EstimationPlan marked not graphically identifiable: $est_plan")
    end
    if est_plan.estimability === :underpowered
        @warn("Empirical support below threshold: min_complete_n=$(est_plan.min_complete_n)")
    elseif est_plan.estimability === :empty
        throw(ArgumentError(
            "EstimationPlan has no complete cases across analysis columns " *
            "(min_complete_n=$(something(est_plan.min_complete_n, 0)))",
        ))
    end
    if !isempty(est_plan.missing_columns)
        @warn("Adjustment columns missing from panel: $(est_plan.missing_columns)")
    end

    opts = recommend_run_options(nrow(df); engine = est_plan.engine)
    folds = something(folds, opts.folds)
    learners_outcome = something(learners_outcome, opts.learners_outcome)
    rng = something(rng, StableRNG(1))
    shared = (;
        baseline = est_plan.baseline,
        folds,
        learners_outcome,
        rng,
    )

    if est_plan.engine === :discrete_lmtp
        if arm_hi !== nothing && arm_ref !== nothing
            levels === nothing && throw(ArgumentError(
                "run_estimation_plan for :discrete_lmtp contrast requires levels",
            ))
            return run_discrete_lmtp_contrast(
                df, est_plan.treatment, est_plan.outcome;
                arm_hi = arm_hi,
                arm_ref = arm_ref,
                levels = levels,
                shared...,
                kwargs...,
            )
        elseif policy !== nothing
            return run_discrete_lmtp(
                df, est_plan.treatment, est_plan.outcome;
                policy = policy,
                shared...,
                kwargs...,
            )
        else
            throw(ArgumentError(
                "run_estimation_plan for :discrete_lmtp requires arm_hi/arm_ref/levels " *
                "or policy",
            ))
        end
    elseif est_plan.engine === :two_part_discrete_lmtp
        est_plan.presence_col === nothing && throw(ArgumentError(
            "two-part plan missing presence_col; set OutcomeKind.hurdle in outcome_specs",
        ))
        est_plan.intensity_col === nothing && throw(ArgumentError(
            "two-part plan missing intensity_col; set OutcomeKind.hurdle in outcome_specs",
        ))
        arm_hi === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        arm_ref === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        levels === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        return run_two_part_discrete_lmtp_contrast(
            df, est_plan.treatment;
            presence = est_plan.presence_col,
            intensity = est_plan.intensity_col,
            arm_hi = arm_hi,
            arm_ref = arm_ref,
            levels = levels,
            shared...,
            kwargs...,
        )
    elseif est_plan.engine === :sequential_lmtp
        throw(ArgumentError(
            "run_estimation_plan does not run :sequential_lmtp; use " *
            "sequential_spec_from_identification + run_sequential_lmtp",
        ))
    elseif est_plan.engine === :lmtp_grid
        throw(ArgumentError(
            "run_estimation_plan does not run :lmtp_grid; use run_lmtp_grid with " *
            "est_plan.treatment, est_plan.outcome, and est_plan.baseline",
        ))
    else
        throw(ArgumentError("unsupported EstimationPlan engine=$(est_plan.engine)"))
    end
end

export run_estimation_plan
