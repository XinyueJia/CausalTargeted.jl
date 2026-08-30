# SPDX-License-Identifier: MIT

"""Gaussian MMRM (static treatment + repeated visits) for CausalTargetedMixedModelsExt."""

using CategoricalArrays: categorical
using Graphs: AbstractGraph
using StatsModels: @formula

const _MMRM_TIME_COL = :_mmrm_time

function _validate_mmrm_covariance(covariance)
    covariance isa Symbol || throw(ArgumentError(
        "covariance must be :random_intercept or :unstructured; got $(typeof(covariance))",
    ))
    covariance in (:random_intercept, :unstructured) || throw(ArgumentError(
        "covariance must be :random_intercept or :unstructured; got $(repr(covariance))",
    ))
    return covariance
end

function _mmrm_formula_string(
    outcome::Symbol,
    treatment::Symbol,
    time::Symbol,
    id::Symbol,
    covariates::AbstractVector{Symbol},
    covariance::Symbol,
)
    cov = isempty(covariates) ? "" : " + " * join(string.(covariates), " + ")
    re = if covariance == :random_intercept
        "(1 | $id)"
    else
        "(1 + $time | $id)"
    end
    return "$(outcome) ~ 1 + $(treatment) * $(time)$(cov) + $re"
end

function _mmrm_formula(str::AbstractString)
    ex = Meta.parse(str)
    return @eval @formula($ex)
end

function _prepare_mmrm_data(
    data::DataFrame,
    time::Symbol,
    covariance::Symbol,
)
    if covariance == :random_intercept
        return data, time, false
    end
    work = copy(data)
    levels = sort(unique(work[!, time]))
    work[!, _MMRM_TIME_COL] = categorical(
        string.(work[!, time]);
        levels = string.(levels),
        ordered = true,
    )
    return work, _MMRM_TIME_COL, true
end

"""
    fit_mmrm(data; outcome, treatment, time, id, baseline=[], covariance=:random_intercept)

See [`fit_mmrm`](@ref CausalTargeted.fit_mmrm). Remaining keywords forward to
`MixedModels.fit`.
"""
function CausalTargeted.fit_mmrm(
    data::DataFrame;
    outcome::Symbol,
    treatment::Symbol,
    time::Symbol,
    id::Symbol,
    baseline::AbstractVector{Symbol} = Symbol[],
    covariance::Symbol = :random_intercept,
    kwargs...,
)
    nrow(data) > 0 || throw(ArgumentError("data must contain at least one row"))
    covariance = _validate_mmrm_covariance(covariance)
    baseline = collect(Symbol, baseline)
    _validate_columns(data, (outcome, treatment, time, id))
    _validate_columns(data, baseline)
    _validate_complete_finite(data, (outcome, treatment, time, id))
    _validate_complete_finite(data, baseline)
    _validate_static_treatment(data, treatment, id)

    work, time_col, = _prepare_mmrm_data(data, time, covariance)
    formula_str = _mmrm_formula_string(
        outcome, treatment, time_col, id, baseline, covariance,
    )
    formula = _mmrm_formula(formula_str)
    return MixedModels.fit(
        MixedModel,
        formula,
        work;
        progress = false,
        kwargs...,
    )
end

"""
    run_mmrm(data; outcome, treatment, time, id, values=(0, 1)) -> MMRMResult

See [`run_mmrm`](@ref CausalTargeted.run_mmrm). Keywords for
[`mixed_g_computation`](@ref) (`random_effects`, `strata`, …) are explicit;
remaining keywords forward to `MixedModels.fit`.
"""
function CausalTargeted.run_mmrm(
    data::DataFrame;
    outcome::Symbol,
    treatment::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    baseline::AbstractVector{Symbol} = Symbol[],
    covariance::Symbol = :random_intercept,
    random_effects::Symbol = :zero,
    strata = nothing,
    kwargs...,
)
    covariance = _validate_mmrm_covariance(covariance)
    work, time_col, time_categorical = _prepare_mmrm_data(data, time, covariance)
    model = fit_mmrm(
        work;
        outcome,
        treatment,
        time = time_col,
        id,
        baseline,
        covariance,
        kwargs...,
    )
    formula_str = _mmrm_formula_string(
        outcome, treatment, time_col, id, baseline, covariance,
    )
    contrast = mixed_g_computation(
        model,
        work;
        treatment,
        outcome,
        time = time_col,
        id,
        values,
        random_effects,
        strata,
    )
    return MMRMResult(
        model,
        contrast,
        _mmrm_formula(formula_str),
        covariance,
        time_categorical,
    )
end

# Graph-identified adjustment: fit on same data then record adjustment set.
function CausalTargeted.run_mmrm(
    g::Union{CausalGraph, AbstractGraph},
    data::DataFrame;
    outcome::Symbol,
    treatment::Symbol,
    time::Symbol,
    id::Symbol,
    node_names = nothing,
    values = (0, 1),
    baseline::AbstractVector{Symbol} = Symbol[],
    covariance::Symbol = :random_intercept,
    random_effects::Symbol = :zero,
    strata = nothing,
    kwargs...,
)
    covariance = _validate_mmrm_covariance(covariance)
    work, time_col, time_categorical = _prepare_mmrm_data(data, time, covariance)
    model = fit_mmrm(
        work;
        outcome,
        treatment,
        time = time_col,
        id,
        baseline,
        covariance,
        kwargs...,
    )
    formula_str = _mmrm_formula_string(
        outcome, treatment, time_col, id, baseline, covariance,
    )
    if g isa CausalGraph
        contrast = mixed_g_computation(
            g,
            model,
            work;
            treatment,
            outcome,
            time = time_col,
            id,
            values,
            random_effects,
            strata,
            node_names,
        )
    else
        node_names === nothing && throw(ArgumentError(
            "node_names is required when passing a Graphs.jl graph",
        ))
        contrast = mixed_g_computation(
            g,
            model,
            work;
            treatment,
            outcome,
            time = time_col,
            id,
            values,
            random_effects,
            strata,
            node_names,
        )
    end
    return MMRMResult(
        model,
        contrast,
        _mmrm_formula(formula_str),
        covariance,
        time_categorical,
    )
end
