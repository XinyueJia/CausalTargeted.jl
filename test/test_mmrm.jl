# SPDX-License-Identifier: MIT

using CategoricalArrays
using DataFrames
using LinearAlgebra
using MixedModels
using StableRNGs
using Statistics

@testset "MMRM (#25)" begin
    rng = StableRNG(20260830)
    n_units = 100
    visits = [0.0, 1.0, 2.0, 3.0]
    β_a = 0.9
    β_at = 0.55

    rows = NamedTuple[]
    for unit in 1:n_units
        W = randn(rng)
        A = Float64(W + 0.7randn(rng) > 0)
        b = 1.0randn(rng)
        for visit in visits
            Y = 1.5 + β_a * A + 0.3visit + β_at * A * visit + 0.8W + b + 0.4randn(rng)
            push!(rows, (; id = unit, arm = A, visit, W, Y))
        end
    end
    data = DataFrame(rows)

    @testset "random intercept MMRM" begin
        result = run_mmrm(
            data;
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
            baseline = [:W],
            values = (0.0, 1.0),
        )
        @test result.covariance == :random_intercept
        @test !result.time_categorical
        @test result.contrast.times == visits
        known = β_a .+ β_at .* visits
        @test maximum(abs.(result.contrast.effect .- known)) < 0.45
        @test result.contrast.vcov !== nothing
        @test all(isfinite, result.contrast.se)
        @test occursin("MMRMResult", sprint(show, result))
    end

    @testset "unstructured (random-effects) MMRM" begin
        result = run_mmrm(
            data;
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
            baseline = [:W],
            covariance = :unstructured,
            values = (0.0, 1.0),
        )
        @test result.covariance == :unstructured
        @test result.time_categorical
        @test length(result.contrast.times) == length(visits)
        @test all(isfinite, result.contrast.effect)
        re_terms = [
            term for term in formula(result.model).rhs
            if term isa MixedModels.RandomEffectsTerm
        ]
        @test length(re_terms) == 1
        random_names = MixedModels.StatsModels.coefnames(only(re_terms).lhs)
        @test length(random_names) > 1
    end

    @testset "fit_mmrm + manual g-comp matches run_mmrm" begin
        model = fit_mmrm(
            data;
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
            baseline = [:W],
        )
        manual = mixed_g_computation(
            model,
            data;
            treatment = :arm,
            outcome = :Y,
            time = :visit,
            id = :id,
            values = (0.0, 1.0),
        )
        wrapped = run_mmrm(
            data;
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
            baseline = [:W],
            values = (0.0, 1.0),
        )
        @test wrapped.contrast.effect ≈ manual.effect rtol = 1.0e-10
        @test wrapped.contrast.mean_reference ≈ manual.mean_reference rtol = 1.0e-10
    end

    @testset "validation" begin
        @test_throws ArgumentError run_mmrm(
            DataFrame(id = Int[], arm = Float64[], visit = Float64[], Y = Float64[]);
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
        )
        @test_throws ArgumentError fit_mmrm(
            data;
            outcome = :Y,
            treatment = :arm,
            time = :visit,
            id = :id,
            covariance = :compound_symmetry,
        )
    end
end
