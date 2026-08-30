using CausalTargeted
using Distributions
using StableRNGs
using Test

@testset "family_outcome heuristics (#34, #36)" begin
    @testset "suggest_family_outcome" begin
        @test suggest_family_outcome([0, 1, 0, 1]) === :binomial
        @test suggest_family_outcome([0.0, 1.0, 0.5]) === :gaussian
        counts = [0, 2, 5, 1, 8, 3, 12, 0, 4]
        @test suggest_family_outcome(counts) in (:poisson, :negbin)
        rng_over = StableRNG(9)
        over = [rand(rng_over, NegativeBinomial(0.8, 0.12)) for _ in 1:100]
        @test maximum(over) > 1
        @test suggest_family_outcome(over) === :negbin
    end

    @testset "validate_family_outcome" begin
        validate_family_outcome([0.0, 2.0, 5.0], :negbin)
        @test_throws ArgumentError validate_family_outcome([0.0, -1.0, 2.0], :poisson)
        @test_throws ArgumentError validate_family_outcome([0.0, 2.0], :zinb)
    end

    @testset "recommend_run_options" begin
        bin = [0.0, 1.0, 0.0, 1.0]
        opts_bin = recommend_run_options(200; outcome = bin)
        @test opts_bin.family_outcome === :binomial

        counts = [0, 2, 5, 1, 8, 3, 12, 0, 4]
        opts_count = recommend_run_options(200; outcome = counts)
        @test opts_count.family_outcome in (:poisson, :negbin)
        @test :glm_nb in opts_count.learners_outcome || :glm_poisson in opts_count.learners_outcome
    end
end
