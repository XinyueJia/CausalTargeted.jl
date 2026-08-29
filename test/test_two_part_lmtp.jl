using CausalTargeted
using DataFrames
using Distributions
using Random
using StableRNGs
using Test

@testset "two-part discrete LMTP" begin
    rng = StableRNG(11)
    n = 500
    arms = ["R", "SS", "SC"]
    A = rand(rng, arms, n)
    W = randn(rng, n)
    p_pres = [a == "SS" ? 0.65 : a == "SC" ? 0.45 : 0.30 for a in A]
    pres = [rand(rng, Bernoulli(p)) for p in p_pres]
    shift = [a == "SS" ? 1.2 : a == "SC" ? 0.4 : 0.0 for a in A]
    intensity = [
        p ? max(0.1, 0.5 + 0.3 * W[i] + shift[i] + 0.2 * randn(rng)) : 0.0
        for (i, p) in enumerate(pres)
    ]
    df = DataFrame(A = A, W = W, Y_pres = Float64.(pres), Y_int = intensity)

    res = run_two_part_discrete_lmtp_contrast(
        df, :A;
        presence = :Y_pres,
        intensity = :Y_int,
        arm_hi = "SS",
        arm_ref = "R",
        levels = arms,
        baseline = [:W],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(12),
    )
    @test isfinite(res.presence.estimate)
    @test isfinite(res.presence.se)
    @test res.presence.n == n
    @test isfinite(res.intensity.estimate)
    @test isfinite(res.intensity.se)
    @test res.intensity.n == count(>(0), df.Y_pres)
    @test res.intensity.conditional_on === :Y_pres
    @test res.contrast == "SS_vs_R"

    policy = discrete_static_policy("SS"; levels = arms)
    single = run_two_part_discrete_lmtp(
        df, :A;
        presence = :Y_pres,
        intensity = :Y_int,
        policy = policy,
        baseline = [:W],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(13),
    )
    @test isfinite(single.presence.estimate)
    @test isfinite(single.intensity.estimate)
    @test single.n_intensity == count(>(0), df.Y_pres)

    estimand = TwoPartInterventionalMean(:A, :Y_pres, :Y_int, [:W], policy)
    @test estimand_engine(estimand) === :two_part_discrete_lmtp
    grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(14), metadata = false)
    @test nrow(grid) == 2
    @test all(isfinite, grid.est)
end
