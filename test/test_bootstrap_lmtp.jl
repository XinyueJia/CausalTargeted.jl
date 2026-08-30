using DataFrames
using Distributions
using StableRNGs
using Test

@testset "bootstrap discrete LMTP (#42)" begin
    rng = StableRNG(1)
    n = 180
    df = DataFrame(arm = rand(rng, [0, 1], n), w = randn(rng, n))
    logit_p = @. 0.2 + 0.6 * df.arm + 0.1 * df.w
    df.y = rand.(rng, Bernoulli.(1 ./ (1 .+ exp.(-logit_p))))

    point = run_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        learners_outcome = (:glm, :mean),
        folds = 2,
        rng = StableRNG(2),
    )

    boot = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        learners_outcome = (:glm, :mean),
        B = 40,
        folds = 2,
        rng = StableRNG(3),
    )
    @test boot.estimate == boot.point.estimate
    @test isfinite(point.estimate)
    @test boot.n_success >= 10
    @test isfinite(boot.se)
    @test boot.lower <= boot.estimate <= boot.upper || boot.n_success < 40

    rng_c = StableRNG(4)
    df.cluster = repeat(1:30, inner = 6)
    boot_c = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        B = 20,
        folds = 2,
        cluster = :cluster,
        rng = rng_c,
    )
    @test boot_c.n_success >= 5
end

@testset "bootstrap two-part LMTP (#42)" begin
    rng = StableRNG(5)
    n = 200
    df = DataFrame(
        arm = rand(rng, ["R", "SS"], n),
        w = randn(rng, n),
    )
    logit_p = @. -0.5 + 0.7 * (df.arm == "SS") + 0.1 * df.w
    df.pres = Float64.(rand.(rng, Bernoulli.(1 ./ (1 .+ exp.(-logit_p)))))
    df.int = [p > 0 ? 1.0 + 0.2 * w + randn(rng) * 0.3 : missing for (p, w) in zip(df.pres, df.w)]

    boot = bootstrap_two_part_discrete_lmtp_contrast(
        df, :arm;
        presence = :pres,
        intensity = :int,
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        baseline = [:w],
        B = 25,
        folds = 2,
        rng = StableRNG(6),
    )
    @test boot.presence.n_success >= 5
    @test boot.intensity.n_success >= 3
    @test all(isfinite, (boot.presence.estimate, boot.intensity.estimate))
end
