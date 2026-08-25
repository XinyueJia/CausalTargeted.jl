using LinearAlgebra: diag, issymmetric

@testset "parametric treatment×time MSM" begin
    # Known contrast profile: τ(t) = 0.2 + 0.15*(t-1)
    T = 4
    β_true = [0.2, 0.15]  # intercept + linear time slope on τ
    τ_true = [β_true[1] + β_true[2] * (t - 1) for t in 1:T]
    df, truth = simulate_repeated_outcome_ate(
        900; T = T, β_a = τ_true, rng = StableRNG(201),
    )
    @test truth.tau ≈ τ_true

    res = run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W],
        design = :linear_time,
        folds = 3,
        learners = (:glm, :mean),
        rng = StableRNG(202),
    )
    @test length(res.coefficients) == 2
    @test res.design === :linear_time
    @test res.coef_names == [:intercept, :time]
    @test maximum(abs.(res.coefficients .- β_true)) < 0.12
    @test size(res.covariance) == (2, 2)
    @test all(>(0), res.se)
    @test res.se ≈ sqrt.(diag(res.covariance)) atol = 1e-10
    # Fitted profile close to truth
    @test maximum(abs.(res.fitted_tau .- τ_true)) < 0.15
    # Contrast on β: slope
    c_slope = msm_contrast(res, [0.0, 1.0])
    @test abs(c_slope.estimate - β_true[2]) < 0.12

    # Constant ATE across time
    τ_c = fill(0.55, 3)
    df_c, _ = simulate_repeated_outcome_ate(
        700; T = 3, β_a = τ_c, rng = StableRNG(203),
    )
    res_c = run_parametric_repeated_msm(
        df_c, :A, [:Y1, :Y2, :Y3];
        baseline = [:W], design = :constant,
        folds = 3, learners = (:glm, :mean), rng = StableRNG(204),
    )
    @test length(res_c.coefficients) == 1
    @test abs(only(res_c.coefficients) - 0.55) < 0.12

    # Factor time (saturated on τ) recovers unstructured
    res_f = run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], design = :factor_time,
        folds = 3, learners = (:glm, :mean), rng = StableRNG(202),
    )
    @test length(res_f.coefficients) == 4
    un = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(202),
    )
    @test res_f.fitted_tau ≈ un.estimates atol = 1e-8

    # Mean model m(t,a) = β0 + βA*a + time dummies + A×time
    df_m, truth_m = simulate_mean_treatment_time_msm(800; T = 3, rng = StableRNG(205))
    res_m = run_parametric_repeated_msm(
        df_m, :A, [:Y1, :Y2, :Y3];
        baseline = [:W],
        design = :mean_treatment_time,
        target = :mean,
        folds = 3,
        learners = (:glm, :mean),
        rng = StableRNG(206),
    )
    @test res_m.target === :mean
    @test length(res_m.coefficients) == 2 + 2 * (3 - 1)  # β0, βA, βt×2, βAt×2
    @test maximum(abs.(res_m.coefficients .- truth_m.beta)) < 0.20
    # Implied τ(t) = m(t,1)-m(t,0)
    @test maximum(abs.(res_m.fitted_tau .- truth_m.tau)) < 0.22

    estimand = ParametricRepeatedOutcomeMSM(
        :A, [:Y1, :Y2, :Y3, :Y4], [:W]; design = :linear_time,
    )
    @test estimand_engine(estimand) == :parametric_msm
    grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(207), metadata = false)
    @test nrow(grid) == 2
    @test all(isfinite, grid.est)

    @test_throws ArgumentError run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2];
        baseline = [:W], design = :bogus, folds = 2, rng = StableRNG(208),
    )

    # Custom design matrix for τ
    D = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    res_d = run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], design = D,
        folds = 2, learners = (:glm, :mean), rng = StableRNG(209),
    )
    @test length(res_d.coefficients) == 2
end
