using LinearAlgebra: Symmetric, diag, eigvals, issymmetric

@testset "repeated-outcome MSM (point treatment)" begin
    df, truth = simulate_repeated_outcome_ate(800; T = 4, rng = StableRNG(101))
    @test length(truth.tau) == 4
    @test Symbol.(names(df)) == [:W, :A, :Y1, :Y2, :Y3, :Y4]

    res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W],
        folds = 3,
        learners = (:glm, :mean),
        rng = StableRNG(102),
    )
    @test length(res.estimates) == 4
    @test size(res.covariance) == (4, 4)
    @test all(isfinite, res.estimates)
    @test all(>(0), res.se)
    @test issymmetric(res.covariance) ||
        maximum(abs.(res.covariance .- res.covariance')) < 1e-10
    @test maximum(abs.(res.estimates .- truth.tau)) < 0.15
    # Marginal SE should match sqrt(diag(Σ))
    @test res.se ≈ sqrt.(diag(res.covariance)) atol = 1e-10

    # Joint contrast τ₃ − τ₂
    c = msm_contrast(res, 3, 2)
    @test isfinite(c.estimate)
    @test c.se > 0
    @test abs(c.estimate - (truth.tau[3] - truth.tau[2])) < 0.20
    # Linear form with explicit weights
    c2 = msm_contrast(res, [0.0, -1.0, 1.0, 0.0])
    @test c2.estimate ≈ c.estimate atol = 1e-12
    @test c2.se ≈ c.se atol = 1e-12

    estimand = RepeatedOutcomeMSM(:A, [:Y1, :Y2, :Y3, :Y4], [:W])
    @test estimand_engine(estimand) == :repeated_msm
    grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(103), metadata = false)
    @test nrow(grid) == 4
    @test all(isfinite, grid.est)

    plan = plan_mtp(estimand, df; folds = 2)
    @test plan.estimated_fits >= 2 * (1 + 4)  # shared g + per-outcome Q

    # Binary treatment required for MVP
    df_bad = copy(df)
    df_bad.A = df_bad.A .+ 0.1
    @test_throws ArgumentError run_repeated_outcome_msm(
        df_bad, :A, [:Y1, :Y2];
        baseline = [:W], folds = 2, rng = StableRNG(104),
    )

    # ID handoff: same backdoor set for each outcome
    g = DiGraph(6)
    add_edge!(g, 1, 2)  # W → A
    for y in 3:6
        add_edge!(g, 1, y)  # W → Y_t
        add_edge!(g, 2, y)  # A → Y_t
    end
    node_names = [:W, :A, :Y1, :Y2, :Y3, :Y4]
    ids = identify_repeated_outcomes(
        g, :A, [:Y1, :Y2, :Y3, :Y4]; node_names = node_names,
    )
    @test length(ids) == 4
    @test all(r -> r.identifiable, ids)
    @test all(r -> Set(r.adjustment) == Set([:W]), ids)

    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, Symbol[]; baseline = [:W], folds = 2, rng = StableRNG(105),
    )
    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, [:Y1, :Y1]; baseline = [:W], folds = 2, rng = StableRNG(105),
    )
    @test_throws BoundsError msm_contrast(res, 9, 1)
    @test_throws ArgumentError msm_contrast(res, [1.0, 0.0])
    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2];
        baseline = [:W], folds = 2, rng = StableRNG(105), estimator = :bogus,
    )

    eif = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean),
        rng = StableRNG(102), estimator = :eif,
    )
    @test maximum(abs.(eif.estimates .- truth.tau)) < 0.18

    grid_meta = execute_estimand(estimand, df; folds = 2, rng = StableRNG(106))
    @test only(unique(grid_meta.meta_engine)) == "repeated_msm"
    @test only(unique(grid_meta.meta_density_ratio)) == "propensity"
    Σ_grid = metadata(grid_meta, "causal_targeted_msm_covariance")
    @test size(Σ_grid) == (4, 4)

    n = nrow(df)
    long = DataFrame(id = Int[], time = Int[], A = Float64[], W = Float64[], Y = Float64[])
    for i in 1:n
        for t in 1:4
            push!(long, (id = i, time = t, A = df.A[i], W = df.W[i], Y = df[i, Symbol("Y", t)]))
        end
    end
    wide = unstack_repeated_outcomes(
        long; id = :id, time = :time, outcome = :Y, treatment = :A, covariates = [:W],
    )
    @test Symbol.(names(wide)) == [:A, :W, :Y1, :Y2, :Y3, :Y4]
    @test wide.Y1 ≈ df.Y1
    res_long = run_repeated_outcome_msm(
        wide, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 2, learners = (:glm, :mean), rng = StableRNG(107),
    )
    @test all(isfinite, res_long.estimates)

    # Missingness: shared complete profile; do not impute Y
    df_miss = copy(df)
    df_miss.Y2 = Vector{Union{Float64, Missing}}(df.Y2)
    df_miss.Y2[1:40] .= missing
    df_miss.W = Vector{Union{Float64, Missing}}(df.W)
    df_miss.W[41:50] .= missing
    for strat in (:drop, :ipcw, :impute, :ipcw_impute)
        rmiss = run_repeated_outcome_msm(
            df_miss, :A, [:Y1, :Y2, :Y3, :Y4];
            baseline = [:W], folds = 2, learners = (:glm, :mean),
            rng = StableRNG(108), handle_missing = strat,
        )
        @test all(isfinite, rmiss.estimates)
        @test rmiss.missingness.strategy === strat
        @test rmiss.missingness.time_indexed === true
        @test rmiss.n < n
        @test !any(ismissing, rmiss.ic)
    end

    # Cluster-robust Σ: same point estimates; sampling hierarchy only
    df_c = copy(df)
    df_c.cluster = Float64.(mod1.(1:n, 20))
    rng_u = StableRNG(210)
    u_by = Dict{Float64, Float64}(c => randn(rng_u) for c in unique(df_c.cluster))
    uvec = [u_by[c] for c in df_c.cluster]
    for y in (:Y1, :Y2, :Y3, :Y4)
        df_c[!, y] = df_c[!, y] .+ uvec
    end
    res_unit = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
    )
    res_cl = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
        cluster = :cluster,
    )
    @test res_cl.covariance_kind === :cluster
    @test res_unit.covariance_kind === :unit
    @test res_cl.estimates ≈ res_unit.estimates atol = 1e-12
    @test res_cl.covariance ≉ res_unit.covariance
    # Cluster sandwich re-aggregates ICs; finite-sample SE_cluster ≱ SE_unit always
    @test all(isfinite, res_cl.se)
    @test all(≥(0), eigvals(Symmetric(res_cl.covariance)))
    @test length(unique(res_cl.cluster)) == 20
    # Vector form of cluster= matches Symbol form
    res_vec = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
        cluster = df_c.cluster,
    )
    @test res_vec.covariance ≈ res_cl.covariance atol = 1e-12
    @test_throws ArgumentError run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 2, learners = (:glm, :mean), rng = StableRNG(212),
        cluster = :missing_col,
    )
    pres = run_parametric_repeated_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], design = :constant, folds = 3,
        learners = (:glm, :mean), rng = StableRNG(211), cluster = :cluster,
    )
    @test pres.covariance_kind === :cluster

    rec = CausalTargeted.run_julia_synthetic_once(
        :repeated_outcome_ate; n = 600, folds = 3,
        rng = StableRNG(109), learners = (:glm, :mean),
    )
    @test maximum(rec.abs_error) < 0.18
end

