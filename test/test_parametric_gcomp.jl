using Distributions
using GLM

@testset "Formula-based parametric g-computation" begin
    function factorial_data()
        rows = NamedTuple[]
        infection_levels = ("0_0", "0_1", "1_0", "1_1")
        for replicate in 1:8,
            line in ("ROL", "ROH"),
            protein in ("LP", "HP"),
            infection in infection_levels
            line_high = line == "ROH"
            protein_high = protein == "HP"
            infection_11 = infection == "1_1"
            outcome = 2.0 + 0.6line_high + 0.3protein_high +
                0.4line_high * protein_high + 0.10infection_11 +
                0.05line_high * infection_11 +
                0.08protein_high * infection_11 + 0.01replicate
            push!(rows, (;
                Y = outcome,
                Line = line,
                Protein = protein,
                InfectionHistory = infection,
                replicate,
            ))
        end
        return DataFrame(rows)
    end

    data = factorial_data()
    formula_term = @formula(
        Y ~ Line + Protein + InfectionHistory +
            Line & Protein + Line & InfectionHistory + Protein & InfectionHistory
    )
    fit = fit_parametric_gcomp(formula_term, data; family = :gaussian)

    @testset "factorial marginal, subgroup, and formal interaction" begin
        marginal = gcomp_contrast(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        subgroup = gcomp_contrast(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            by = (; Line = "ROH"),
        )
        interaction = gcomp_interaction(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            modifier = :Line,
            modifier_reference = "ROL",
            modifier_comparison = "ROH",
        )

        @test marginal.estimate ≈ 0.52 atol = 1.0e-10
        @test subgroup.estimate ≈ 0.72 atol = 1.0e-10
        @test interaction.estimate ≈ 0.40 atol = 1.0e-10
        @test interaction.estimate ≈
            interaction.modifier_comparison_effect - interaction.modifier_reference_effect
        @test marginal.n == nrow(data)
        @test subgroup.n == count(==("ROH"), data.Line)
        @test fit.covariance_type == :hc3

        manual_target = data[data.Line .== "ROH", :]
        manual_hp = copy(manual_target)
        manual_hp.Protein .= "HP"
        manual_lp = copy(manual_target)
        manual_lp.Protein .= "LP"
        @test subgroup.comparison_mean ≈ mean(GLM.predict(fit.model, manual_hp))
        @test subgroup.reference_mean ≈ mean(GLM.predict(fit.model, manual_lp))
    end

    @testset "three-level categorical intervention and fitted levels" begin
        three_level = DataFrame(
            Y = [1.0, 1.1, 2.0, 2.1, 4.0, 4.1, 1.2, 2.2, 4.2],
            A = repeat(["low", "middle", "high"], 3),
            W = repeat([0.0, 1.0, 2.0], inner = 3),
        )
        three_fit = fit_parametric_gcomp(
            @formula(Y ~ A + W), three_level; family = :gaussian, covariance = :model,
        )
        result = gcomp_contrast(
            three_fit,
            three_level;
            treatment = :A,
            reference = "middle",
            comparison = "high",
        )
        high = copy(three_level)
        high.A .= "high"
        middle = copy(three_level)
        middle.A .= "middle"
        @test result.estimate ≈
            mean(GLM.predict(three_fit.model, high)) - mean(GLM.predict(three_fit.model, middle))
        error = try
            gcomp_mean(three_fit, three_level; set = (; A = "unseen"))
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("unseen level", sprint(showerror, error))
    end

    @testset "response-scale logistic risks and gradient" begin
        logistic_data = factorial_data()
        eta = -1.2 .+ 0.9 .* (logistic_data.Protein .== "HP") .+
            0.7 .* (logistic_data.Line .== "ROH") .+
            0.35 .* (logistic_data.InfectionHistory .== "1_1")
        probability = 1.0 ./ (1.0 .+ exp.(-eta))
        rng = StableRNG(991)
        logistic_data.Y = Float64.(rand.(Ref(rng), Bernoulli.(probability)))
        logistic_fit = fit_parametric_gcomp(
            formula_term, logistic_data; family = :binomial,
        )
        risk_difference = gcomp_contrast(
            logistic_fit,
            logistic_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        risk_ratio = gcomp_contrast(
            logistic_fit,
            logistic_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp = copy(logistic_data)
        hp = copy(logistic_data)
        lp.Protein .= "LP"
        hp.Protein .= "HP"
        manual_lp = mean(GLM.predict(logistic_fit.model, lp))
        manual_hp = mean(GLM.predict(logistic_fit.model, hp))
        @test risk_difference.estimate ≈ manual_hp - manual_lp atol = 1.0e-12
        @test risk_ratio.estimate ≈ manual_hp / manual_lp atol = 1.0e-12
        @test risk_ratio.ci_lower > 0

        component = CausalTargeted._gcomp_mean_components(
            logistic_fit, logistic_data; set = (; Protein = "HP"),
        )
        counterfactual = copy(logistic_data)
        counterfactual.Protein .= "HP"
        design = CausalTargeted._gcomp_design(logistic_fit, counterfactual)
        beta = coef(logistic_fit.model)
        finite_difference = similar(beta)
        for index in eachindex(beta)
            step = 1.0e-6 * (1 + abs(beta[index]))
            upper, lower = copy(beta), copy(beta)
            upper[index] += step
            lower[index] -= step
            upper_mean = mean(1.0 ./ (1.0 .+ exp.(-(design * upper))))
            lower_mean = mean(1.0 ./ (1.0 .+ exp.(-(design * lower))))
            finite_difference[index] = (upper_mean - lower_mean) / (2step)
        end
        @test component.gradient ≈ finite_difference rtol = 2.0e-6 atol = 2.0e-8
    end

    @testset "Gamma and NB response-scale mean ratios" begin
        log_data = factorial_data()
        eta = 0.3 .+ 0.55 .* (log_data.Protein .== "HP") .+
            0.2 .* (log_data.Line .== "ROH") .+
            0.1 .* log_data.replicate
        rng = StableRNG(883)
        means = exp.(eta)
        log_data.Y = [rand(rng, Gamma(4.0, mu / 4.0)) for mu in means]
        gamma_fit = fit_parametric_gcomp(formula_term, log_data; family = :gamma)
        gamma_ratio = gcomp_contrast(
            gamma_fit,
            log_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp = copy(log_data)
        hp = copy(log_data)
        lp.Protein .= "LP"
        hp.Protein .= "HP"
        @test gamma_ratio.estimate ≈
            mean(GLM.predict(gamma_fit.model, hp)) / mean(GLM.predict(gamma_fit.model, lp))

        mean_eta = mean(CausalTargeted._gcomp_design(gamma_fit, hp) * coef(gamma_fit.model))
        @test gamma_ratio.comparison_mean ≈ mean(GLM.predict(gamma_fit.model, hp))
        @test abs(gamma_ratio.comparison_mean - exp(mean_eta)) > 1.0e-4

        count_data = factorial_data()
        count_means = exp.(0.7 .+ 0.4 .* (count_data.Protein .== "HP") .+
            0.15 .* (count_data.Line .== "ROH"))
        rng_count = StableRNG(884)
        count_data.Y = Float64[
            rand(rng_count, NegativeBinomial(2.3, 2.3 / (2.3 + mu))) for mu in count_means
        ]
        fixed_nb = fit_parametric_gcomp(
            formula_term, count_data; family = :negativebinomial, theta = 2.3,
        )
        fixed_ratio = gcomp_contrast(
            fixed_nb,
            count_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp_count, hp_count = copy(count_data), copy(count_data)
        lp_count.Protein .= "LP"
        hp_count.Protein .= "HP"
        @test fixed_ratio.estimate ≈
            mean(GLM.predict(fixed_nb.model, hp_count)) / mean(GLM.predict(fixed_nb.model, lp_count))
        @test fixed_nb.theta ≈ 2.3
    end

    @testset "ratio interaction reconciliation and bootstrap strata" begin
        positive_data = factorial_data()
        positive_data.Y = exp.(positive_data.Y)
        log_fit = fit_parametric_gcomp(formula_term, positive_data; family = :gamma)
        interaction = gcomp_interaction(
            log_fit,
            positive_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            modifier = :Line,
            modifier_reference = "ROL",
            modifier_comparison = "ROH",
            scale = :ratio,
        )
        @test interaction.estimate ≈
            interaction.modifier_comparison_effect / interaction.modifier_reference_effect

        rng = StableRNG(123)
        indices = CausalTargeted._gcomp_bootstrap_indices(
            rng, positive_data, [:Line, :Protein, :InfectionHistory],
        )
        resample = positive_data[indices, :]
        original_counts = combine(
            groupby(positive_data, [:Line, :Protein, :InfectionHistory]), nrow => :n,
        )
        resample_counts = combine(
            groupby(resample, [:Line, :Protein, :InfectionHistory]), nrow => :n,
        )
        sort!(original_counts, [:Line, :Protein, :InfectionHistory])
        sort!(resample_counts, [:Line, :Protein, :InfectionHistory])
        @test original_counts.n == resample_counts.n
    end

    @testset "direct formula GLM and convenience runner" begin
        direct = glm(formula_term, data, Normal(), IdentityLink())
        direct_result = gcomp_contrast(
            direct,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        run_result = run_parametric_gcomp(
            formula_term,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        @test direct_result.estimate ≈ 0.52 atol = 1.0e-10
        @test run_result.estimate ≈ direct_result.estimate
    end
end
