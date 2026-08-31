using CausalTargeted
using CausalDynamics
using DataFrames
using Graphs
using Random
using Test

@testset "RxInfer unified stack smoke (#30)" begin
    if Base.find_package("RxInfer") === nothing
        @test_skip "RxInfer not installed in test environment"
    else
        @eval using RxInfer

        @test isdefined(Main, :RxInfer)
        @test CausalDynamics.has_rxinfer()

        Random.seed!(42)
        n = 80
        z = randn(n)
        x = z .+ 0.3 .* randn(n)
        y = 1.5 .* x .+ 0.8 .* z .+ 0.2 .* randn(n)
        data = DataFrame(Z = z, X = x, Y = y)

        g = DiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        names = Dict(1 => :Z, 2 => :X, 3 => :Y)

        result = CausalDynamics.infer_backdoor_effect(
            g, data, 2, 3; node_names = names, iterations = 15,
        )
        @test result.identifiable
        @test result.confounders == [:Z]
        @test 0.5 < result.τ_mean < 2.5
    end
end
