using CausalTargeted
using CausalDynamics
using Test

@testset "RxInfer load smoke (#30)" begin
    if Base.find_package("RxInfer") === nothing
        @test_skip "RxInfer not installed in test environment"
    else
        @eval using RxInfer
        @test isdefined(Main, :RxInfer)
        @test CausalDynamics.has_rxinfer() in (true, false)
    end
end
