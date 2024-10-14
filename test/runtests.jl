using Test

@testset "GPUPolynomials.jl" begin
    include("montgomery_reduction_tests.jl")
    include("polynomials_tests.jl")
end
