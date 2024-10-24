using Test

include("../src/GPUPolynomials.jl")

@testset "GPUPolynomials.jl" begin
    include("homogeneous_polynomial_tests.jl")
end
