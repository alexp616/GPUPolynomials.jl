using Test
using Oscar
include("../src/GPUPolynomials.jl")
using .GPUPolynomials

@testset "GPUPolynomials.jl" begin
    include("CuZZPolyRingElemTests.jl")
    include("CuZZMPolyRingElemTests.jl")
end
