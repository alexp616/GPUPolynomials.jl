using Test
using Oscar
# include("../src/GPUPolynomials.jl")
# using .GPUPolynomials
using GPUPolynomials

@testset "GPUPolynomials.jl" begin
    include("CuZZPolyRingElemTests.jl")
    include("CuZZMPolyRingElemTests.jl")
    include("CufpMPolyRingElemTests.jl")
end
