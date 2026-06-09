using Test
using CUDA
using Oscar
# include("../src/GPUPolynomials.jl")
# using .GPUPolynomials
using GPUPolynomials

@testset "GPUPolynomials.jl" begin
    if CUDA.functional()
        include("CuZZPolyRingElemTests.jl")
        include("CuZZMPolyRingElemTests.jl")
        include("CufpMPolyRingElemTests.jl")
    else
        @info "Skipping GPU tests because CUDA.functional() is false"
        @test true
    end
end
