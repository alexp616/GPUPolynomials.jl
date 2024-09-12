module GPUPolynomials

using CUDA
using Combinatorics
using Oscar
using Primes

using Random
using BitIntegers

#using CSV
#using BenchmarkTools
#using DataFrames

include("Polynomials.jl")
include("gpu_ntt_pow.jl")
include("Delta1.jl")
include("RandomPolynomials.jl")

end
