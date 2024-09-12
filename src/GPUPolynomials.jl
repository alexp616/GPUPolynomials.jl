module GPUPolynomials

using CUDA
using Combinatorics
using Oscar
using Primes

using Random
using BitIntegers

#using BenchmarkTools
using CSV
using DataFrames

include("Polynomials.jl")
include("gpu_ntt_pow.jl")
include("Delta1.jl")
include("RandomPolynomials.jl")
include("../benchmarks/Benchmarks.jl")

export delta1, pregen_delta1, convert_to_gpu_representation

end
