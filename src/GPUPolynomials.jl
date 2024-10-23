module GPUPolynomials

__precompile__(false)

using BitIntegers
using CUDA
using FLINT_jll
using Oscar
using Primes

export GPUNTTPregen
export gpu_ntt!
export memorysafe_gpu_ntt!
export gpu_intt!
export GPUPowPregen
export pregen_gpu_pow
export gpu_ntt_pow
export sparsify
export build_result
export generate_butterfly_permutations
export memorysafe_gpu_ntt_pow

include("algorithms/gpu/gpu_ntt_pow.jl")


export get_coeffs
export get_exps
export exps_matrix

include("get_oscar_data.jl")


export HomogeneousPolynomial
export nvars
export convert
export get_dense_representation
export get_sparse_representation
export new_MPolyRingElem
export gpu_pow
export pregen_gpu_pow
export exp_matrix_to_vec
export fpMPolyRingElem
export sort_terms

include("homogeneous_polynomial.jl")

end