module GPUPolynomials

using BitIntegers
using CUDA
using FLINT_jll
using Oscar
using Primes

include("algorithms/gpu/gpu_ntt_pow.jl")

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

include("get_oscar_data.jl")
include("homogeneous_polynomial.jl")

export HomogeneousPolynomial
export get_coeffs
export get_exps
export exps_matrix
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

end


