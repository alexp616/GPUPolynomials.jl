module GPUPolynomials

using CUDA
using Primes
using Dates
using BitIntegers
using Oscar
using FLINT_jll

include("algorithms/gpu/gpu_ntt_pow.jl")

export mod, div

export GPUNTTPregen, gpu_ntt!, memorysafe_gpu_ntt!, gpu_intt!, GPUPowPregen, pregen_gpu_pow, gpu_ntt_pow, sparsify, build_result, generate_butterfly_permutations, memorysafe_gpu_ntt_pow

include("get_oscar_data.jl")
include("homogeneous_polynomial.jl")

export HomogeneousPolynomial, get_coeffs, get_exps, exps_matrix, nvars, convert, get_dense_representation, get_sparse_representation, new_MPolyRingElem, gpu_pow, pregen_gpu_pow, exp_matrix_to_vec, fpMPolyRingElem, sort_terms

end


