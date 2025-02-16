module GPUPolynomials

using BitIntegers
using CUDA
using FLINT_jll
using Oscar
using Primes
using CudaNTTs
using Combinatorics

include("utils/nttutils.jl")
include("algorithms/ntt_mul.jl")
include("algorithms/ntt_pow.jl")

include("OperationPlans.jl")

import Base: +, -, ==, *, ^, convert, length, zero, one, eltype
import CUDA.cu

export CuZZPolyRingElem
export ZZPolyRingElem
export cu
export convert
export NTTMulPlan
export NTTPowPlan

include("univariate/CuPolyRingElem.jl")
include("univariate/CuZZPolyRingElem.jl")

export CuZZMPolyRingElem
export MPowPlan

include("multivariate/CuMPolyRingElem.jl")
include("multivariate/CuZZMPolyRingElem.jl")

# export CufpMPolyRingElem
# include("CufpMPolyRingElem.jl")

include("random_polynomials.jl")

end