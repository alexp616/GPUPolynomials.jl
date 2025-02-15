module GPUPolynomials

using BitIntegers
using CUDA
using FLINT_jll
using Oscar
using Primes
using CudaNTTs

include("utils/nttutils.jl")
include("algorithms/multimod_ntt.jl")
include("algorithms/ntt_mul.jl")
include("algorithms/ntt_pow.jl")

import Base: +, -, ==, *, ^, convert, length, zero, one, eltype
import CUDA.cu

export CuZZPolyRingElem
export cu
export ZZPolyRingElem

include("OperationPlans.jl")

include("univariate/CuPolyRingElem.jl")
include("univariate/CuZZPolyRingElem.jl")

# export CuZZMPolyRingElem
# export cu
# export convert
# export GPUPowPlan
# include("CuZZMPolyRingElem.jl")

# export CufpMPolyRingElem
# include("CufpMPolyRingElem.jl")

end