module GPUPolynomials

using BitIntegers
using CUDA
using FLINT_jll
using Oscar
using Primes
using NTTs

include("utils/nttutils.jl")
include("algorithms/ntt_pow.jl")

include("OperationPlans.jl")

export CuZZMPolyRingElem
export cu
export convert
export GPUPowPlan
include("CuZZMPolyRingElem.jl")

export CufpMPolyRingElem
include("CufpMPolyRingElem.jl")

end