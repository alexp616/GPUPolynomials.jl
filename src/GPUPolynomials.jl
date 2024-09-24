module GPUPolynomials

include("algorithms/gpu/gpu_ntt_pow.jl")
include("get_oscar_data.jl")
include("polynomials.jl")

using .GetOscarData
using .Polynomials
using .GPUNTTPow

for submodule in [GetOscarData, Polynomials, GPUNTTPow]
    for name in names(submodule, all = false, imported = true)
        @eval export $(Symbol(name))
    end
end

end
