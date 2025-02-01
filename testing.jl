using CUDA
using Oscar
# using GPUPolynomials
include("src/GPUPolynomials.jl")

function run()
    R, (x, y, z, w) = polynomial_ring(ZZ, 4)
    
    f = x^16 + 2*y^16 + 3*z^16 + 4*w^16

    cu_f = GPUPolynomials.cu(f)
    plan = GPUPolynomials.GPUPowPlan(cu_f, 5)
    cu_f.opPlan = plan

    result = cu_f ^ 5

    cpu_result = ZZMPolyRingElem(result)
    actual_result = f ^ 5

    @assert string(cpu_result) == string(actual_result)
end

run()