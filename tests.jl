include("src/Delta1.jl")
include("src/gpu_ntt_pow.jl")
include("src/Polynomials.jl")
using .Delta1

using Oscar
using Test


function convert_to_oscar(hp::HomogeneousPolynomial, ring::FqMPolyRing)
    vars = gens(ring)
    numVars = size(hp.degrees, 2)

    @assert length(vars) == numVars "Number of variables of hp and ring not compatible"

    result = zero(ring)

    for (i, coeff) in enumerate(hp.coeffs)
        expRow = hp.degrees[i, :]
        term = coeff * prod(vars[j] ^ expRow[j] for j in 1:numVars)
        result += term
    end

    return result
end

function test_gpu_pow()
    R, (x, y, z, w) = polynomial_ring(GF(5), 4)

    f = x^4 + y^4 + z^4 + w^4
    
    oscar_result = f^4

    f_gpu = HomogeneousPolynomial(f)
    result_gpu = gpu_pow(f_gpu, 4)
    gpu_back = convert_to_oscar(result_gpu, R)

    @test oscar_result == gpu_back
end

test_gpu_pow()

# R, (x, y, z, w) = polynomial_ring(poly.parent.base_ring, 4)
# Δ₁fpminus1 = zero(R)

# for (i, coeff) in enumerate(Δ₁fpminus1_gpu.coeffs)
#     exp_row = Δ₁fpminus1_gpu.degrees[i, :]
#     term = coeff * x^exp_row[1] * y^exp_row[2] * z^exp_row[3] * w^exp_row[4]
#     Δ₁fpminus1 += term
# end