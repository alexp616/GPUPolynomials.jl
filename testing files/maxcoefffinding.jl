include("../src/Polynomials.jl")
include("../src/gpu_ntt_pow.jl")

using ..Polynomials
using ..GPUPow

"""
    generate_compositions(n, k)

Escaped hypertriangle code because still useful for generating all possible monomials
"""
function generate_compositions(n, k, type::DataType = Int64)
    compositions = zeros(type, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(type, k)
    current_composition[1] = n
    idx = 1
    while true
        compositions[idx, :] .= current_composition
        idx += 1
        v = current_composition[k]
        if v == n
            break
        end
        current_composition[k] = 0
        j = k - 1
        while 0 == current_composition[j]
            j -= 1
        end
        current_composition[j] -= 1
        current_composition[j + 1] = 1 + v
    end

    return compositions
end

println("Pregenerating...")
pregentime = @timed begin
    pregen1 = pregen_gpu_pow([2281701377, 3221225473, 3489660929, 3892314113, 7918845953, 8858370049], 134217728, BigInt, BigInt)
    # pregen2 = pregen_gpu_pow([3892314113, 7918845953, 8858370049], 134217728, BigInt, BigInt)
end
println("Pregenerating took $(pregentime.time) s")

println("Generating degrees...")
degrees = generate_compositions(40, 4)
coeffs = fill(10, size(degrees, 1))

display(degrees)

poly = HomogeneousPolynomial(coeffs, degrees)

println("Running power...")
powertime = @timed begin
    result1 = gpu_pow_cpu_crt(poly, 11; pregen = pregen1)
    # result2 = gpu_pow_cpu_crt(poly, 11; pregen = pregen2)
end

println("gpupow took $(powertime.time) s")

display(result1.coeffs)
display(result1.degrees)
# display(result2.coeffs)
# display(result2.degrees)

println("maximum coefficient: $(maximum(result1.coeffs))")