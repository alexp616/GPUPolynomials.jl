using Test
using BenchmarkTools
using CUDA

include("utilsV2.jl")


struct HostPolynomial{T}
    coeffs::Array{T, 1}
    degrees::Array{T, 2}
    numVars::Int32
    numTerms::Int

    function HostPolynomial(c::Array{T, 1}, d::Array{T}) where {T<:Real}
        println(length(size(d)))
        !(length(size(d)) != 1 || length(size(d)) != 2)  ? throw(ArgumentError("Degrees array must have dimension 1 or 2")) :
        length(c) != size(d, 1) ? throw(ArgumentError("Length of coeffs and degrees must be the same")) : new{T}(c, d, size(d, 2), length(c))
    end
end

struct DevicePolynomial{T}
    coeffs::CuArray{T, 1}
    degrees::CuArray{T, 2}
    numVars::Int32
    numTerms::Int

    function DevicePolynomial(hp::HostPolynomial{T}) where {T<:Real}
        new{T}(CuArray(hp.coeffs), CuArray(hp.degrees), hp.numVars, hp.numTerms)
    end
end

# mode = false -> normal representation
# mode = true -> sparse representation
struct Pregenerated{T}
    cu_weakIntegerCompositions::CuArray{T, 2}
    cu_multinomialCoeffs::CuArray{T, 2}
    mode::Bool
end



function pregenerate(numTerms, power)
    factorials = CuArray(generate_factorials(power))
    
    weakIntegerCompositions = generate_compositions(power, num_terms)

    nthreads = min(512, size(weakIntegerCompositions, 1))
    nblocks = cld(size(weakIntegerCompositions, 1), nthreads)

    # Pad to multiple of nthreads if needed
    # +1 is thrown at the end to make reduce_by_key work
    num_paddedrows = cld(size(weakIntegerCompositions, 1), nthreads) * nthreads - size(weakIntegerCompositions, 1) + 1
    cu_weakIntegerCompositions = CuArray(vcat(weakIntegerCompositions, fill(zero(Int32), (num_paddedrows, size(weakIntegerCompositions, 2)))))

    cu_multinomial_coeffs = CUDA.fill(zero(Int64), size(cu_weakIntegerCompositions, 1))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_multinomial_coeffs!(cu_weakIntegerCompositions, cu_multinomial_coeffs, power, num_terms, factorials)
    )

    return Pregenerated(cu_weakIntegerCompositions, cu_multinomial_coeffs)
end

function raise_to_power(polynomial::HostPolynomial{T}, power, pregen = nothing) where {T<:Real}
    if pregen === nothing
        pregen = pregenerate(polynomial.numTerms, power)
    end

end

function pregenerate(numTerms, power)
    factorials = CuArray(generate_factorials(n))

    weakIntegerCompositions = generate_compositions
end

coeffs = vec([1, 2, 3, 4, 5])
degrees = [
    2 2
    3 3
    2 4
    3 3
    2 5
]

hostPolynomial = HostPolynomial(coeffs, degrees)
devicePolynomial = DevicePolynomial(hostPolynomial)

