module BINB

include("TrivialMultiply.jl")
include("Polynomials.jl")
include("ReduceByKey.jl")

using .TrivialMultiply
using .Polynomials

function binb_raise_to_power(p::SparseDevicePolynomial{T}, n::Int) where T<:Real
    sort_keys_with_values(p.encodedDegrees, p.coeffs)

    gterms = div(p.numTerms, 2)
    hterms = p.numTerms - gterms
    g = SparseDevicePolynomial(p.coeffs[1:gterms], p.encodedDegrees[1:gterms], p.key, gterms)
    h = SparseDevicePolynomial(p.coeffs[gterms + 1:hterms], p.encodedDegrees[gterms + 1:hterms], p.key, hterms)

    one = SparseDevicePolynomial(CuArray([1]), CuArray([1]), p.key, 1)

    gpowers = Array{SparseDevicePolynomial{T}}(undef, n + 1)
    hpowers = Array{SparseDevicePolynomial{T}}(undef, n + 1)

    gpowers[1] = one
    hpowers[1] = one

    gpowers[2] = g
    hpowers[2] = h

    for i in 2:n
        gpowers[i + 1] = trivial_multiply(gpowers[i], g)
        hpowers[i + 1] = trivial_multiply(hpowers[i], h)
    end

    for i in 0:n
        gpowers[i + 1].coeffs .*= binomial(n, i)
    end

    prods = Array{SparseDevicePolynomial{T}}(undef, n + 1)

    for i in 0:n
        prods[i + 1] = trivial_multiply(gpowers[i + 1], hpowers[n + 1 - i])
    end

    prods = SparsePolynomial.(prods)

    result = prods[1]
    for i in 2:k + 1
        result = result + prods[i]
    end

    return result
end

function test_binb()
    degrees = generate_compositions(4, 4)
    coeffs = [rand(1:4) for _ in 1:size(degrees, 1)]

    polynomial = HostPolynomial(coeffs, degrees, 81)
    sparsePolynomial = SparseDevicePolynomial(polynomial)

    println("Time for first step: ")
    CUDA.@time binb_raise_to_power(sparsePolynomial, 4)



end



end