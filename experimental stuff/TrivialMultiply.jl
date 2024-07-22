module TrivialMultiply

using CUDA
include("ReduceByKey.jl")
include("Polynomials.jl")

using BenchmarkTools
using .Polynomials


export trivial_multiply, raise_sparse_to_power, raise_to_power

# make p2 the polynomial with less terms
function trivial_multiply(p1::SparseDevicePolynomial{T}, p2::SparseDevicePolynomial{T}, mod = -1) where {T<:Real}
    numEndingUnreducedTerms = p1.numTerms * p2.numTerms
    unreducedResultLength = nextpow(2, numEndingUnreducedTerms) + 1

    cu_unreducedResultCoeffs = CUDA.zeros(T, unreducedResultLength)
    cu_unreducedResultDegrees = CUDA.zeros(Int, unreducedResultLength)

    kernel = @cuda launch=false trivial_multiply_kernel!(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms, numEndingUnreducedTerms)
    config = launch_configuration(kernel.fun)
    threads = min(numEndingUnreducedTerms, config.threads)
    blocks = cld(numEndingUnreducedTerms, threads)

    kernel(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms, numEndingUnreducedTerms; threads = threads, blocks = blocks)

    # For some reason, this blows up when we try to use Int128's (i know why now)
    if mod > 1
        reduce_mod_m(cu_unreducedResultCoeffs, mod)
    end

    cu_sortedResultDegrees, cu_sortedResultCoeffs = sort_keys_with_values(cu_unreducedResultDegrees, cu_unreducedResultCoeffs)

    CUDA.unsafe_free!(cu_unreducedResultCoeffs)
    CUDA.unsafe_free!(cu_unreducedResultDegrees)

    cu_reducedResultDegrees, cu_reducedResultCoeffs = reduce_by_key(cu_sortedResultDegrees, cu_sortedResultCoeffs)

    CUDA.unsafe_free!(cu_sortedResultCoeffs)
    CUDA.unsafe_free!(cu_sortedResultDegrees)

    if mod > 1
        reduce_mod_m(cu_reducedResultCoeffs, mod)
    end

    return SparseDevicePolynomial(cu_reducedResultCoeffs, cu_reducedResultDegrees, p1.key, length(cu_reducedResultCoeffs))
end

function trivial_multiply_kernel!(coeffs1, degrees1, coeffs2, degrees2, result_coeffs, result_degrees, length2, numEndingUnreducedTerms)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if tid <= numEndingUnreducedTerms
        idx1 = cld(tid, length2)
        idx2 = tid - (idx1 - 1) * length2

        @inbounds begin
            if coeffs1[idx1] == 0
                result_coeffs[tid] = 0
                result_degrees[tid] = 0
            else
                result_coeffs[tid] = coeffs1[idx1] * coeffs2[idx2]
                result_degrees[tid] = degrees1[idx1] + degrees2[idx2]
            end
        end
    end

    return 
end

function raise_sparse_to_power(p::SparseDevicePolynomial{T}, n::Int, mod = -1)::SparseDevicePolynomial{T} where {T<:Real}
    if n == 0
        return SparseDevicePolynomial(CuArray([1]), CuArray([0]), p.key, 1)
    end

    temp = copy(p)
    for _ in 2:n
        temp = trivial_multiply(temp, p)
    end
    return temp
end

function raise_to_power(p::HostPolynomial{T}, n::Int, mod = -1)::HostPolynomial{T} where {T<:Real}
    sparsep = SparseDevicePolynomial(p)
    result = raise_sparse_to_power(sparsep, n, mod)

    return HostPolynomial(result, p.numVars)
end

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

function test_trivial_multiply()
    degrees = generate_compositions(4, 4)
    coeffs = [rand(1:4) for _ in 1:size(degrees, 1)]

    polynomial = HostPolynomial(coeffs, degrees, 81)

    println("Time for first step: ")
    CUDA.@time polynomial2 = raise_to_power(polynomial, 4, 5)

    polynomial2.coeffs .%= 5
    println("Time for second step: ")
    CUDA.@profile polynomial3 = raise_to_power(polynomial2, 5)
end

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

    result = prods[1]
    for i in 2:n + 1
        result = add(result, prods[i])
    end

    return result
end

function test_binb()
    degrees = generate_compositions(4, 4)
    coeffs = [1 for _ in 1:size(degrees, 1)]

    polynomial = HostPolynomial(coeffs, degrees, 81)
    sparsePolynomial = SparseDevicePolynomial(polynomial)

    println("Time for first step: ")
    CUDA.@time binb_raise_to_power(sparsePolynomial, 4)

end

end