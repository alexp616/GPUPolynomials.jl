using CUDA
include("ReduceByKey.jl")
include("PolynomialModule.jl")
include("utils.jl")

using BenchmarkTools

using .PolynomialModule
CUDA.allowscalar(false)

# The CUDA command that gives the max number of supported threads per block returns 738 for my machine, but
# when I actually try to run code with 738 threads it says I requested too many resources
global MAX_THREADS_PER_BLOCK = 512

# make p2 the polynomial with less terms
function trivial_multiply(p1::PolynomialModule.EncodedDevicePolynomial{T}, p2::PolynomialModule.EncodedDevicePolynomial{T}, mod = -1) where {T<:Real}

    numEndingUnreducedTerms = p1.numTerms * p2.numTerms
    unreducedResultLength = next_pow_2(numEndingUnreducedTerms) + 1

    cu_unreducedResultCoeffs = CUDA.zeros(T, unreducedResultLength)
    cu_unreducedResultDegrees = CUDA.zeros(UInt64, unreducedResultLength)

    nthreads = min(MAX_THREADS_PER_BLOCK, numEndingUnreducedTerms)
    nblocks = fld(numEndingUnreducedTerms, nthreads)

    threadsInLastBlock = numEndingUnreducedTerms - nthreads * nblocks

    if (threadsInLastBlock > 0)
        @cuda(
            threads = threadsInLastBlock,
            blocks = 1,
            trivial_multiply_kernel!(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        trivial_multiply_kernel!(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms, 0)
    )

    # For some reason, this blows up when we try to use Int128's
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

    return PolynomialModule.EncodedDevicePolynomial(cu_reducedResultCoeffs, cu_reducedResultDegrees, p1.maxResultingDegree, length(cu_reducedResultCoeffs))
end

function trivial_multiply_kernel!(coeffs1, degrees1, coeffs2, degrees2, result_coeffs, result_degrees, length2, offset)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
    idx1 = cld(tid, length2)
    idx2 = tid - (idx1 - 1) * length2

    if coeffs1[idx1] == 0
        result_coeffs[tid] = 0
        result_degrees[tid] = 0
    else
        result_coeffs[tid] = coeffs1[idx1] * coeffs2[idx2]
        result_degrees[tid] = degrees1[idx1] + degrees2[idx2]
    end

    return nothing
end

function raise_to_power(p::PolynomialModule.HostPolynomial{T}, n::Int, mod = -1)::PolynomialModule.HostPolynomial{T} where {T<:Real}
    # Only takes positive integer n>=1
    bitarr = to_bits(n)

    encodedResult = PolynomialModule.EncodedDevicePolynomial(CuArray(T.([1])), CuArray(UInt64.([0])), p.maxResultingDegree, 1)
    temp = PolynomialModule.EncodedDevicePolynomial(p)

    for i in 1:length(bitarr) - 1
        if bitarr[i] == 1
            encodedResult = trivial_multiply(temp, encodedResult, mod)
        end
        temp = trivial_multiply(temp, PolynomialModule.EncodedDevicePolynomial(copy(temp.coeffs), copy(temp.encodedDegrees), temp.maxResultingDegree, temp.numTerms), mod)
    end

    encodedResult = trivial_multiply(temp, encodedResult, mod)

    return PolynomialModule.HostPolynomial(encodedResult, p.numVars)
end

function to_bits(n)
    bits = [0 for _ in 1:floor(Int, log2(n)) + 1]
    for i in eachindex(bits)
        bits[i] = n & 1
        n >>= 1
    end
    return bits
end



# EXAMPLE USAGE

# Corresponds to x^4 + y^4 + z^4 + w^4
# coeffs = [1, 1, 1, 1]
# degrees = [
#     1 0 0 0
#     0 1 0 0
#     0 0 1 0
#     0 0 0 1
# ]

#=
degrees = generate_compositions(4, 4)
coeffs = [4 for _ in 1:size(degrees, 1)]

polynomial = PolynomialModule.HostPolynomial(coeffs, degrees, 81)

println("raising to the 4th power:")
polynomial2 = raise_to_power(polynomial, 4)

println(maximum(polynomial2.coeffs))
=#


# maxpolynomial2 = PolynomialModule.HostPolynomial([4 for _ in 1:polynomial2.numTerms], polynomial2.degrees)
# println("raising $(polynomial2.numTerms) terms to the 5th power:")
# polynomial3 = raise_to_power(maxpolynomial2, 5)
# maxcoeff, idx = findmax(polynomial3.coeffs)
# maxdegrees = Int.((polynomial3.degrees)[idx, :])
# println("max coeff: ", maxcoeff)
# println("max coeff's variables: ", maxdegrees)
# println("number of terms: $(polynomial3.numTerms)")


# println("raising $(polynomial2.numTerms) terms to the 5th power:")
# # polynomial3 = raise_to_power(polynomial2, 5)
# CUDA.@profile raise_to_power(polynomial2, 5)
# # println("Resulting polynomial has $(polynomial3.numTerms) terms")







# # Construct Polynomial object
# polynomial = PolynomialModule.HostPolynomial(coeffs, degrees)

# # Raise to the 4th power mod 5
# polynomial2 = raise_to_power(polynomial, 4, 5)
# println(polynomial2.numTerms)
# # @btime raise_to_power(polynomial, 4, 5)
# println(Int.(polynomial2.degrees)[1:20, :])
# raise_to_power(polynomial, 4, 5)
# # Raise to the 5th power
# polynomial3 = raise_to_power(polynomial2, 5)
# println(Int.(polynomial3.degrees)[1:20, :])
# @btime raise_to_power(polynomial2, 5)

# # Corresponds to sum of all possible homogeneous terms of degree 4 and 4 variables
# # degrees = [4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]
# # coeffs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
