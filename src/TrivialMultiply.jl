using CUDA
include("ReduceByKey.jl")
include("PolynomialModule.jl")
include("../bunch of random files/utilsV2.jl")

using BenchmarkTools

using .PolynomialModule
CUDA.allowscalar(false)

# The CUDA command that gives the max number of supported threads per block returns 738 for my machine, but
# when I actually try to run code with 738 threads it says I requested too many resources
global MAX_THREADS_PER_BLOCK = 512

function faster_mod(x, m)
    return x - div(x, m) * m
end

function reduce_mod_m(arr, m)
    CUDA.@sync arr .= faster_mod.(arr, m)
end

# make p2 the polynomial with less terms
function trivial_multiply(p1::PolynomialModule.EncodedHostPolynomial{T}, p2::PolynomialModule.EncodedHostPolynomial{T}, mod = -1) where {T<:Real}
    # pad longer polynomial to multiple of MAX_THREADS_PER_BLOCK to avoid out of bounds
    if (p1.numTerms * p2.numTerms > MAX_THREADS_PER_BLOCK && p1.numTerms % MAX_THREADS_PER_BLOCK != 0)
        # TEST_original = p1.numTerms

        numPaddedRows = cld(p1.numTerms, MAX_THREADS_PER_BLOCK) * MAX_THREADS_PER_BLOCK - p1.numTerms
        p1.coeffs = vcat(p1.coeffs, zeros(T, numPaddedRows))
        p1.encodedDegrees = vcat(p1.encodedDegrees, zeros(Int, numPaddedRows))
        p1.numTerms = length(p1.coeffs)

        # @assert (TEST_original != p1.numTerms) "numTerms never changed"
    end

    numEndingUnreducedTerms = p1.numTerms * p2.numTerms

    # @assert ((numEndingUnreducedTerms % MAX_THREADS_PER_BLOCK == 0) || (numEndingUnreducedTerms <= MAX_THREADS_PER_BLOCK)) "p1 not padded correctly"

    nthreads = min(MAX_THREADS_PER_BLOCK, numEndingUnreducedTerms)
    nblocks = cld(numEndingUnreducedTerms, nthreads)

    cu_unreducedResultCoeffs = CUDA.zeros(T, numEndingUnreducedTerms)
    cu_unreducedResultDegrees = CUDA.zeros(UInt32, numEndingUnreducedTerms)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        trivial_multiply_kernel!(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms)
    )

    # For some reason, this blows up when we try to use Int128's
    if mod > 1
        reduce_mod_m(cu_unreducedResultCoeffs, mod)
    end

    cu_sortedResultDegrees, cu_sortedResultCoeffs = sort_keys_with_values(cu_unreducedResultDegrees, cu_unreducedResultCoeffs)

    CUDA.unsafe_free!(cu_unreducedResultCoeffs)
    CUDA.unsafe_free!(cu_unreducedResultDegrees)

    sortedUnreducedResultCoeffs = Array(cu_sortedResultCoeffs)
    sortedUnreducedResultDegrees = Array(cu_sortedResultDegrees)

    # reduce_by_key returns CuArray
    cu_reducedResultDegrees, cu_reducedResultCoeffs = reduce_by_key(sortedUnreducedResultDegrees, sortedUnreducedResultCoeffs)

    CUDA.unsafe_free!(cu_sortedResultCoeffs)
    CUDA.unsafe_free!(cu_sortedResultDegrees)

    if mod > 1
        reduce_mod_m(cu_reducedResultCoeffs, mod)
    end

    reducedResultCoeffs = Array(cu_reducedResultCoeffs)
    reducedResultDegrees = Array(cu_reducedResultDegrees)

    CUDA.unsafe_free!(cu_reducedResultCoeffs)
    CUDA.unsafe_free!(cu_reducedResultDegrees)

    return PolynomialModule.EncodedHostPolynomial(reducedResultCoeffs, reducedResultDegrees, p1.maxResultingDegree, length(reducedResultCoeffs))
end

function trivial_multiply_kernel!(coeffs1, degrees1, coeffs2, degrees2, result_coeffs, result_degrees, length2)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
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

    encodedResult = PolynomialModule.EncodedHostPolynomial(T.([1]), UInt32.([0]), p.maxResultingDegree, 1)
    temp = PolynomialModule.EncodedHostPolynomial(p)

    for i in 1:length(bitarr) - 1
        if bitarr[i] == 1
            encodedResult = trivial_multiply(temp, encodedResult, mod)
        end
        temp = trivial_multiply(temp, PolynomialModule.EncodedHostPolynomial(copy(temp.coeffs), copy(temp.encodedDegrees), temp.maxResultingDegree, temp.numTerms), mod)
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

# coeffs = [1, 1, 1, 1, 1]
# degrees = [
#     4 0 0 0
#     3 1 0 0
#     1 0 1 2
#     0 0 3 1
#     0 4 0 0
# ]

# polynomial = PolynomialModule.HostPolynomial(coeffs, degrees)
# polynomial2 = raise_to_power(polynomial, 5)

# @btime raise_to_power(polynomial, 6)

# max_value, max_index = findmax(polynomial2.coeffs)
# println(max_value)
# println(Int.(polynomial2.degrees[max_index, :]))

# println(polynomial2.coeffs)
# println(Int.(polynomial2.degrees))

# @benchmark raise_to_power(polynomial, 20)

function delta1(polynomial::PolynomialModule.HostPolynomial{T}, prime) where {T<:Integer}
    g = raise_to_power(polynomial, prime - 1, prime)

    gp = raise_to_power(g, prime)

    is_any_negative = any(x -> x < 0, gp.coeffs)
    @assert(is_any_negative == false)

    return gp
end

function test_for_negative()
    degrees = generate_compositions(4, 4, Int64)
    coeffs = [1 for _ in 1:size(degrees, 1)]

    polynomial = PolynomialModule.HostPolynomial(coeffs, degrees)

    g = raise_to_power(polynomial, 4)
    g.coeffs = fill(4, length(coeffs))

    gp = raise_to_power(g, 5)
    is_any_negative = any(x -> x < 0, gp.coeffs)
    @assert(is_any_negative == false)
    println("chilling")
end

test_for_negative()