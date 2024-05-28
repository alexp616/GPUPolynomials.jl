using CUDA
include("reduce_by_keyv3.jl")
include("PolynomialModule.jl")

using .PolynomialModule
CUDA.allowscalar(false)

# The CUDA command that gives the max number of supported threads per block returns 738 for my machine, but
# when I actually try to run code with 738 threads it says I requested too many resources
global MAX_THREADS_PER_BLOCK = 512

# make p2 the polynomial with less terms
function trivial_multiply(p1::PolynomialModule.EncodedHostPolynomial{T}, p2::PolynomialModule.EncodedHostPolynomial{T}) where {T<:Real}
    # pad longer polynomial to multiple of MAX_THREADS_PER_BLOCK to avoid out of bounds
    if (p1.numTerms * p2.numTerms > MAX_THREADS_PER_BLOCK && p1.numTerms % MAX_THREADS_PER_BLOCK != 0)
        TEST_original = p1.numTerms

        numPaddedRows = cld(length(degrees), MAX_THREADS_PER_BLOCK) * MAX_THREADS_PER_BLOCK - length(degrees)
        p1.coeffs = vcat(p1.coeffs, zeros(T, numPaddedRows))
        p1.encodedDegrees = vcat(p1.coeffs, zeros(Int, numPaddedRows))
        p1.numTerms = length(p1.coeffs)

        @assert (TEST_original != p1.numTerms) "numTerms never changed"
    end

    numEndingUnreducedTerms = p1.numTerms * p2.numTerms

    @assert ((numEndingUnreducedTerms % MAX_THREADS_PER_BLOCK == 0) || (numEndingUnreducedTerms <= MAX_THREADS_PER_BLOCK)) "p1 not padded correctly"

    nthreads = min(MAX_THREADS_PER_BLOCK, numEndingUnreducedTerms)
    nblocks = cld(numEndingUnreducedTerms, nthreads)

    cu_unreducedResultCoeffs = CUDA.zeros(T, numEndingUnreducedTerms)
    cu_unreducedResultDegrees = CUDA.zeros(Int, numEndingUnreducedTerms)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        trivial_multiply_kernel!(CuArray(p1.coeffs), CuArray(p1.encodedDegrees), CuArray(p2.coeffs), CuArray(p2.encodedDegrees), cu_unreducedResultCoeffs, cu_unreducedResultDegrees, p2.numTerms)
    )

    # println("cu_unreducedResultCoeffs: ", cu_unreducedResultCoeffs)
    # println("cu_unreducedResultDegrees: ", cu_unreducedResultDegrees)

    # cu_unreducedResultDegrees, cu_unreducedResultCoeffs = sort_keys_with_values(cu_unreducedResultDegrees, cu_unreducedResultCoeffs)
    sortedResultDegrees, sortedResultCoeffs = sort_keys_with_values(cu_unreducedResultDegrees, cu_unreducedResultCoeffs)

    sortedUnreducedResultCoeffs = Array(sortedResultCoeffs)
    sortedUnreducedResultDegrees = Array(sortedResultDegrees)

    reducedResultDegrees, reducedResultCoeffs = reduce_by_key(sortedUnreducedResultDegrees, sortedUnreducedResultCoeffs)

    return PolynomialModule.EncodedHostPolynomial(reducedResultCoeffs, reducedResultDegrees, p1.maxResultingDegree, length(reducedResultCoeffs))
end

function trivial_multiply_kernel!(coeffs1, degrees1, coeffs2, degrees2, result_coeffs, result_degrees, length2)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx1 = cld(tid, length2)
    idx2 = tid - (idx1 - 1) * length2

    result_coeffs[tid] = coeffs1[idx1] * coeffs2[idx2]
    result_degrees[tid] = degrees1[idx1] + degrees2[idx2]

    result_coeffs[tid] = coeffs1[idx1] * coeffs2[idx2]
    result_degrees[tid] = degrees1[idx1] + degrees2[idx2]

    return nothing
end

function raise_to_power(p::PolynomialModule.HostPolynomial{T}, n::Int)::PolynomialModule.HostPolynomial where {T<:Real}
    # Only takes positive integer n>=1
    bitarr = to_bits(n)

    encodedResult = PolynomialModule.EncodedHostPolynomial([1], [0], p.maxResultingDegree, 1)
    temp = PolynomialModule.EncodedHostPolynomial(p)

    for i in 1:length(bitarr) - 1
        if bitarr[i] == 1
            encodedResult = trivial_multiply(temp, encodedResult)
        end
        temp = trivial_multiply(temp, temp)
    end

    encodedResult = trivial_multiply(temp, encodedResult)

    return PolynomialModule.HostPolynomial(encodedResult, p.numVars)
end

function to_bits(n)
    bits = [0 for _ in 1:ceil(log2(n))]
    for i in eachindex(bits)
        bits[i] = n & 1
        n >>= 1
    end
    return bits
end

coeffs = [1, 1, 1]
degrees = [
    2 0
    1 1
    0 2
]

polynomial = PolynomialModule.HostPolynomial(coeffs, degrees, 11)

polynomial2 = raise_to_power(polynomial, 5)

println(polynomial2.coeffs)
println(polynomial2.degrees)