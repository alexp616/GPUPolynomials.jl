module PolynomialModule
using CUDA
global MAX_THREADS_PER_BLOCK = 512

mutable struct HomogeneousPolynomial
    coeffs::Array{Int, 1}
    degrees::Array{Int, 2}
    homogeneousDegree::Int
end

function encode_degrees(degrees::Array{T, 2}, homogeneousDegree)::CuArray{T, 1} where {T<:Number}
    nthreads = min(size(degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = fld(size(degrees, 1), nthreads)

    last_block_threads = size((degrees), 1) - nthreads * nblocks

    result = CUDA.zeros(Float32, size(degrees, 1))
    cu_degrees = CuArray(degrees)

    if (last_block_threads > 0)
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            encode_degrees_kernel!(cu_degrees, homogeneousDegree, result, size(degrees, 2), nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        encode_degrees_kernel!(cu_degrees, homogeneousDegree, result, size(degrees, 2), 0)
    )

    return result
end

function encode_degrees_kernel!(cu_degrees, homogeneousDegree, result, numVars, offset)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
    for i in 1:numVars
        result[tid] += cu_degrees[tid, i] * (homogeneousDegree) ^ (i - 1)
    end

    return nothing
end

function decode_degrees(degrees::CuArray{T, 1}, numVars, homogeneousDegree)::Array{T, 2} where {T<:Number}
    nthreads = min(size(degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = fld(size(degrees, 1), nthreads)

    last_block_threads = length(degrees) - nthreads * nblocks

    result = CUDA.zeros(T, (length(degrees), numVars))

    if (last_block_threads > 0)
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            decode_degrees_kernel!(degrees, homogeneousDegree, result, numVars, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        decode_degrees_kernel!(degrees, homogeneousDegree, result, numVars, 0)
    )

    return Array(result)
end

function decode_degrees_kernel!(cu_degrees, homogeneousDegree, result, numVars, offset)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    val = cu_degrees[tid]

    for i in numVars:-1:1
        x = fld(val, homogeneousDegree ^ (i - 1))
        result[tid, i] = x
        val -= x * homogeneousDegree ^ (i - 1)
    end

    @cuassert val == 0

    return nothing
end

function encode_degrees_fft(hp::HomogeneousPolynomial, pow::Int)
    resultDegree = hp.homogeneousDegree * pow
    resultLength = hp.homogeneousDegree * (resultDegree + 1) ^ 2 + 1
    
    cu_coeffs = CuArray(hp.coeffs)
    cu_degrees = CuArray(hp.degrees)

    result = CUDA.zeros(Int, resultLength)

    function encode_degrees_fft_kernel(cu_coeffs, cu_degrees, result, key, offset = 0)
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
        encoded = 0
        for i in 1:size(cu_degrees, 2) - 1
            encoded += cu_degrees[tid, i] * (key) ^ (i - 1)
        end

        result[encoded + 1] = cu_coeffs[tid]

        return
    end

    nthreads = min(size(degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = fld(size(degrees, 1), nthreads)

    last_block_threads = size(degrees, 1) - nthreads * nblocks

    if last_block_threads > 0
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            encode_degrees_fft_kernel(cu_coeffs, cu_degrees, result, resultDegree + 1, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        encode_degrees_fft_kernel(cu_coeffs, cu_degrees, result, resultDegree + 1)
    )

    return result
end

# coeffs = [1, 1, 1, 1]
# degrees = [
#     4 0 0 0
#     0 4 0 0
#     0 0 4 0
#     0 0 0 4
# ]

# polynomial = HomogeneousPolynomial(coeffs, degrees, 4)
# encoded = Array(encode_degrees_fft(polynomial, 4))
# println(encoded[5])

mutable struct HostPolynomial{T}
    coeffs::Array{T, 1}
    degrees::Array{UInt64, 2}
    homogeneousDegree::Int
    numVars::Int
    numTerms::Int
end

mutable struct EncodedDevicePolynomial{T, U}
    coeffs::CuArray{T, 1}
    encodedDegrees::CuArray{U, 1}
    homogeneousDegree::Int
    numTerms::Int
end

function EncodedDevicePolynomial(hp::HostPolynomial{T}) where T
    return EncodedDevicePolynomial(CuArray(hp.coeffs), encode_degrees(hp.degrees, hp.homogeneousDegree), hp.homogeneousDegree, hp.numTerms)
end

# function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, homogeneousDegree = 100) where {T, U<:Integer}
function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, homogeneousDegree = 100) where {T, U<:Number}
    if !(length(size(degrees)) != 1 || length(size(degrees)) != 2)
        throw(ArgumentError("Degrees array must have dimension 1 or 2"))
    end

    if (length(coeffs) != size(degrees, 1))
        throw(ArgumentError("Length of coeffs and degrees must be the same"))
    end

    return HostPolynomial(coeffs, UInt64.(degrees), homogeneousDegree, size(degrees, 2), length(coeffs))
end


function HostPolynomial(edp::EncodedDevicePolynomial{T}, numVars) where {T<:Integer}
    return HostPolynomial(Array(edp.coeffs), decode_degrees(edp.encodedDegrees, numVars, edp.homogeneousDegree), edp.homogeneousDegree)
end

end