module PolynomialModule
using CUDA
global MAX_THREADS_PER_BLOCK = 512

function encode_degrees(degrees::Array{T, 2}, maxResultingDegree)::CuArray{T, 1} where {T<:Number}
    
    nthreads = min(size(degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = fld(size(degrees, 1), nthreads)

    last_block_threads = size((degrees), 1) - nthreads * nblocks

    result = CUDA.zeros(Float32, size(degrees, 1))
    cu_degrees = CuArray(degrees)

    if (last_block_threads > 0)
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            encode_degrees_kernel!(cu_degrees, maxResultingDegree, result, size(degrees, 2), nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        encode_degrees_kernel!(cu_degrees, maxResultingDegree, result, size(degrees, 2), 0)
    )

    return result
end

function encode_degrees_kernel!(cu_degrees, maxResultingDegree, result, numVars, offset)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
    for i in 1:numVars
        result[tid] += cu_degrees[tid, i] * (maxResultingDegree) ^ (i - 1)
    end

    return nothing
end

function decode_degrees(degrees::CuArray{T, 1}, numVars, maxResultingDegree)::Array{T, 2} where {T<:Number}
    nthreads = min(size(degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = fld(size(degrees, 1), nthreads)

    last_block_threads = length(degrees) - nthreads * nblocks

    result = CUDA.zeros(T, (length(degrees), numVars))

    if (last_block_threads > 0)
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            decode_degrees_kernel!(degrees, maxResultingDegree, result, numVars, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        decode_degrees_kernel!(degrees, maxResultingDegree, result, numVars, 0)
    )

    return Array(result)
end

function decode_degrees_kernel!(cu_degrees, maxResultingDegree, result, numVars, offset)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    val = cu_degrees[tid]

    for i in numVars:-1:1
        x = fld(val, maxResultingDegree ^ (i - 1))
        result[tid, i] = x
        val -= x * maxResultingDegree ^ (i - 1)
    end

    @cuassert val == 0

    return nothing
end

mutable struct HostPolynomial{T}
    coeffs::Array{T, 1}
    degrees::Array{UInt64, 2}
    maxResultingDegree::Int
    numVars::Int
    numTerms::Int
end

mutable struct EncodedDevicePolynomial{T, U}
    coeffs::CuArray{T, 1}
    encodedDegrees::CuArray{U, 1}
    maxResultingDegree::Int
    numTerms::Int
end

function EncodedDevicePolynomial(hp::HostPolynomial{T}) where T
    return EncodedDevicePolynomial(CuArray(hp.coeffs), encode_degrees(hp.degrees, hp.maxResultingDegree), hp.maxResultingDegree, hp.numTerms)
end

# function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, maxResultingDegree = 100) where {T, U<:Integer}
function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, maxResultingDegree = 100) where {T, U<:Number}
    if !(length(size(degrees)) != 1 || length(size(degrees)) != 2)
        throw(ArgumentError("Degrees array must have dimension 1 or 2"))
    end

    if (length(coeffs) != size(degrees, 1))
        throw(ArgumentError("Length of coeffs and degrees must be the same"))
    end

    return HostPolynomial(coeffs, UInt64.(degrees), maxResultingDegree, size(degrees, 2), length(coeffs))
end


function HostPolynomial(edp::EncodedDevicePolynomial{T}, numVars) where {T<:Integer}
    return HostPolynomial(Array(edp.coeffs), decode_degrees(edp.encodedDegrees, numVars, edp.maxResultingDegree), edp.maxResultingDegree)
end

end