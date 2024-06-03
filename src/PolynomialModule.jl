module PolynomialModule
using CUDA
global MAX_THREADS_PER_BLOCK = 512

function encode_degrees(degrees::Array{T, 2}, maxResultingDegree)::Array{} where {T<:Unsigned}
    if (size(degrees, 1) > MAX_THREADS_PER_BLOCK)
        numPaddedRows = cld(size(degrees, 1), MAX_THREADS_PER_BLOCK) * MAX_THREADS_PER_BLOCK - size(degrees, 1)
        cu_degrees = CuArray(vcat(degrees, fill(zero(Int), (numPaddedRows, size(degrees, 2)))))
    else
        cu_degrees = CuArray(degrees)
    end

    nthreads = min(size(cu_degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = cld(size(cu_degrees, 1), nthreads)

    result = CUDA.zeros(UInt32, length(cu_degrees))
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        encode_degrees_kernel!(cu_degrees, maxResultingDegree, result, size(degrees, 2))
    )
    return Array(result)[1:size(degrees, 1)]
end

function encode_degrees_kernel!(cu_degrees, maxResultingDegree, result, numVars)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    for i in 1:numVars
        result[tid] += cu_degrees[tid, i] * (maxResultingDegree) ^ (i - 1)
    end
    return nothing
end

function decode_degrees(degrees::CuArray{T, 1}, numVars, maxResultingDegree)::Array{T, 2} where {T<:Unsigned}
    if (length(degrees) > MAX_THREADS_PER_BLOCK)
        numPaddedRows = cld(length(degrees), MAX_THREADS_PER_BLOCK) * MAX_THREADS_PER_BLOCK - length(degrees)
        cu_degrees = CuArray(vcat(degrees, fill(zero(Int), numPaddedRows)))
    else
        cu_degrees = CuArray(degrees)
    end

    nthreads = min(size(cu_degrees, 1), MAX_THREADS_PER_BLOCK)
    nblocks = cld(size(cu_degrees, 1), nthreads)

    result = CUDA.zeros(T, (length(cu_degrees), numVars))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        decode_degrees_kernel!(cu_degrees, maxResultingDegree, result, numVars)
    )

    return Array(result)[1:size(degrees, 1), :]
end

function decode_degrees_kernel!(cu_degrees, maxResultingDegree, result, numVars)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    val = cu_degrees[tid]

    for i in numVars:-1:1
        x = fld(val, maxResultingDegree ^ (i - 1))
        result[tid, i] = x
        val -= x * maxResultingDegree ^ (i - 1)
    end

    # @cuassert val = 0
    return nothing
end

mutable struct HostPolynomial{T}
    coeffs::Array{T, 1}
    degrees::Array{UInt32, 2}
    maxResultingDegree::Int
    numVars::Int
    numTerms::Int
end

mutable struct EncodedHostPolynomial{T}
    coeffs::Array{T, 1}
    encodedDegrees::Array{UInt32, 1}
    maxResultingDegree::Int
    numTerms::Int
end

function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, maxResultingDegree = 100) where {T, U<:Integer}
    if !(length(size(degrees)) != 1 || length(size(degrees)) != 2)
        throw(ArgumentError("Degrees array must have dimension 1 or 2"))
    end

    if (length(coeffs) != size(degrees, 1))
        throw(ArgumentError("Length of coeffs and degrees must be the same"))
    end

    return HostPolynomial(coeffs, UInt32.(degrees), maxResultingDegree, size(degrees, 2), length(coeffs))
end


function HostPolynomial(ehp::EncodedHostPolynomial{T}, numVars) where {T<:Integer}
    return HostPolynomial(ehp.coeffs, decode_degrees(CuArray(ehp.encodedDegrees), numVars, ehp.maxResultingDegree), ehp.maxResultingDegree)
end


function EncodedHostPolynomial(hp::HostPolynomial{T}) where {T<:Integer}
    return EncodedHostPolynomial(hp.coeffs, encode_degrees(hp.degrees, hp.maxResultingDegree), hp.maxResultingDegree, hp.numTerms)
end

# Constructor with empty values
function EncodedHostPolynomial(numTerms::Int, maxResultingDegree::Int, T::DataType)
    return EncodedHostPolynomial(zeros(T, numTerms), zeros(Int, numTerms), maxResultingDegree, numTerms)
end

end