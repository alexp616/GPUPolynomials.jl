module Polynomials
using CUDA

include("ReduceByKey.jl")
include("gpu_merge.jl")
export HostPolynomial, SparseDevicePolynomial, SparsePolynomial, encode_degrees, decode_degrees, copy, add

mutable struct HostPolynomial{T}
    coeffs::Vector{T}
    degrees::Array{Int, 2}
    key::Int
    numVars::Int
    numTerms::Int
end

mutable struct SparseDevicePolynomial{T}
    coeffs::CuVector{T}
    encodedDegrees::CuVector{Int}
    key::Int
    numTerms::Int
end

function SparseDevicePolynomial(hp::HostPolynomial, sorted = false)
    coeffs = CuArray(hp.coeffs)
    degrees = CuArray(encode_degrees(hp.degrees, hp.key))
    perm = sortperm(degrees)
    degrees = degrees[perm]
    coeffs = coeffs[perm]
    return SparseDevicePolynomial(coeffs, degrees, hp.key, hp.numTerms)
end

function Base.copy(sdp::SparseDevicePolynomial{T}) where T<:Real
    return SparseDevicePolynomial(copy(sdp.coeffs), copy(sdp.encodedDegrees), sdp.key, sdp.numTerms)
end

function add(sdp1::SparseDevicePolynomial{T}, sdp2::SparseDevicePolynomial{T}) where T<:Real
    urc, urd = merge_by_key(sdp1.encodedDegrees, sdp1.coeffs, sdp2.encodedDegrees, sdp2.coeffs)
    rc, rd = reduce_by_key(urc, urd)

    return SparseDevicePolynomial(rc, rd, sdp1.key, length(rc))
end


# function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, key = 100) where {T, U<:Integer}
function HostPolynomial(coeffs::Array{T, 1}, degrees::Array{U}, key = 100) where {T, U<:Number}
    if !(length(size(degrees)) != 1 || length(size(degrees)) != 2)
        throw(ArgumentError("Degrees array must have dimension 1 or 2"))
    end

    if (length(coeffs) != size(degrees, 1))
        throw(ArgumentError("Length of coeffs and degrees must be the same"))
    end

    return HostPolynomial(coeffs, Int.(degrees), key, size(degrees, 2), length(coeffs))
end


function HostPolynomial(edp::SparseDevicePolynomial{T}, numVars) where {T<:Integer}
    return HostPolynomial(Array(edp.coeffs), decode_degrees(edp.encodedDegrees, numVars, edp.key), edp.key)
end


function encode_degrees(degrees::Array{Int, 2}, key)::CuVector{Int}
    result = CUDA.zeros(Int, size(degrees, 1))
    cu_degrees = CuArray(degrees)

    kernel = @cuda launch = false encode_degrees_kernel!(cu_degrees, key, result, size(degrees, 2))
    config = launch_configuration(kernel.fun)
    threads = min(size(degrees, 1), config.threads)
    blocks = cld(size(degrees, 1), threads)

    kernel(cu_degrees, key, result, size(degrees, 2), threads = threads, blocks = blocks)

    return result
end

function encode_degrees_kernel!(cu_degrees, key, result, numVars)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if tid <= size(cu_degrees, 1)
        for i in 1:numVars
            result[tid] += cu_degrees[tid, i] * (key) ^ (i - 1)
        end
    end

    return nothing
end

function decode_degrees(degrees::CuVector{T}, numVars, key)::Array{T, 2} where {T<:Number}
    result = CUDA.zeros(T, (length(degrees), numVars))

    kernel = @cuda launch = false decode_degrees_kernel!(degrees, key, result, numVars)
    config = launch_configuration(kernel.fun)
    threads = min(length(degrees), config.threads)
    blocks = cld(length(degrees), threads)

    kernel(degrees, key, result, numVars; threads = threads, blocks = blocks)

    return Array(result)
end

function decode_degrees_kernel!(cu_degrees, key, result, numVars)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= length(cu_degrees)
        val = cu_degrees[tid]
        for i in numVars:-1:1
            x = fld(val, key ^ (i - 1))
            result[tid, i] = x
            val -= x * key ^ (i - 1)
        end
    end

    return nothing
end

end