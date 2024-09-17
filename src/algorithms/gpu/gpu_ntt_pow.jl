include("gpu_ntt.jl")

module GPUNTTPow

include("../../utils/ntt_utils.jl")
using ..GPUNTT

export GPUPowPregen, get_fft_size, pregen_gpu_pow, gpu_ntt_pow

mutable struct GPUPowPregen{T<:Integer}
    primeArray::CuVector{T}
    nttpregen::GPUNTTPregen{T}
    inttpregen::GPUINTTPregen{T}
    crtpregen::CuArray{<:Integer}
    resultType::DataType
end

function pregen_gpu_pow(primeArray::Vector{<:Integer}, fftSize)
    pregentime = @timed begin
        nttType, crtType, resultType = get_types(primeArray)
        nttpregen, inttpregen = pregen_ntt(nttType.(primeArray), fftSize)
        crtpregen = CuArray(pregen_crt(crtType.(primeArray)))
    end

    return GPUPowPregen{nttType}(CuArray(nttType.(primeArray)), nttpregen, inttpregen, crtpregen, resultType)
end

function gpu_ntt_pow(vec::CuVector{<:Integer}, pow::Int; pregen::Union{GPUPowPregen, Nothing} = nothing, docrt = true)
    finalLength = (length(vec) - 1) * pow + 1

    if pregen === nothing
        throw(ArgumentError("Default pregeneration has not been implemented yet."))
    end

    if eltype(vec) != eltype(pregen.nttpregen.primeArray)
        println("Casting vec ($(eltype(vec))) to pregenerated ntt type ($(eltype(pregen.nttpregen.primeArray)))...")
        vec = eltype(pregen.nttpregen.primeArray).(vec)
    end

    multimodvec = repeat(vcat(vec, zeros(eltype(vec), pregen.nttpregen.len - length(vec))), 1, length(pregen.primeArray))

    gpu_ntt!(multimodvec, pregen.nttpregen)

    broadcast_pow!(multimodvec, pregen.nttpregen.primeArray, pow)

    gpu_intt!(multimodvec, pregen.inttpregen)

    if length(pregen.primeArray) == 1
        return pregen.resultType.(multimodvec)[1:finalLength]
    elseif docrt
        return build_result(multimodvec, pregen.crtpregen, finalLength, pregen.resultType)
    else
        return eltype(pregen.crtpregen).(multimodvec[1:finalLength])
    end
end

function broadcast_pow!(multimodvec, primeArray, pow)
    kernel = @cuda launch=false broadcast_pow_kernel!(multimodvec, primeArray, pow)
    config = launch_configuration(kernel.fun)
    threads = min(size(multimodvec, 1), Base._prevpow2(config.threads))
    blocks = cld(size(multimodvec, 1), threads)

    kernel(multimodvec, primeArray, pow; threads = threads, blocks = blocks)
end

function broadcast_pow_kernel!(multimodvec, primeArray, pow)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds for p in axes(multimodvec, 2)
        multimodvec[idx, p] = power_mod(multimodvec[idx, p], pow, primeArray[p])
    end
end

function build_result(multimodvec, crtpregen, finalLength, resultType)::CuVector
    result = CUDA.zeros(eltype(crtpregen), finalLength)
    # zerovec = CUDA.zeros(eltype(multimodvec), size(multimodvec, 2))

    kernel = @cuda launch=false build_result_kernel!(multimodvec, crtpregen, result)
    config = launch_configuration(kernel.fun)
    threads = min(finalLength, Base._prevpow2(config.threads))
    blocks = cld(finalLength, threads)

    kernel(multimodvec, crtpregen, result; threads = threads, blocks = blocks)

    return resultType.(result)
end

function build_result_kernel!(multimodvec, crtpregen, result)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(result)
        @inbounds result[idx] = crt(view(multimodvec, idx, :), crtpregen)
    end

    return nothing
end

end