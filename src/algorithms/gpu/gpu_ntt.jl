module GPUNTT

using CUDA

include("../../utils/ntt_utils.jl")

export GPUNTTPregen, GPUINTTPregen, pregen_ntt, pregen_intt, gpu_ntt!, gpu_intt!

mutable struct GPUNTTPregen{T<:Integer}
    primeArray::CuVector{T}
    npruArray::CuVector{T}
    thetaArray::CuArray{T}
    len::Int
    log2len::Int
    butterfly::CuVector{Int}
end

mutable struct GPUINTTPregen{T<:Integer}
    nttpregen::GPUNTTPregen{T}
    lenInverseArray::CuVector{T}
end

function pregen_ntt(primeArray::Vector{<:Integer}, len)
    if !ispow2(len)
        throw(ArgumentError("len must be a power of 2."))
    end

    butterfly = generate_butterfly_permutations(len)

    lenInverseArray = map(p -> mod_inverse(len, p), primeArray)
    @assert all([i > 0 for i in lenInverseArray])
    log2len = Int(log2(len))

    npruArray = npruarray_generator(primeArray, len)
    # display(npruArray)
    @assert all([i > 0 for i in npruArray])
    thetaArray = generate_theta_m(primeArray, len, log2len, npruArray)
    @assert all([i > 0 for i in thetaArray])
    nttpregen = GPUNTTPregen{eltype(primeArray)}(CuArray(primeArray), CuArray(npruArray), CuArray(thetaArray), len, log2len, butterfly)

    npruInverseArray = eltype(primeArray).(mod_inverse.(npruArray, primeArray))
    @assert all([i > 0 for i in npruInverseArray])

    inverseThetaArray = generate_theta_m(primeArray, len, log2len, npruInverseArray)
    @assert all([i > 0 for i in npruInverseArray])

    temp = GPUNTTPregen{eltype(primeArray)}(CuArray(primeArray), CuArray(npruInverseArray), CuArray(inverseThetaArray), len, log2len, butterfly)
    inttpregen = pregen_intt(temp)

    return nttpregen, inttpregen
end

function pregen_intt(nttpregen::GPUNTTPregen{T}) where T<:Integer
    lenInverseArray = CuArray(T.(map(p -> mod_inverse(nttpregen.len, p), Array(nttpregen.primeArray))))
    return GPUINTTPregen{T}(nttpregen, lenInverseArray)
end

function generate_theta_m(primeArray, len, log2len, npruArray)
    result = zeros(eltype(primeArray), length(primeArray), log2len)
    for i in 1:log2len
        m = 1 << i
        result[:, i] = power_mod.(npruArray, len ÷ m, primeArray)
    end

    return result
end

function gpu_ntt!(stackedvec::CuArray{T}, pregen::GPUNTTPregen{T}) where T<:Integer
    if size(stackedvec, 1) != pregen.len
        throw(ArgumentError("Vector doesn't have same length as pregen object. Vector length: $(size(stackedvec, 1)), pregen.len: $(pregen.len)"))
    end
    stackedvec .= stackedvec[pregen.butterfly, :]

    kernel = @cuda launch=false gpu_ntt_kernel!(stackedvec, pregen.primeArray, pregen.primeArray, 0, 0, 0, 0)
    config = launch_configuration(kernel.fun)
    threads = min(pregen.len >> 1, Base._prevpow2(config.threads))
    blocks = cld(pregen.len >> 1, threads)

    for i in 1:pregen.log2len
        m = 1 << i
        m2 = m >> 1
        magicbits = pregen.log2len - i
        magicmask = (1 << magicbits) - 1
        
        theta_m = view(pregen.thetaArray, :, i)

        kernel(stackedvec, pregen.primeArray, theta_m, magicmask, magicbits, m, m2; threads = threads, blocks = blocks)
    end

    return nothing
end

function gpu_intt!(stackedvec::CuArray{T}, pregen::GPUINTTPregen{T}) where T<:Integer
    if size(stackedvec, 1) != pregen.nttpregen.len
        throw(ArgumentError("Vector doesn't have same length as pregen object."))
    end

    gpu_ntt!(stackedvec, pregen.nttpregen)

    kernel = @cuda launch=false intt_thing_kernel!(stackedvec, pregen.nttpregen.primeArray, pregen.lenInverseArray)
    config = launch_configuration(kernel.fun)
    threads = min(pregen.nttpregen.len, Base._prevpow2(config.threads))
    blocks = cld(pregen.nttpregen.len, threads)

    kernel(stackedvec, pregen.nttpregen.primeArray, pregen.lenInverseArray; threads = threads, blocks = blocks)

    return nothing
end

function gpu_ntt_kernel!(stackedvec::CuDeviceArray{T}, primeArray::CuDeviceVector{T}, theta_m, magicmask, magicbits, m, m2::Int) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = m * (idx & magicmask) + (idx >> magicbits)

    @inbounds for p in eachindex(primeArray)
        theta = power_mod(theta_m[p], idx >> magicbits, primeArray[p])
        t = theta * stackedvec[k + m2 + 1, p]
        u = stackedvec[k + 1, p]

        stackedvec[k + 1, p] = mod(u + t, primeArray[p])
        stackedvec[k + m2 + 1, p] = sub_mod(u, t, primeArray[p])
    end

    return nothing
end

function intt_thing_kernel!(stackedvec::CuDeviceArray{T}, primeArray::CuDeviceVector{T}, lenInverseArray::CuDeviceVector{T}) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds for i in eachindex(primeArray)
        stackedvec[idx, i] = mod(stackedvec[idx, i] * lenInverseArray[i], primeArray[i])
    end

    return nothing
end

end