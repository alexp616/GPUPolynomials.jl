include("../../utils/ntt_utils.jl")

mutable struct GPUNTTPregen{T<:Integer}
    primeArray::CuVector{T}
    npruArray::CuVector{T}
    thetaArray::CuArray{T}
    len::Int
    log2len::Int
    butterfly
    nttType::DataType
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
    nttType = get_uint_type(Base._nextpow2(Int(ceil(log2(maximum(primeArray))))))

    npruArray = npruarray_generator(primeArray, len)
    # display(npruArray)
    @assert all([i > 0 for i in npruArray])
    thetaArray = generate_theta_m(primeArray, len, log2len, npruArray)
    @assert all([i > 0 for i in thetaArray])
    nttpregen = GPUNTTPregen{eltype(primeArray)}(CuArray(primeArray), CuArray(npruArray), CuArray(thetaArray), len, log2len, butterfly, nttType)

    npruInverseArray = eltype(primeArray).(mod_inverse.(npruArray, primeArray))
    @assert all([i > 0 for i in npruInverseArray])

    inverseThetaArray = generate_theta_m(primeArray, len, log2len, npruInverseArray)
    @assert all([i > 0 for i in npruInverseArray])

    temp = GPUNTTPregen{eltype(primeArray)}(CuArray(primeArray), CuArray(npruInverseArray), CuArray(inverseThetaArray), len, log2len, butterfly, nttType)
    inttpregen = pregen_intt(temp)

    return nttpregen, inttpregen
end

function pregen_intt(nttpregen::GPUNTTPregen{T}) where T<:Integer
    lenInverseArray = CuArray(T.(map(p -> mod_inverse(nttpregen.len, p), Array(nttpregen.primeArray))))
    return GPUINTTPregen{T}(nttpregen, lenInverseArray)
end

function generate_theta_m(primeArray::Vector{T}, len, log2len, npruArray::Vector{T}) where T<:Integer
    result = zeros(eltype(primeArray), length(primeArray), log2len)
    for i in 1:log2len
        m = 1 << i
        result[:, i] = power_mod.(npruArray, len ÷ m, primeArray)
    end

    return result
end

# Assumes vec is already butterflied
function single_gpu_ntt!(vec::CuVector{T}, prime, thetaArray) where T<:Integer
    len = length(vec)
    log2len = Int(ceil(log2(len)))
    kernel = @cuda launch=false single_gpu_ntt_kernel!(vec, prime, prime, 0, 0, 0, 0)
    config = launch_configuration(kernel.fun)
    threads = min(len >> 1, Base._prevpow2(config.threads))
    blocks = cld(len >> 1, threads)

    for i in 1:log2len
        m = 1 << i
        m2 = m >> 1
        magicbits = log2len - i
        magicmask = (1 << magicbits) - 1

        theta_m = thetaArray[i]

        kernel(vec, prime, theta_m, magicmask, magicbits, m, m2; threads = threads, blocks = blocks)
    end

    return nothing
end

function gpu_ntt!(stackedvec::CuArray{T}, pregen::GPUNTTPregen{T}) where T<:Integer
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

function single_gpu_ntt_kernel!(vec::CuDeviceVector{T}, prime::T, theta_m, magicmask, magicbits, m, m2::Int) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = m * (idx & magicmask) + (idx >> magicbits)

    @inbounds begin
        theta = power_mod(theta_m, idx >> magicbits, prime)
        t = mul_mod(theta, vec[k + m2 + 1], prime)
        u = vec[k + 1]

        vec[k + 1] = add_mod(u, t, prime)
        vec[k + m2 + 1] = sub_mod(u, t, prime)
    end

    return nothing
end

function gpu_ntt_kernel!(stackedvec::CuDeviceArray{T}, primeArray::CuDeviceVector{T}, theta_m, magicmask, magicbits, m, m2::Int) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = m * (idx & magicmask) + (idx >> magicbits)

    @inbounds for p in eachindex(primeArray)
        theta = power_mod(theta_m[p], idx >> magicbits, primeArray[p])
        
        t = mul_mod(theta, stackedvec[k + m2 + 1, p], primeArray[p])
        u = stackedvec[k + 1, p]

        stackedvec[k + 1, p] = add_mod(u, t, primeArray[p])
        stackedvec[k + m2 + 1, p] = sub_mod(u, t, primeArray[p])
    end

    return nothing
end

function single_gpu_intt!(vec, prime, thetaArray, lenInverse)
    single_gpu_ntt!(vec, prime, thetaArray)
    len = length(vec)
    kernel = @cuda launch=false single_intt_thing_kernel!(vec, prime, lenInverse)
    config = launch_configuration(kernel.fun)
    threads = min(len, Base._prevpow2(config.threads))
    blocks = cld(len, threads)

    kernel(vec, prime, lenInverse; threads = threads, blocks = blocks)

    return nothing
end

function gpu_intt!(stackedvec::CuArray{T}, pregen::GPUINTTPregen{T}) where T<:Integer
    gpu_ntt!(stackedvec, pregen.nttpregen)
    kernel = @cuda launch=false intt_thing_kernel!(stackedvec, pregen.nttpregen.primeArray, pregen.lenInverseArray)
    config = launch_configuration(kernel.fun)
    threads = min(pregen.nttpregen.len, Base._prevpow2(config.threads))
    blocks = cld(pregen.nttpregen.len, threads)

    kernel(stackedvec, pregen.nttpregen.primeArray, pregen.lenInverseArray; threads = threads, blocks = blocks)

    return nothing
end

function single_intt_thing_kernel!(vec, prime, lenInverse)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    vec[idx] = mul_mod(vec[idx], lenInverse, prime)

    return nothing
end

function intt_thing_kernel!(stackedvec::CuDeviceArray{T}, primeArray::CuDeviceVector{T}, lenInverseArray::CuDeviceVector{T}) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds for i in eachindex(primeArray)
        stackedvec[idx, i] = mul_mod(stackedvec[idx, i], lenInverseArray[i], primeArray[i])
    end

    return nothing
end

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
    println("GPUPowPregen took $(pregentime.time) s to pregenerate")

    return GPUPowPregen{nttType}(CuArray(nttType.(primeArray)), nttpregen, inttpregen, crtpregen, resultType)
end

function memorysafe_gpu_ntt_pow(vec::Vector{<:Integer}, pow::Int; pregen::Union{GPUPowPregen, Nothing} = nothing)
    finalLength = (length(vec) - 1) * pow + 1

    if pregen === nothing
        throw(ArgumentError("Default pregeneration has not been implemented yet."))
    end

    @assert pregen.nttpregen.butterfly isa Vector{Int}

    if eltype(vec) != eltype(pregen.nttpregen.primeArray)
        println("Casting vec ($(eltype(vec))) to pregenerated ntt type ($(eltype(pregen.nttpregen.primeArray)))...")
        vec = eltype(pregen.nttpregen.primeArray).(vec)
    end

    result = zeros(eltype(vec), finalLength, length(pregen.primeArray))

    # Butterflied input on CPU
    cpuvec = vcat(vec, zeros(eltype(vec), pregen.nttpregen.len - length(vec)))
    cpuvec .= cpuvec[pregen.nttpregen.butterfly]
    cpuvecptr = pointer(cpuvec)

    primeArray = Array(pregen.primeArray)
    thetaArray = Array(pregen.nttpregen.thetaArray)

    inttthetaArray = Array(pregen.inttpregen.nttpregen.thetaArray)
    inttinverseArray = Array(pregen.inttpregen.lenInverseArray)

    # Place to store result of broadcast_pow! before butterflying
    temp = zeros(eltype(cpuvec), pregen.nttpregen.len)
    tempptr = pointer(temp)

    # Allocate space to do all of our computations on
    gpuvec = CUDA.zeros(eltype(cpuvec), pregen.nttpregen.len)
    gpuvecptr = pointer(gpuvec)
    for p in eachindex(primeArray)
        CUDA.copyto!(gpuvec, cpuvec)
        prime = primeArray[p]
        # NTT
        single_gpu_ntt!(gpuvec, prime, view(thetaArray, p, :))
        # BROADCAST_POW
        broadcast_pow!(gpuvec, CuArray([prime]), pow)
        # INTT
        CUDA.copyto!(temp, gpuvec)
        # Copy 
        temp .= temp[pregen.nttpregen.butterfly]

        CUDA.copyto!(gpuvec, temp)
        single_gpu_intt!(gpuvec, prime, view(inttthetaArray, p, :), inttinverseArray[p])
        # COPY TO RESULT
        CUDA.unsafe_copyto!(pointer(result) + (p - 1) * finalLength * sizeof(eltype(result)), gpuvecptr, finalLength)
    	println("finished a prime: $p / $(length(primeArray))")
    end

    CUDA.unsafe_free!(gpuvec)

    return result
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
    # display(multimodvec)
    broadcast_pow!(multimodvec, pregen.nttpregen.primeArray, pow)

    gpu_intt!(multimodvec, pregen.inttpregen)

    if docrt
        return build_result(multimodvec, pregen.crtpregen, pregen.resultType)[1:finalLength]
    else
        return multimodvec[1:finalLength, :]
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

function build_result(multimodvec, crtpregen, resultType::DataType, cpureturn = false)
    # result = CUDA.zeros(eltype(crtpregen), finalLength)
    result = CuArray(zeros(eltype(crtpregen), size(multimodvec, 1)))

    kernel = @cuda launch=false build_result_kernel!(multimodvec, crtpregen, result)
    config = launch_configuration(kernel.fun)
    threads = min(length(result), config.threads)
    blocks = cld(length(result), threads)

    kernel(multimodvec, crtpregen, result; threads = threads, blocks = blocks)

    if cpureturn
        cpuresult = Array(result)
        CUDA.unsafe_free!(result)
        return cpuresult
    else
	    return result
    end
end

function sparsify(dense::Array)
    resultLen = 0
    zerovec = zeros(eltype(dense), size(dense, 2))
    for i in axes(dense, 1)
        if view(dense, i, :) != zerovec
            resultLen += 1
        end
    end

    resultCoeffs = zeros(eltype(dense), resultLen, size(dense, 2))
    resultDegrees = zeros(Int64, resultLen)

    curridx = 1
    for i in axes(dense, 1)
        if dense[i, :] != zerovec
            resultCoeffs[curridx, :] .= dense[i, :]
            resultDegrees[curridx] = i
            curridx += 1
        end
    end

    return CuArray(resultCoeffs), CuArray(resultDegrees)
end

function sparsify(dense::CuArray)
    flags = CUDA.zeros(Int32, size(dense, 1))

    kernel1 = @cuda launch=false generate_flags_kernel!(dense, flags)
    config = launch_configuration(kernel1.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel1(dense, flags; threads = threads, blocks = blocks)

    indices = accumulate(+, flags)

    CUDA.@allowscalar resultLen = indices[end]

    resultCoeffs = CUDA.zeros(eltype(dense), resultLen, size(dense, 2))
    resultDegrees = CUDA.zeros(Int64, resultLen)

    kernel2 = @cuda launch=false generate_result_kernel!(dense, flags, indices, resultCoeffs, resultDegrees)
    config = launch_configuration(kernel2.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel2(dense, flags, indices, resultCoeffs, resultDegrees; threads = threads, blocks = blocks)

    return resultCoeffs, resultDegrees
end

function generate_flags_kernel!(dense, flags)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(flags)
        for i in axes(dense, 2)
            if dense[idx, i] != zero(eltype(dense))
                flags[idx] = one(Int32)
            end
        end
    end

    return nothing
end

function generate_result_kernel!(dense, flags, indices, resultCoeffs, resultDegrees)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(flags)
        if flags[idx] != 0
            termNum = indices[idx]
            resultDegrees[termNum] = idx
            for i in axes(dense, 2)
                resultCoeffs[termNum, i] = dense[idx, i]
            end
        end
    end

    return nothing
end

function build_result_kernel!(multimodvec, crtpregen, result)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(result)
        @inbounds result[idx] = crt(view(multimodvec, idx, :), crtpregen)
    end

    return nothing
end
