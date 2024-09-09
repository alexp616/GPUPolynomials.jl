module GPUPow

export GPUPowPregen, pregen_gpu_pow, gpu_pow, gpu_pow_cpu_crt, do_crt

include("ntt_utils.jl")

using ..Polynomials
using CUDA
using BitIntegers

"""
    GPUPowPregen

Struct that contains all information needed for quickly executing gpu_pow
"""
struct GPUPowPregen
    primeArray::Vector{Int}
    npruArray::Vector{Int}
    npruInverseArray::Vector{Int}
    len::Int
    log2len::Int
    lenInverseArray::Vector{Int}
    pregenButterfly::CuVector{Int}
    crtPregen::Array{T} where T<:Integer
    resultType::DataType
end

"""
    pregen_gpu_pow(primeArray, len, crtType, resultType)

Takes a primeArray, a FFT length, and two datatypes to generate a 
GPUPowPregen object.

# Arguments
- `primeArray::Vector{Int}`: Array of prime numbers of the form k * len + 1, whose product is an upper bound for the largest coefficient of the resulting polynomial
- `len::Int`: Length of resulting FFT
- `crtType::DataType`: Type used for intermediate steps of CRT. Doesn't necessarily have to be the same as resultType
- `resultType::DataType`: Type used for result of CRT, or ending result
"""
function pregen_gpu_pow(primeArray::Vector{Int}, len::Int, crtType::DataType, resultType::DataType)::GPUPowPregen
    @assert ispow2(len) "len must be a power 2"

    npruArray = npruarray_generator(primeArray, len)
    npruInverseArray = mod_inverse.(npruArray, primeArray)
    log2len = Int(log2(len))
    lenInverseArray = map(p -> mod_inverse(len, p), primeArray)
    pregenButterfly = generate_butterfly_permutations(len)
    crtPregen = pregen_crt(primeArray, crtType)

    return GPUPowPregen(primeArray, npruArray, npruInverseArray, len, log2len, lenInverseArray, pregenButterfly, crtPregen, resultType)
end

"""
    gpu_pow(vec, pow; pregen)

Raises polynomial represented by `vec` to the `pow` power. Current implementation requires a pregenerated object to work
"""
function gpu_pow(vec::CuVector{Int}, pow::Int; pregen::Union{GPUPowPregen, Nothing} = nothing)
    finalLength = (length(vec) - 1) * pow + 1

    if pregen === nothing
        println("Using default pregeneration! (not recommended)")
        pregen = pregen_gpu_pow([13631489, 23068673], Base._nextpow2(finalLength), Int128, Int)
    end

    # Padding, stacking for multiple prime modulus, and butterflying indices
    stackedvec = repeat(((vcat(vec, zeros(Int, pregen.len - length(vec))))[pregen.pregenButterfly])', length(pregen.primeArray), 1)

    # DFT
    gpu_ntt!(stackedvec, pregen.primeArray, pregen.npruArray, pregen.len, pregen.log2len)

    # Broadcasting power
    broadcast_pow!(stackedvec, pregen.primeArray, pow)
    # IDFT
    multimodularResultArr = gpu_intt(stackedvec, pregen)

    # here for memories of debugging
    # if pregen.len > 100000
    #     # temp = Array(multimodularResultArr)
    #     # temp[:, 156681] .= [11, 11, 11]
    #     # multimodularResultArr = CuArray(temp)
    #     println("three coeffs of term [17, 82, 5, 64]: ", Array(multimodularResultArr)[:, 156681])
    #     # println("two coeffs of term [15, 20, 39, 6]: ", Array(multimodularResultArr)[:, 257515])
    #     println("primearray: ", pregen.primeArray)
    # end

    # CRT
    result = build_result(multimodularResultArr, pregen.crtPregen, pregen.resultType)[1:finalLength]

    return result
end

function pregen_gpu_pow(hp::HomogeneousPolynomial, pow::Int, primeArray = [13631489, 23068673])
    key = hp.homogeneousDegree * pow + 1
    inputLen = hp.homogeneousDegree * (key) ^ (size(hp.degrees, 2) - 2) + 1

    return pregen_gpu_pow(primeArray, Base._nextpow2((inputLen - 1) * pow + 1), Int128, Int) 
end

"""
    gpu_pow(hp, pow; pregen)

Overload of gpu_pow method that takes in HomogeneousPolynomial object and returns HomogeneousPolynomial object.
"""
function gpu_pow(hp::HomogeneousPolynomial{T}, pow::Int; pregen::Union{GPUPowPregen, Nothing} = nothing) where T<:Integer
    key = hp.homogeneousDegree * pow + 1
    inputLen = hp.homogeneousDegree * (key) ^ (size(hp.degrees, 2) - 2) + 1
    if pregen === nothing
        println("Using default pregeneration... (not recommended)")
        
        pregenTime = CUDA.@timed begin 
            pregen = pregen_gpu_pow([13631489, 23068673], Base._nextpow2((inputLen - 1) * pow + 1), Int128, Int)
        end
        println("Pregeneration took $(pregenTime.time) s")
    end

    denseHP = CuArray(Polynomials.kronecker_substitution(hp, key, inputLen))
    denseResult = gpu_pow(denseHP, pow; pregen = pregen)
    result = decode_kronecker_substitution(denseResult, key, size(hp.degrees, 2), key - 1)

    return result
end

function gpu_pow_cpu_crt(hp::HomogeneousPolynomial{T}, pow::Int, key = 0; pregen::Union{GPUPowPregen, Nothing} = nothing) where T<:Integer
    if key == 0
        key = hp.homogeneousDegree * pow + 1
    end

    inputLen = hp.homogeneousDegree * (key) ^ (size(hp.degrees, 2) - 2) + 1
    if pregen === nothing
        println("Using default pregeneration... (not recommended)")
        
        pregenTime = CUDA.@timed begin 
            pregen = pregen_gpu_pow([13631489, 23068673], Base._nextpow2((inputLen - 1) * pow + 1), Int128, Int)
        end
        println("Pregeneration took $(pregenTime.time) s")
    end
    ntttime = CUDA.@timed begin
        vec = CuArray(kronecker_substitution(hp, key, inputLen))

        stackedvec = repeat(((vcat(vec, zeros(T, pregen.len - length(vec))))[pregen.pregenButterfly])', length(pregen.primeArray), 1)

        gpu_ntt!(stackedvec, pregen.primeArray, pregen.npruArray, pregen.len, pregen.log2len)

        broadcast_pow!(stackedvec, pregen.primeArray, pow)

        gpumultimod = gpu_intt(stackedvec, pregen)
        CUDA.unsafe_free!(stackedvec)
    end

    # println("ntt took $(ntttime.time) s")

    sparsifytime = CUDA.@timed begin
        multimodResult, resultDegrees = sparsify(gpumultimod, size(hp.degrees, 2), key, hp.homogeneousDegree * pow)
        CUDA.unsafe_free!(gpumultimod)
        multimodResult = pregen.crtType.(multimodResult)
        resultCoeffs = zeros(pregen.resultType, size(multimodResult, 2))
    end
    # println("sparsifying took $(sparsifytime.time) s")

    crtpregen = crtType.(pregen.crtPregen)
    
    # println("converting everything to bigint took $(biginttime.time) s")

    do_crt(multimodResult, crtpregen; dest = resultCoeffs)
    # for col in axes(multimodResult, 2)
    #     x = multimodResult[1, col]
    #     for i in axes(crtpregen, 2)
    #         x = mod(x * crtpregen[2, i] + multimodResult[i + 1, col] * crtpregen[1, i], crtpregen[3, i])
    #     end
    #     resultCoeffs[col] = pregen.resultType(x)
    # end
    # println("crt took $(crttime.time) s")

    return HomogeneousPolynomial(resultCoeffs, resultDegrees)
end

function do_crt(multimodarr, pregen)
    result = zeros(eltype(multimodarr), size(multimodarr, 2))

    for col in axes(multimodarr, 2)
        x = multimodarr[1, col]
        for i in axes(pregen, 2)
            x = mod(x * pregen[2, i] + multimodarr[i + 1, col] * pregen[1, i], pregen[3, i])
        end
        result[col] = x
    end

    return result
end

function do_crt(multimodarr, pregen; dest)
    for col in axes(multimodarr, 2)
        x = multimodarr[1, col]
        for i in axes(pregen, 2)
            x = mod(x * pregen[2, i] + multimodarr[i + 1, col] * pregen[1, i], pregen[3, i])
        end
        dest[col] = x
    end

    return
end

# function generate_flags_kernel!(flags::CuVector{Int32}, multimodarr::CuMatrix{T}) where T<:Integer
function generate_flags_kernel!(flags, multimodarr)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(flags)
        # if idx == 2
        #     for i in axes(multimodarr, 1)
        #         @cuprintln(multimodarr[])
        #     end
            
        # end
        for i in axes(multimodarr, 1)
            if multimodarr[i, idx] != 0
                flags[idx] = 1
                return
            end
        end
    end

    return
end

function generate_result_kernel!(multimodarr, flags, indices, resultmultimodarr, resultDegrees, numVars, key, totalDegree)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if idx <= length(flags)
        if flags[idx] != 0
            num = idx - 1
            a = num
            termNum = indices[idx]
            for i in 1:numVars - 1
                num, r = divrem(num, key)
                resultDegrees[termNum, i] = r
                totalDegree -= r
            end
            
            resultDegrees[termNum, numVars] = totalDegree
            for i in axes(multimodarr, 1)
                resultmultimodarr[i, termNum] = multimodarr[i, idx]
            end
        end
    end

    return
end

function sparsify(multimodarr::Union{CuMatrix{T}, Matrix{T}}, numVars, key, totalDegree) where T<:Integer
    inputtype = typeof(multimodarr)
    if multimodarr isa Matrix{T}
        multimodarr = CuArray(multimodarr)
    end
    flags = CUDA.zeros(Int32, size(multimodarr, 2))

    # Initialize one thread for each column
    kernel = @cuda launch = false generate_flags_kernel!(flags, multimodarr)
    config = launch_configuration(kernel.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel(flags, multimodarr; threads = threads, blocks = blocks)

    indices = accumulate(+, flags)

    CUDA.@allowscalar resultLen = indices[end]

    resultmultimodarr = CUDA.zeros(T, size(multimodarr, 1), resultLen)
    resultDegrees = CUDA.zeros(Int, resultLen, numVars)

    kernel2 = @cuda launch = false generate_result_kernel!(multimodarr, flags, indices, resultmultimodarr, resultDegrees, numVars, key, totalDegree)
    config = launch_configuration(kernel.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel2(multimodarr, flags, indices, resultmultimodarr, resultDegrees, numVars, key, totalDegree; threads = threads, blocks = blocks)

    return Array(resultmultimodarr), Array(resultDegrees)
end

"""
    broadcast_pow!(arr, primearray, pow)

Raises the i-th row of `arr` to `pow` mod `primearray[i]`.
"""
function broadcast_pow!(arr::CuArray, primearray::Vector{Int}, pow::Int)
    @assert size(arr, 1) == length(primearray)

    cu_primearray = CuArray(primearray)

    function broadcast_pow_kernel!(arr, cu_primearray, pow)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        for i in axes(arr, 1)
            @inbounds arr[i, idx] = power_mod(arr[i, idx], pow, cu_primearray[i])
        end

        return
    end

    kernel = @cuda launch=false broadcast_pow_kernel!(arr, cu_primearray, pow)
    config = launch_configuration(kernel.fun)
    threads = min(size(arr, 2), prevpow(2, config.threads))
    blocks = cld(size(arr, 2), threads)

    kernel(arr, cu_primearray, pow; threads = threads, blocks = blocks)

    return
end

"""
    gpu_intt(vec, pregen)

Computes inverse NTT of vec. Note that because I'm a bad coder, the input of this shouldn't be butterflied
"""
function gpu_intt(vec::CuArray{Int, 2}, pregen::GPUPowPregen)
    arg1 = vec[:, pregen.pregenButterfly]
    result = gpu_ntt!(arg1, pregen.primeArray, pregen.npruInverseArray, pregen.len, pregen.log2len)

    @inbounds for i in 1:length(pregen.primeArray)
        # result[i, :] .= map(x -> faster_mod(x * mod_inverse(len, primearray[i]), primearray[i]), result[i, :])
        result[i, :] .*= pregen.lenInverseArray[i]
        result[i, :] .%= pregen.primeArray[i]
    end

    return result
end

"""
    gpu_ntt!(vec, primeArray, npruArray, len, log2len)

Computes NTT of vec.

# Arguments
- `stackedvec`: 2-dimensional array, with each row `i` containing the NTT of the original vec mod `primeArray[i]`
- `primeArray`: Array of selected NTT primes
- `npruArray`: Array of `len`-th principal roots of unity mod the primes in primeArray
- `len`: Length of the FFT
- `log2len`: log2(len)
"""
function gpu_ntt!(stackedvec::CuArray{Int}, primeArray, npruArray, len, log2len)
    cu_primearray = CuArray(primeArray)
    kernel = @cuda launch=false gpu_ntt_kernel!(stackedvec, cu_primearray, cu_primearray, 0, 0)
    config = launch_configuration(kernel.fun)
    threads = min(len ÷ 2, prevpow(2, config.threads))
    blocks = cld(len ÷ 2, threads)

    for i in 1:log2len
        m = 1 << i
        m2 = m >> 1
        magic = 1 << (log2len - i)
        theta_m = CuArray(powermod.(npruArray, Int(len / m), primeArray))

        kernel(stackedvec, cu_primearray, theta_m, magic, m2; threads=threads, blocks=blocks)
    end

    return stackedvec
end

"""
    gpu_ntt_kernel!(vec, primearray, theta_m, magic, m2)

Kernel function for gpu_ntt!()
"""
function gpu_ntt_kernel!(vec::CuDeviceArray{T}, primearray::CuDeviceVector{Int}, theta_m::CuDeviceVector{Int}, magic::Int, m2::Int) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = Int(2 * m2 * (idx % magic) + floor(idx/magic))

    @inbounds for p in eachindex(primearray)
        theta = power_mod(theta_m[p], idx ÷ magic, primearray[p])
        t = theta * vec[p, k + m2 + 1]
        u = vec[p, k + 1]
        
        vec[p, k + 1] = faster_mod(u + t, primearray[p])
        vec[p, k + m2 + 1] = faster_mod(u - t, primearray[p])
    end

    return 
end

"""
    build_result(multimodularResultArr, crtPregen, resultType)

Combines all rows of multimodularResultArr using Chinese Remainder Theorem
"""
function build_result(multimodularResultArr::CuArray{Int, 2}, crtPregen::Array{T, 2}, resultType::DataType) where T<:Integer
    # @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = T.(multimodularResultArr[1, :])
    cu_crtPregen = CuArray(crtPregen)

    kernel = @cuda launch=false build_result_kernel(result, multimodularResultArr, cu_crtPregen)
    config = launch_configuration(kernel.fun)
    threads = min(length(result), prevpow(2, config.threads))
    blocks = cld(length(result), threads)

    kernel(result, multimodularResultArr, cu_crtPregen; threads = threads, blocks = blocks)

    return resultType.(result)
end

"""
    build_result_kernel(result, multimodularResultArr, pregen)

Kernel function for build_result()
"""
function build_result_kernel(result::CuDeviceVector, multimodularResultArr::CuDeviceArray{Int, 2}, pregen::CuDeviceArray)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds begin
        x = multimodularResultArr[1, idx]
        for i in axes(pregen, 2)
            x = mod(x * pregen[2, i] + multimodularResultArr[i + 1, idx] * pregen[1, i], pregen[3, i])
        end
        result[idx] = x
    end

    return 
end

end