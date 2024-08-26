module GPUPow

export GPUPowPregen, pregen_gpu_pow, gpu_pow

include("ntt_utils.jl")

using ..Polynomials
using CUDA

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
    crtPregen::CuArray{T} where T<:Integer
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
function gpu_ntt_kernel!(vec::CuDeviceArray{Int}, primearray::CuDeviceVector{Int}, theta_m::CuDeviceVector{Int}, magic::Int, m2::Int)
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
function build_result(multimodularResultArr::CuArray{Int, 2}, crtPregen::CuArray{T, 2}, resultType::DataType) where T<:Integer
    # @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = T.(multimodularResultArr[1, :])

    kernel = @cuda launch=false build_result_kernel(result, multimodularResultArr, crtPregen)
    config = launch_configuration(kernel.fun)
    threads = min(length(result), prevpow(2, config.threads))
    blocks = cld(length(result), threads)

    kernel(result, multimodularResultArr, crtPregen; threads = threads, blocks = blocks)

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