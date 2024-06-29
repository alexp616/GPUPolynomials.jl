include("ntt_utils.jl")

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
    crtType::DataType
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

    return GPUPowPregen(primeArray, npruArray, npruInverseArray, len, log2len, lenInverseArray, pregenButterfly, crtType, resultType)
end

"""
    gpu_pow(vec, pow; pregen)

Raises polynomial represented by `vec` to the `pow` power. Current implementation requires a pregenerated object to work
"""
function gpu_pow(vec::CuVector{Int}, pow::Int; pregen::GPUPowPregen)
    finalLength = (length(vec) - 1) * pow + 1

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
    result = build_result(multimodularResultArr, pregen.primeArray, pregen.crtType, pregen.resultType)[1:finalLength]

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
            arr[i, idx] = power_mod(arr[i, idx], pow, cu_primearray[i])
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

    for i in 1:length(pregen.primeArray)
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
function gpu_ntt_kernel!(vec, primearray, theta_m, magic::Int, m2::Int)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = Int(2 * m2 * (idx % magic) + floor(idx/magic))

    for p in eachindex(primearray)
        theta = power_mod(theta_m[p], idx ÷ magic, primearray[p])
        t = theta * vec[p, k + m2 + 1]
        u = vec[p, k + 1]

        vec[p, k + 1] = faster_mod(u + t, primearray[p])
        vec[p, k + m2 + 1] = faster_mod(u - t, primearray[p])
    end

    return 
end

"""
    build_result(multimodularResultArr, primearray, crtType, resultType)

Combines all rows of multimodularResultArr into one using Chinese Remainder Theorem
"""
function build_result(multimodularResultArr::CuArray{Int, 2}, primearray::Vector{Int}, crtType::DataType, resultType::DataType)
    # @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = crtType.(multimodularResultArr[1, :])
    currmod = crtType(primearray[1])

    kernel = @cuda launch=false build_result_kernel(result, multimodularResultArr, 0, currmod, 0)
    config = launch_configuration(kernel.fun)
    threads = Base.min(length(result), prevpow(2, config.threads))
    blocks = cld(length(result), threads)

    for i in 2:size(multimodularResultArr, 1)
        kernel(result, multimodularResultArr, i, currmod, primearray[i]; threads = threads, blocks = blocks)
        currmod *= primearray[i]
    end

    return resultType.(result)
end

"""
    build_result_kernel(result, multimodularResultArr, row, currmod, newprime)

Kernel function for build_result()
"""
function build_result_kernel(result, multimodularResultArr, row, currmod, newprime)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    result[idx] = chinese_remainder_two(result[idx], currmod, multimodularResultArr[row, idx], newprime)

    return
end
