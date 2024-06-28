include("ntt_utils.jl")

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

function pregen_gpu_pow(;primeArray::Vector{Int}, len::Int, crtType::DataType, resultType::DataType)
    @assert ispow2(len) "len must be a power 2"

    npruArray = npruarray_generator(primeArray, len)
    npruInverseArray = mod_inverse.(npruArray, primeArray)
    log2len = Int(log2(len))
    lenInverseArray = map(p -> mod_inverse(len, p), primeArray)
    pregenButterfly = generate_butterfly_permutations(len)

    return GPUPowPregen(primeArray, npruArray, npruInverseArray, len, log2len, lenInverseArray, pregenButterfly, crtType, resultType)
end

function generate_butterfly_permutations(n::Int)::CuVector{Int}
    @assert ispow2(n) "n must be a power of 2"
    perm = parallelBitReverseCopy(CuArray([i for i in 1:n]))
    return perm
end

function bit_reverse(x::Integer, log2n::Integer)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

function gpu_pow(p1::CuVector{Int}, pow::Int; pregen::GPUPowPregen)
    finalLength = (length(p1) - 1) * pow + 1

    # Padding, stacking for multiple prime modulus, and butterflying indices
    stackedp1 = repeat(((vcat(p1, zeros(Int, pregen.len - length(p1))))[pregen.pregenButterfly])', length(pregen.primeArray), 1)

    # DFT
    gpu_ntt(stackedp1, pregen.primeArray, pregen.npruArray, pregen.len, pregen.log2len)

    # Broadcasting power
    broadcast_pow(stackedp1, pregen.primeArray, pow)

    # IDFT
    multimodularResultArr = gpu_intt(stackedp1, pregen)

    # here for memories of debugging
    # if pregen.len > 9000
    #     temp = Array(multimodularResultArr)
    #     # temp[:, 257512] .= [11, 11]
    #     multimodularResultArr = CuArray(temp)
    #     println("two coeffs of term [12, 20, 39, 9]: ", Array(multimodularResultArr)[:, 257512])
    #     println("two coeffs of term [15, 20, 39, 6]: ", Array(multimodularResultArr)[:, 257515])
    #     println("primearray: ", pregen.primeArray)

    # end

    # CRT
    result = build_result(multimodularResultArr, pregen.primeArray, pregen.crtType, pregen.resultType)[1:finalLength]

    return result
end

function broadcast_pow(arr::CuArray{}, primearray, pow)
    @assert size(arr, 1) == length(primearray)

    nthreads = min(size(arr, 2), 512)
    nblocks = cld(size(arr, 2), nthreads)

    cu_primearray = CuArray(primearray)

    function broadcast_pow_kernel(arr, cu_primearray, pow)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        for i in axes(arr, 1)
            arr[i, idx] = power_mod(arr[i, idx], pow, cu_primearray[i])
        end

        return
    end

    @cuda(
        threads = nthreads,
        blocks = nblocks,
        broadcast_pow_kernel(arr, cu_primearray, pow)
    )

    return
end

function gpu_intt(vec::CuArray{Int, 2}, pregen::GPUPowPregen)
    arg1 = vec[:, pregen.pregenButterfly]
    result = gpu_ntt(arg1, pregen.primeArray, pregen.npruInverseArray, pregen.len, pregen.log2len)

    for i in 1:length(pregen.primeArray)
        # result[i, :] .= map(x -> faster_mod(x * mod_inverse(len, primearray[i]), primearray[i]), result[i, :])
        result[i, :] .*= pregen.lenInverseArray[i]
        result[i, :] .%= pregen.primeArray[i]
    end

    return result
end

function gpu_ntt(vec::CuArray{Int}, primeArray, npruArray, len, log2len)
    cu_primearray = CuArray(primeArray)
    nthreads = min(512, len ÷ 2)
    nblocks = cld(len ÷ 2, nthreads)

    for i in 1:log2len
        m = 1 << i
        m2 = m >> 1
        magic = 1 << (log2len - i)
        theta_m = CuArray(powermod.(npruArray, Int(len / m), primeArray))
        @assert all(x -> x >= 0, theta_m)
        @cuda(
            threads = nthreads,
            blocks = nblocks,
            gpu_ntt_kernel!(vec, cu_primearray, theta_m, magic, m2)
        )
    end

    return vec
end

function gpu_ntt_kernel!(vec, primearray, theta_m, magic, m2)
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

function build_result(multimodularResultArr::CuArray{Int, 2}, primearray::Vector{Int}, crtType::DataType, resultType::DataType)
    @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = crtType.(multimodularResultArr[1, :])

    nthreads = min(512, length(result))
    nblocks = cld(length(result), nthreads)

    currmod = crtType(primearray[1])

    for i in 2:size(multimodularResultArr, 1)
        CUDA.@sync @cuda(
            threads = nthreads,
            blocks = nblocks,
            build_result_kernel(result, multimodularResultArr, i, currmod, primearray[i])
        )
        currmod *= primearray[i]
    end

    return resultType.(result)
end

function build_result_kernel(result, multimodularResultArr, row, currmod, newprime)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    result[idx] = chinese_remainder_two(result[idx], currmod, multimodularResultArr[row, idx], newprime)

    return
end