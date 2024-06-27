include("ntt_utils.jl")

struct GPUPowPregen
    primeArray::Vector{Int}
    npruArray::Vector{Int}
    npruInverseArray::Vector{Int}
    len::Int
    log2len::Int
    lenInverseArray::Vector{Int}
    pregenButterfly::CuVector{Int}
    resultType::DataType
end

function pregen_gpu_pow(;primeArray::Vector{Int}, len::Int, resultType::DataType)
    @assert ispow2(len) "len must be a power 2"

    npruArray = npruarray_generator(primeArray, len)
    npruInverseArray = mod_inverse.(npruArray, primeArray)
    log2len = Int(log2(len))
    lenInverseArray = map(p -> mod_inverse(len, p), primeArray)
    pregenButterfly = generate_butterfly_permutations(len)

    return GPUPowPregen(primeArray, npruArray, npruInverseArray, len, log2len, lenInverseArray, pregenButterfly, resultType)
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

function extended_gcd_iterative(a::T, b::T) where T<:Integer
    x0, x1 = T(1), T(0)
    y0, y1 = T(0), T(1)
    while b != 0
        q = a ÷ b
        a, b = b, faster_mod(a, b)
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end
    return (a, x0, y0)
end

function chinese_remainder_two(a::T, n, b::Integer, m) where T<:Integer
    if a < 0
        a += n
    end
    if b < 0
        b += m
    end
    d, x, y = extended_gcd_iterative(n, m)
    if d != 1
        throw(ArgumentError("n and m must be coprime"))
    end

    c = faster_mod((a * m * y + b * n * x), (n * m))
    return c < 0 ? c + n * m : c
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

    # CRT
    result = build_result(multimodularResultArr, pregen.primeArray, pregen.resultType)[1:finalLength]

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

function build_result(multimodularResultArr::CuArray{Int, 2}, primearray::Vector{Int}, resultType::DataType)
    @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = resultType.(multimodularResultArr[1, :])

    nthreads = min(512, length(result))
    nblocks = cld(length(result), nthreads)

    currmod = primearray[1]

    for i in 2:size(multimodularResultArr, 1)
        @cuda(
            threads = nthreads,
            blocks = nblocks,
            build_result_kernel(result, multimodularResultArr, i, currmod, primearray[i])
        )
        currmod *= primearray[i]
    end

    return result
end

function build_result_kernel(result, multimodularResultArr, row, currmod, newprime)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    result[idx] = chinese_remainder_two(result[idx], currmod, multimodularResultArr[row, idx], newprime)

    return
end

# arr = CUDA.ones(Int, 4 * 17^2 + 1)
# pregenButterfly = generate_butterfly_permutations(8192)
# primeArray = [13631489, 23068673]
# gpuNpruArray = npruarray_generator(primeArray, 8192)

# gpu_pow(arr, 4; primearray = primeArray, npruarray = gpuNpruArray, len = 8192, pregenButterfly = pregenButterfly)
# @benchmark gpu_pow(arr, 4; primearray = primeArray, npruarray = gpuNpruArray, len = 8192, pregenButterfly = pregenButterfly)