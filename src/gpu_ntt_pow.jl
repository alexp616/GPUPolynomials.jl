include("ntt_utils.jl")

function generate_butterfly_permutations(n::Int)::CuArray{Int, 1}
    @assert ispow2(n) "n must be a power of 2"
    perm = parallelBitReverseCopy(CuArray([i for i in 1:n]))
    return perm
end


function bit_reverse(x, log2n)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

function extended_gcd_iterative(a, b)
    x0, x1 = 1, 0
    y0, y1 = 0, 1
    while b != 0
        q = a ÷ b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end
    return (a, x0, y0)
end

function chinese_remainder_two(a, n, b, m)
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

function GPUPow(p1::CuArray{Int, 1}, pow; primearray::Array{Int, 1}, npruarray::Array{Int, 1}, len = -1, pregenButterfly = nothing)
    if len == -1
        len = nextpow(2, (length(p1) - 1) * pow + 1)
    end
    log2length = Int(log2(len));
    finalLength = (length(p1) - 1) * pow + 1

    if pregenButterfly === nothing
        pregenButterfly = generate_butterfly_permutations(len)
    end

    @assert length(pregenButterfly) == len "pregenerated butterfly doesn't have same length as input"

    # Padding, stacking for multiple prime modulus, and butterflying indices
    stackedp1 = repeat((vcat(p1, zeros(Int, len - length(p1)))[pregenButterfly])', length(primearray), 1)

    # DFT
    GPUDFT(stackedp1, primearray, npruarray, len, log2length, butterflied = true)

    # Broadcasting power
    broadcast_pow(stackedp1, primearray, pow)

    # IDFT
    multimodularResultArr = GPUIDFT(stackedp1, primearray, npruarray, len, log2length, pregenButterfly)

    # CRT
    result = build_result(multimodularResultArr, primearray)[1:finalLength]

    return result
end

function broadcast_pow(arr, primearray, pow)
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

function GPUIDFT(vec::CuArray{Int, 2}, primearray::Vector{Int}, npruarray::Vector{Int}, len::Int, log2length::Int, pregenButterfly = nothing)
    if pregenButterfly === nothing
        pregenButterfly = generate_butterfly_permutations(length)
    end

    arg1 = vec[:, pregenButterfly]
    result = GPUDFT(arg1, primearray, mod_inverse.(npruarray, primearray), len, log2length, butterflied = true)

    for i in 1:length(primearray)
        result[i, :] .*= mod_inverse(len, primearray[i])
        result[i, :] .%= primearray[i]
    end

    return result
end

function GPUDFT(vec, primearray::Array{Int, 1}, npruarray, len, log2length; butterflied = false)
    if !butterflied
        perm = generate_butterfly_permutations(len)
        vec = vec[:, perm]
    end

    cu_primearray = CuArray(primearray)

    nthreads = min(512, len ÷ 2)
    nblocks = cld(len ÷ 2, nthreads)

    for i in 1:log2length
        m = 1 << i
        m2 = m >> 1
        magic = 1 << (log2length - i)
        theta_m = CuArray(powermod.(npruarray, Int(len / m), primearray))
        @cuda(
            threads = nthreads,
            blocks = nblocks,
            GPUDFTKernel!(vec, cu_primearray, theta_m, magic, m2)
        )
    end

    return vec
end

function GPUDFTKernel!(vec, primearray, theta_m, magic, m2)
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

function build_result(multimodularResultArr::CuArray{Int, 2}, primearray::Array{Int, 1})
    @assert size(multimodularResultArr, 1) == length(primearray) "number of rows of input array and number of primes must be equal"

    result = multimodularResultArr[1, :]

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