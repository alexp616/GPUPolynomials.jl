using CUDA
using Test
using Primes
using BenchmarkTools
using Dates

@inline function faster_mod(x, m)
    return Int(x - div(x, m) * m)
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

    c = (a * m * y + b * n * x) % (n * m)
    return c < 0 ? c + n * m : c
end

"""
    power_mod(n, p, m)


"""
function power_mod(n, p, m)
    result = 1
    p = faster_mod(p, m - 1)
    base = faster_mod(n, m)

    while p > 0
        if p % 2 == 1
            result = (result * base) % m
        end
        base = (base * base) % m
        p = div(p, 2)
    end

    return result
end

"""
    bit_reverse(x, log2n)

Compute the bit-reversal of x
"""
function bit_reverse(x, log2n)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

"""
    mod_inverse(n, p)

Return n^-1 mod p.
"""
function mod_inverse(n::Int, p::Integer)
    # Ensure n is positive
    @assert isprime(p)
    n = mod(n, p)
    
    # Extended Euclidean Algorithm
    t, new_t = 0, 1
    r, new_r = p, n

    while new_r != 0
        quotient = div(r, new_r)
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    end

    if t < 0
        t = t + p
    end

    return t
end


function nth_principal_root_of_unity(n::Int, p::Int)
    # Ensure n divides p-1
    if (p - 1) % n != 0
        error("n must divide p-1")
    end

    # Compute the multiplicative order of p-1 mod n
    order = (p - 1) ÷ n
    
    # Find a generator of the multiplicative group of integers modulo p
    function is_primitive_root(g, p, order)
        for i in 1:(n-1)
            if powermod(g, i * order, p) == 1
                return false
            end
        end
        return true
    end
    
    g = 2
    while !is_primitive_root(g, p, order)
        g += 1
    end

    # Compute the n-th principal root of unity
    root_of_unity = powermod(g, order, p)
    @assert root_of_unity > 0 "root_of_unity overflowed"
    return root_of_unity
end

function parallelBitReverseCopy(p)
    @assert ispow2(length(p)) "p must be an array with length of a power of 2"
    len = length(p)
    result = CUDA.zeros(eltype(p), len)
    nthreads = min(512, len ÷ 2)
    nblocks = cld(len ÷ 2, nthreads)
    log2n = Int(log2(len))

    function kernel(p, dest, len, log2n)
        idx1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        idx2 = idx1 + Int(len / 2)
    
        rev1 = bit_reverse(idx1, log2n)
        rev2 = bit_reverse(idx2, log2n)
    
        dest[idx1 + 1] = p[rev1 + 1]
        dest[idx2 + 1] = p[rev2 + 1]
        return nothing
    end

    @cuda(
        threads = nthreads,
        blocks = nblocks,
        kernel(p, result, len, log2n)
    )
    
    return result
end

function generate_butterfly_permutations(n::Int)::CuArray{Int, 1}
    @assert ispow2(n) "n must be a power of 2"
    perm = parallelBitReverseCopy(CuArray([i for i in 1:n]))
    return perm
end

# chinese_remainder_two(70173529, 330301441, 67066783, 311427073)

# the given primearray is three primes that satisfy the max ntt length for the smallest case (80*81*81), and that multiply to over 2^64
function GPUPow(p1::CuArray{Int, 1}, pow; primearray::Array{Int, 1} = Int64[330301441, 311427073], npruarray = [206661007, 241824906], pregen_butterfly = nothing)
    len = nextpow(2, (length(p1) - 1) * pow + 1)
    log2length = Int(log2(len));
    finalLength = (length(p1) - 1) * pow + 1

    if pregen_butterfly === nothing
        pregen_butterfly = generate_butterfly_permutations(len)
    end

    @assert length(pregen_butterfly) == len "pregenerated butterfly doesn't have same length as cuarrays"

    ##########################################
    startTime = now()
    stackedp1 = repeat((vcat(p1, zeros(Int, len - length(p1)))[pregen_butterfly])', length(primearray), 1)
    println("time to stack p1: $(now() - startTime)")
    ##########################################
    startTime = now()
    GPUDFT(stackedp1, primearray, npruarray, len, log2length, butterflied = true)
    println("time to DFT: $(now() - startTime)")
    ##########################################
    startTime = now()
    broadcast_pow(stackedp1, primearray, pow)
    println("time to broadcast power: $(now() - startTime)")
    ##########################################
    startTime = now()
    multimodularResultArr = GPUIDFT(stackedp1, primearray, npruarray, len, log2length, pregen_butterfly)
    println("time to IDFT: $(now() - startTime)")
    ##########################################
    startTime = now()
    result = build_result(multimodularResultArr, primearray)[1:finalLength]
    println("time to CRT result: $(now() - startTime)")
    ##########################################
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

"""
    GPUIDFT(y, n, log2n)

Return the inverse DFT of vector p as a vector.
"""
function GPUIDFT(vec::CuArray{Int, 2}, primearray::Vector{Int}, npruarray::Vector{Int}, len::Int, log2length::Int, pregen_butterfly = nothing)
    if pregen_butterfly === nothing
        pregen_butterfly = generate_butterfly_permutations(length)
    end

    arg1 = vec[:, pregen_butterfly]
    result = GPUDFT(arg1, primearray, mod_inverse.(npruarray, primearray), len, log2length, butterflied = true)

    for i in 1:length(primearray)
        result[i, :] .*= mod_inverse(len, primearray[i])
        result[i, :] .%= primearray[i]
    end

    return result
end


"""
    GPUDFT(vec, primearray, length, log2length, butterflied = false, inverted = 1)

Return the DFT of vector p as a vector. Output can be complex.

Does not work when log2(length(p)) is not an integer
"""
function GPUDFT(vec, primearray, npruarray, len, log2length; butterflied = false)
    if !butterflied
        println("vector was butterflied in dft function instead of multiply")
        perm = generate_butterfly_permutations(len)
        vec = vec[:, perm]
    end

    cu_primearray = CuArray(primearray)

    nthreads = min(512, len ÷ 2)
    nblocks = cld(len ÷ 2, nthreads)
    
    ###############################################
    startTime = now()
    for i in 1:log2length
        startTime2 = now()
        m = 1 << i
        m2 = m >> 1
        magic = 1 << (log2length - i)
        theta_m = CuArray(powermod.(npruarray, Int(len / m), primearray))
        CUDA.@sync @cuda(
            threads = nthreads,
            blocks = nblocks,
            GPUDFTKernel!(vec, cu_primearray, theta_m, magic, m2)
        )
        println("\t\t$i th iteration of DFT: $(now() - startTime2)")
    end
    println("\ttime to DFT: $(now() - startTime)")
    ################################################

    # println()
    return vec
end

"""
    GPUDFTKernel!(vec, primearray, theta_m, magic, m2)

Kernel function of GPUDFT()
"""
function GPUDFTKernel!(vec, primearray, theta_m, magic, m2)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = Int(2 * m2 * (idx % magic) + floor(idx/magic))

    for p in eachindex(primearray)
        theta = power_mod(theta_m[p], idx ÷ magic, primearray[p])
        # @cuprintln("GPU theta: $theta")
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
        CUDA.@sync @cuda(
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

function npruarray_generator(primearray::Array, n)
    return map(p -> nth_principal_root_of_unity(n, p), primearray)
end

function change_encoding(num::Int, e1::Int, e2::Int, numValues::Int)
    result = 0
    for i in 1:numValues
        num, r = divrem(num, e1)
        result += r * e2 ^ (i - 1)
    end

    return result
end

# GPUPow(CuArray([1, 1]), 5)

# npruarray_generator([330301441, 311427073], 2^20)
butterfly = generate_butterfly_permutations(1048576)
# CUDA.@time result = GPUMultiply(polynomial1, polynomial2, primearray = [17, 233], npruarray = npruarray_generator([17, 233], 8))

polynomial1 = CuArray(ones(Int64, 16 * 81 ^ 2))

println("Time to raise 4 variable, 16-homogeneous polynomial to the 5th power: ")
result = GPUPow(polynomial1, 5, primearray = [330301441, 311427073], npruarray = [206661007, 241824906], pregen_butterfly = butterfly)