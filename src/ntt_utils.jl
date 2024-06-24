using CUDA
using Test
using Primes
using BenchmarkTools
using Dates

"""
    find_ntt_primes(n)

Find primes of form k * n + 1 for 5 seconds
isprime() method is probabilistic, actually test for primality using
another method when using a prime from here
"""
function find_ntt_primes(n)
    start_time = now()
    prime_list = []
    k = 1

    while (now() - start_time) < Second(5)
        candidate = k * n + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k += 1
    end

    return prime_list
end


function npruarray_generator(primearray::Array, n)
    return map(p -> nth_principal_root_of_unity(n, p), primearray)
end


# arr = find_ntt_primes(8192)
# println(arr[1:100])

# Also exists to guarantee positive mod numbers so my chinese remainder theorem
# doesn't get messed up
@inline function faster_mod(x, m)
    r = Int(x - div(x, m) * m)
    return r < 0 ? r + m : r
end


"""
    power_mod(n, p, m)

Return n ^ p mod m. Only gives accurate results when
m is prime, since uses fermat's little theorem
"""
function power_mod(n, p, m)
    result = 1
    p = faster_mod(p, m - 1)
    base = faster_mod(n, m)

    while p > 0
        if p % 2 == 1
            result = faster_mod((result * base), m)
        end
        base = (base * base) % m
        p = div(p, 2)
    end

    return result
end



"""
    mod_inverse(n, p)

Return n^-1 mod p.
"""
function mod_inverse(n::Int, p::Integer)
    @assert isprime(p)
    n = mod(n, p)

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
    @assert (p - 1) % n == 0 "n must divide p-1"

    order = (p - 1) ÷ n

    function is_primitive_root(g, p, order)
        for i in 1:(n-1)
            if power_mod(g, i * order, p) == 1
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
    root_of_unity = power_mod(g, order, p)
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