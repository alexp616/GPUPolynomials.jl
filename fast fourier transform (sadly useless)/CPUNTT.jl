using Primes
using Dates
using Test
using CUDA
using BenchmarkTools

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

arr = find_ntt_primes(1048576)
println(arr[1:100])
# sus = Int64[2, 3, 4]

"""
    generate_inverses_mod_p(p)

Return array of inverses mod p.
To get inverse of n mod p, simply do arr[n]
"""
function generate_inverses_mod_p(p::Int)
    @assert isprime(p) "$p is not prime"
    result = zeros(Int, p - 1)
    for i in eachindex(result)
        result[i] = mod_inverse(i, p)
    end
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
    return root_of_unity
end

function cpu_ntt(vec, prime, length, log2length, inverted = 1)
    result = zeros(Int, length)

    npru = nth_principal_root_of_unity(length, prime)

    if inverted == -1
        npru = mod_inverse(npru, prime)
    end

    println("cpu nth root of unity: $npru")

    for i in 0:length-1
        rev = bit_reverse(i, log2length)
        result[i+1] = vec[rev+1]
    end

    for i in 1:log2length
        m = 1 << i
        m2 = m >> 1
        alpha = 1
        alpha_m = npru ^ (length / m)
        for j in 0:m2-1
            for k in j:m:length-1
                @cuprintln("theta: $alpha")
                t = alpha * result[k + m2 + 1]
                u = result[k + 1]

                result[k + 1] = (u + t) % prime
                result[k + m2 + 1] = (u - t) % prime
            end
            alpha *= alpha_m
            alpha %= prime
        end
        println("CPU i: $i, vec: ", result)
    end

    println()
    return result .% prime
end

function cpu_intt(vec, prime, length, log2length)
    return (mod_inverse(length, prime) .* cpu_ntt(vec, prime, length, log2length, -1)) .% prime
end

function cpu_multiply(p1, p2, prime)
    n = Int(2^ceil(log2(length(p1) + length(p2) - 1)))
    log2n = UInt32(log2(n));
    finalLength = length(p1) + length(p2) - 1

    append!(p1, zeros(Int, n - length(p1)))
    append!(p2, zeros(Int, n - length(p2)))

    y1 = cpu_ntt(p1, prime, n, log2n)
    y2 = cpu_ntt(p2, prime, n, log2n)

    println("p1 after CPUDFT: ", y1)
    println("p2 after CPUDFT: ", y2)

    ans = cpu_intt(y1 .* y2, prime, n, log2n)
    return ans[1:finalLength]
end

p1 = [1, 1, 1, 1]
p2 = [1, 1, 1, 1]
p3 = cpu_multiply(p1, p2, 7340033)

println("CPU result: ", p3)