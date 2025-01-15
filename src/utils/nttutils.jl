include("getinttype.jl")
include("modoperations.jl")
include("modsqrt.jl")

function intlog2(x::Int64)
    return 64 - leading_zeros(x - 1)
end

function intlog2(x::Int32)::Int32
    return Int32(32) - leading_zeros(x - Int32(1))
end

function is_primitive_root(npru::T, p::T, order::Integer) where T<:Integer
    temp = npru
    for i in 1:order - 1
        if temp == 1
            return false
        end

        temp = mul_mod(temp, npru, p)
    end

    return temp == 1
end

"""
    primitive_nth_root_of_unity(n::Integer, p::Integer)

Return a primitive n-th root of unity of the field 𝔽ₚ
"""
function primitive_nth_root_of_unity(n::Integer, p::Integer)
    @assert ispow2(n)
    if (p - 1) % n != 0
        throw("n must divide p - 1")
    end

    g = p - typeof(p)(1)

    a = intlog2(n)

    while a > 1
        a -= 1
        original = g
        g = modsqrt(g, p)
        @assert powermod(g, 2, p) == original
    end

    @assert is_primitive_root(g, p, n)
    return g
end

"""
    generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer

Returns array containing powers 0 -> n-1 of npru mod p. Accessed as:
arr[i] = npru ^ (i - 1)
"""
function generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer
    @assert is_primitive_root(npru, p, n)

    result = zeros(T, n)
    curr = T(1)
    for i in eachindex(result)
        result[i] = curr
        curr = mul_mod(curr, npru, p)
    end

    return result
end

function find_ntt_primes(len::Int, T = UInt32, num = 10)
    prime_list = []
    k = fld(typemax(T), len)
    while length(prime_list) < num
        candidate = k * len + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k -= 1
    end

    return prime_list
end

function bit_reverse(x::T, log2n::T)::T where T<:Integer
    temp = zero(T)
    for _ in one(T):log2n
        temp <<= one(T)
        temp |= (x & one(T))
        x >>= one(T)
    end
    return temp
end

function digit_reverse(x::Integer, base::Integer, logn::Integer)
    temp = 0

    for _ in 1:logn
        x, b = divrem(x, base)
        temp = temp * base + b
    end
    
    return temp
end

function get_transposed_index(idx::T, rows::T, cols::T) where T<:Integer
    originalRow = idx % rows
    originalCol = idx ÷ rows

    result = originalCol + originalRow * cols

    return result
end

function final_transpose(idx::Integer, bitlength::Integer, numsPerBlock::Integer, lastFFTLen::Integer)
    firstswaplength = intlog2(lastFFTLen)
    unchangedbitslen = intlog2(numsPerBlock ÷ lastFFTLen)
    middlebitslen = bitlength - 2 * firstswaplength - unchangedbitslen

    lastBits = idx & ((1 << firstswaplength) - 1)
    idx >>= firstswaplength
    unchangedbits = idx & ((1 << unchangedbitslen) - 1)
    idx >>= unchangedbitslen
    middlebits = idx & ((1 << middlebitslen) - 1)
    idx >>= middlebitslen
    firstBits = idx & ((1 << firstswaplength) - 1)
    
    middlebits = digit_reverse(middlebits, numsPerBlock, middlebitslen ÷ intlog2(numsPerBlock))
    offset = firstswaplength

    result = firstBits
    result |= unchangedbits << offset
    offset += unchangedbitslen
    result |= middlebits << offset
    offset += middlebitslen
    result |= lastBits << offset

    return typeof(idx)(result)
end