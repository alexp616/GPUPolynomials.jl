using CUDA

import Base: divrem, mod

@inline function Base.mod(x::Signed, m::Signed)
    q, r = divrem(x, m)
    return r < 0 ? r + m : r
end

function crt(vec, pregen)
    x = vec[1]

    for i in axes(pregen, 2)
        x = mod(x * pregen[2, i] + vec[i + 1] * pregen[1, i], pregen[3, i])
    end

    return x
end

function get_int_type(n)
    return eval(Symbol("Int", n))
end

function get_uint_type(n)
    return eval(Symbol("UInt", n))
end

# These two aren't the most efficient, but its just a few ms in
# pregeneration time who cares
function get_result_type(primeArray::Vector{<:Unsigned})
    primeArray = BigInt.(primeArray)
    totalprod = prod(primeArray)
    sizeNeeded = ceil(Int, log2(totalprod + 1))
    resultType = get_uint_type(Base._nextpow2(sizeNeeded))

    return resultType
end

function pregen_crt(primeArray::Vector{<:Integer})
    primeArray = BigInt.(primeArray)
    totalprod = prod(primeArray)
    biggestnumneeded = totalprod^2
    sizeNeeded = ceil(Int, log2(biggestnumneeded + 1))
    crtType = get_uint_type(Base._nextpow2(sizeNeeded))

    result = zeros(BigInt, 3, length(primeArray) - 1)

    currmod = primeArray[1]
    for i in 2:length(primeArray)
        m1, m2 = extended_gcd_iterative(currmod, primeArray[i])
        result[1, i - 1] = m1 * currmod
        currmod *= primeArray[i]
        result[2, i - 1] = m2 * primeArray[i] + currmod
        result[3, i - 1] = currmod
    end

    @assert all([i > 0 for i in result])
    @assert all(i -> log2(i) < sizeNeeded, result)
    return crtType.(result)
end

function Base.divrem(n::Int128, m::Int128)
    if n == 0
        return Int128(0)
    end

    sign = 1
    if (n < 0) != (m < 0)
        sign = -1
    end

    n = abs(n)
    m = abs(m)

    quotient = Int128(0)
    remainder = Int128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((n >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
            quotient |= (Int128(1) << (127 - i))
        end
    end

    return quotient * sign, remainder
end

function Base.divrem(n::UInt128, m::UInt128)
    if n == 0
        return UInt128(0)
    end

    quotient = UInt128(0)
    remainder = UInt128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((n >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
            quotient |= (UInt128(1) << (127 - i))
        end
    end

    return quotient, remainder
end

function extended_gcd_iterative(a::T, b::T) where T<:Signed
    x0, x1 = T(1), T(0)
    y0, y1 = T(0), T(1)
    while b != 0
        q, r = divrem(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end
    @assert a == 1 "$a and $b aren't coprime"
    return x0, y0
end

function chinese_remainder_two(a::T, n::T, b::Integer, m::Integer) where T<:Integer
    b = T(b)
    m = T(m)

    n0, m0 = n, m
    x0, x1 = T(1), T(0)
    y0, y1 = T(0), T(1)
    while m != 0
        q = div(n, m)
        n, m = m, mod(n, m)
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end

    return mod(a * m0 * y0 + b * n0 * x0, T(n0 * m0))
end

function power_mod(n::Integer, p::Integer, m::Integer)
    result = eltype(n)(1)
    p = mod(p, m - 1)
    base = mod(n, m)

    while p > 0
        if p & 1 == 1
            result = mod((result * base), m)
        end
        base = mod(base * base, m)
        p = p >> 1
    end

    return result
end

function montgomery_power_mod(n::Integer, p::Integer, m::Integer)
    
end

function mod_inverse(n::Integer, p::Integer)
    n = BigInt(n)
    p = BigInt(p)
    n = mod(n, p)

    t, new_t = 0, 1
    r, new_r = p, n

    while new_r != 0
        quotient = r ÷ new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    end

    return t < 0 ? typeof(n)(t + p) : typeof(n)(t)
end


function nth_principal_root_of_unity(n::Integer, p::Unsigned)
    @assert mod(p - 1, n) == 0 "n must divide p-1"

    order = (p - 1) ÷ n

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

    root_of_unity = powermod(g, order, p)
    return typeof(p)(root_of_unity)
end

function npruarray_generator(primearray::Array{<:Unsigned}, n)
    return map(p -> nth_principal_root_of_unity(n, p), primearray)
end

function parallel_bit_reverse_copy(p)
    @assert ispow2(length(p)) "p must be an array with length of a power of 2"
    len = length(p)
    result = CUDA.zeros(eltype(p), len)
    log2n = Int(log2(len))

    function kern(p, dest, len, log2n)
        idx1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        idx2 = idx1 + Int(len / 2)
    
        rev1 = bit_reverse(idx1, log2n)
        rev2 = bit_reverse(idx2, log2n)
    
        dest[idx1 + 1] = p[rev1 + 1]
        dest[idx2 + 1] = p[rev2 + 1]
        return nothing
    end

    kernel = @cuda launch = false kern(p, result, len, log2n)
    config = launch_configuration(kernel.fun)
    threads = min(len ÷ 2, prevpow(2, config.threads))
    blocks = cld(len ÷ 2, threads)

    kernel(p, result, len, log2n; threads = threads, blocks = blocks)
    
    return result
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

function generate_butterfly_permutations(n::Int)::CuVector
    @assert ispow2(n) "n must be a power of 2"
    perm = parallel_bit_reverse_copy(CuArray([i for i in 1:n]))
    return perm
end

# MONTGOMERY REDUCTION

struct MontgomeryReducer{T<:Unsigned}
    modulus::T
    rbits::Int
    r::T
    mask::T
    rinv::T
    k::T
    convertedone::T

    function MontgomeryReducer(n::Unsigned)
        rbits = (ndigits(n, base = 2) ÷ 8 + 1) * 8
        T = get_uint_type(Base._nextpow2(2 * rbits))
        modulus = T(n)
        r = T(1) << rbits
        mask = r - 1
        @assert r > n && gcd(r, n) == 1
        
        rinv = T((mod_inverse(r, n)))
        k = (r * rinv - 1) ÷ n
        convertedone = T(mod(r, n))

        return new{T}(modulus, rbits, r, mask, rinv, k, convertedone)
    end
end

function convert_in(mr::MontgomeryReducer, x::Unsigned)
    return mod(typeof(mr.modulus)(x) << mr.rbits, mr.modulus) 
end

function convert_out(mr::MontgomeryReducer, x::Unsigned)
    return mod(mr.rinv * x, mr.modulus)
end

function mul(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
    m = mr.modulus
    product = x * y
    temp = ((product & mr.mask) * mr.k) & mr.mask
    reduced = (product + temp * m) >> mr.rbits
    result = reduced < m ? reduced : reduced - m

    return result
end

function add(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
    return x + y < mr.modulus ? x + y : x + y - mr.modulus
end

function sub(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
    return y > x ? mr.modulus - (y - x) : x - y
end

function pow(mr::MontgomeryReducer, x::Unsigned, p::Unsigned)
    z = mr.convertedone

    while p != 0
        if p & 1 != 0
            z = mul(mr, z, x)
        end
        x = mul(mr, x, x)
        p >>= 1
    end
    
    return z
end

# Debug version
# struct MontgomeryReducer{T<:Unsigned}
#     modulus::T
#     rbits::Int
#     r::T
#     mask::T
#     rinv::T
#     k::T
#     convertedone::T

#     function MontgomeryReducer(n::Unsigned)
#         rbits = (ndigits(n, base = 2) ÷ 8 + 1) * 8
#         T = get_uint_type(Base._nextpow2(2 * rbits))
#         modulus = T(n)
#         r = T(1) << rbits
#         mask = r - 1
#         @assert r > n && gcd(r, n) == 1
        
#         rinv = T((mod_inverse(r, n)))
#         k = (r * rinv - 1) ÷ n
#         convertedone = T(mod(r, n))

#         return new{T}(modulus, rbits, r, mask, rinv, k, convertedone)
#     end
# end

# function convert_in(mr::MontgomeryReducer, x::Unsigned)
#     return mod(typeof(mr.modulus)(x) << mr.rbits, mr.modulus) 
# end

# function convert_out(mr::MontgomeryReducer, x::Unsigned)
#     return mod(mr.rinv * x, mr.modulus)
# end

# function mul(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
#     m = mr.modulus

#     # product = x * y
#     product, f = Base.mul_with_overflow(x, y)
#     @assert f == false "x: $x, y: $y"

#     # temp = ((product & mr.mask) * mr.k) & mr.mask
#     temp, f = Base.mul_with_overflow((product & mr.mask), mr.k)
#     temp &= mr.mask
#     @assert f == false "product & mr.mask: $(product & mr.mask), mr.k: $(mr.k)"

#     # reduced = (product + temp * m) >> mr.rbits
#     asdf, f = Base.mul_with_overflow(temp, m)
#     @assert f == false "temp: $temp, m: $m"
#     reduced, f = Base.add_with_overflow(product, asdf)
#     reduced >>= mr.rbits
#     # println("log2(product): $(log2(product)), log2(asdf): $(log2(asdf))")
#     @assert f == false "product: $product, asdf: $asdf"

#     result = reduced < m ? reduced : reduced - m
#     return result
# end

# function add(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
#     @assert x <= mr.modulus && y <= mr.modulus
#     sum, f = Base.add_with_overflow(x, y)
#     @assert f == false "Overflow in addition: x: $x, y: $y"

#     # If the sum is greater than or equal to the modulus, reduce it
#     result = sum < mr.modulus ? sum : sum - mr.modulus
#     return result
# end

# function sub(mr::MontgomeryReducer, x::Unsigned, y::Unsigned)
#     @assert x <= mr.modulus && y <= mr.modulus
#     return y > x ? mr.modulus - (y - x) : x - y
# end

# function pow(mr::MontgomeryReducer, x::Unsigned, p::Unsigned)
#     # p = mod(p, mr.modulus - 1)
#     z = mr.convertedone
#     # println(eltype(mr.modulus))
#     while p != 0
#         if p & 1 != 0
#             z = mul(mr, z, x)
#         end
#         x = mul(mr, x, x)
#         p >>= 1
#     end

#     return z
# end