include("ntt_utils.jl")

struct MontReducer{T<:Unsigned}
    modulus::T
    rbits::Int
    r::T
    mask::T
    rinv::T
    k::T
    convertedone::T

    function MontReducer(n::Integer)
        n = unsigned(n)
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

    function MontReducer(n::Integer, T<:Unsigned)
        n = T(n)
        rbits = (ndigits(n, base = 2) ÷ 8 + 1) * 8
        modulus = T(n)
        r = T(1) << rbits
        mask = r - 1
        @assert r > n && gcd(r, n) == 1

        rinv = T((mod_inverse(r, n)))
        k = (r * inv - 1) ÷ n
        convertedone = T(mod(r, n))

        return new{T}(modulus, rbits, r, mask, rinv, k, convertedone)
    end
end

function convert_in(mr::MontReducer, x::Unsigned)
    return mod(typeof(mr.modulus)(x) << mr.rbits, mr.modulus) 
end

function convert_out(mr::MontReducer, x::Unsigned)
    return mod(mr.rinv * x, mr.modulus)
end

function mul(mr::MontReducer, x::Unsigned, y::Unsigned)
    m = mr.modulus
    product = x * y
    temp = ((product & mr.mask) * mr.k) & mr.mask
    reduced = (product + temp * m) >> mr.rbits
    result = reduced < m ? reduced : reduced - m

    return result
end

function add(mr::MontReducer, x::Unsigned, y::Unsigned)
    return x + y < mr.modulus ? x + y : x + y - mr.modulus
end

function sub(mr::MontReducer, x::Unsigned, y::Unsigned)
    return y > x ? mr.modulus - (y - x) : x - y
end

function pow(mr::MontReducer, x::Unsigned, p::Unsigned)
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
# struct MontReducer{T<:Unsigned}
#     modulus::T
#     rbits::Int
#     r::T
#     mask::T
#     rinv::T
#     k::T
#     convertedone::T

#     function MontReducer(n::Unsigned)
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

# function convert_in(mr::MontReducer, x::Unsigned)
#     return mod(typeof(mr.modulus)(x) << mr.rbits, mr.modulus) 
# end

# function convert_out(mr::MontReducer, x::Unsigned)
#     return mod(mr.rinv * x, mr.modulus)
# end

# function mul(mr::MontReducer, x::Unsigned, y::Unsigned)
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

# function add(mr::MontReducer, x::Unsigned, y::Unsigned)
#     @assert x <= mr.modulus && y <= mr.modulus
#     sum, f = Base.add_with_overflow(x, y)
#     @assert f == false "Overflow in addition: x: $x, y: $y"

#     # If the sum is greater than or equal to the modulus, reduce it
#     result = sum < mr.modulus ? sum : sum - mr.modulus
#     return result
# end

# function sub(mr::MontReducer, x::Unsigned, y::Unsigned)
#     @assert x <= mr.modulus && y <= mr.modulus
#     return y > x ? mr.modulus - (y - x) : x - y
# end

# function pow(mr::MontReducer, x::Unsigned, p::Unsigned)
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