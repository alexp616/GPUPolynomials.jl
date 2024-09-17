module UInt256Module

export UInt256

struct UInt256
    hi::UInt128
    lo::UInt128

    function UInt256(hi::UInt128, lo::UInt128)
        return new(hi, lo)
    end

    function UInt256(x::Signed)
        return UInt256(unsigned(x))
    end

    function UInt256(x::BigInt)
        return convert(UInt256, x)
    end

    function UInt256(x::Unsigned)
        return new(0x00, UInt128(x))
    end
end

Base.:+(x::UInt256, y::UInt256) = begin
    lo = x.lo + y.lo
    hi = x.hi + y.hi + (lo < x.lo ? 1 : 0)
    UInt256(hi, lo)
end

Base.:-(x::UInt256, y::UInt256) = begin
    lo = x.lo - y.lo
    hi = x.hi - y.hi - (x.lo < y.lo ? 1 : 0)
    UInt256(hi, lo)
end

# Base.:*(x::UInt256, y::UInt256) = begin
#     x_lo, x_hi = x.lo, x.hi
#     y_lo, y_hi = y.lo, y.hi

#     lo = x_lo * y_lo
#     mid1 = x_hi * y_lo
#     mid2 = x_lo * y_hi
#     hi = x_hi * y_hi + (mid1 >> 128) + (mid2 >> 128)
    
#     mid_lo = (mid1 & 0xffffffffffffffffffffffffffffffff) + (mid2 & 0xffffffffffffffffffffffffffffffff)
#     hi += (mid_lo >> 128)
#     lo += mid_lo & 0xffffffffffffffffffffffffffffffff
#     UInt256(hi, lo)
# end

Base.:*(x::UInt256, y::UInt256) = begin
    res = UInt256(0)
    while y > UInt256(0)
        if (y.lo & 1 != 0)
            res += x
        end

        x <<= 1
        y >>= 1
    end

    return res
end

Base.:/(x::UInt256, y::UInt256) = begin
    q, r = divrem(x, y)
    return q
end

Base.:÷(x::UInt256, y::UInt256) = begin
    q, r = divrem(x, y)
    return q
end

Base.mod(x::UInt256, y::UInt256) = begin
    q, r = divrem(x, y)
    return r
end

function Base.divrem(n::UInt256, m::UInt256)
    if n == 0
        return UInt256(0)
    end

    zero = UInt256(0)
    one = UInt256(1)

    quotient = UInt256(0)
    remainder = UInt256(0)


    for i in 0:255
        remainder = (remainder << 1) | ((n >> (255 - i)) & one)
        if remainder >= m
            remainder -= m
            quotient |= (UInt256(1) << (255 - i))
        end
    end

    return quotient, remainder
end

Base.:(==)(x::UInt256, y::UInt256) = (x.hi == y.hi && x.lo == y.lo)

Base.:(<)(x::UInt256, y::UInt256) = (x.hi == y.hi ? x.lo < y.lo : x.hi < y.hi)
Base.:(<=)(x::UInt256, y::UInt256) = (x == y || x < y)

Base.:&(x::UInt256, y::UInt256) = UInt256(x.hi & y.hi, x.lo & y.lo)
Base.:|(x::UInt256, y::UInt256) = UInt256(x.hi | y.hi, x.lo | y.lo)
Base.:~(x::UInt256) = UInt256(~x.hi, ~x.lo)
Base.:⊻(x::UInt256, y::UInt256) = UInt256(x.hi ⊻ y.hi, x.lo ⊻ y.lo)

Base.:<<(x::UInt256, n::Int) = begin
    if n >= 128
        UInt256(x.lo << (n - 128), 0)
    else
        hi = (x.hi << n) | (x.lo >> (128 - n))
        lo = x.lo << n
        UInt256(hi, lo)
    end
end

Base.:>>(x::UInt256, n::Int) = begin
    if n >= 128
        UInt256(UInt128(0), x.hi >> (n - 128))
    else
        lo = (x.lo >> n) | (x.hi << (128 - n))
        hi = x.hi >> n
        UInt256(hi, lo)
    end
end

Base.convert(::Type{UInt128}, x::UInt256) = x.lo

function Base.convert(::Type{BigInt}, x::UInt256)
    return BigInt(x.hi) << 128 | BigInt(x.lo)
end

function Base.convert(::Type{UInt256}, x::BigInt)
    if x < 0 || x >= BigInt(1) << 256
        throw(ArgumentError("BigInt out of range for UInt256"))
    end
    lo = UInt128(x & ((BigInt(1) << 128) - 1))
    hi = UInt128(x >> 128)
    return UInt256(hi, lo)
end

function Base.BigInt(x::UInt256)
    return convert(BigInt, x)
end

Base.show(io::IO, x::UInt256) = print(io, BigInt(x))

end