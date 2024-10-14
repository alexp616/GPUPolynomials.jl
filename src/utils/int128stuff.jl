import Base: mod, ÷, %

function Base.mod(x::UInt128, m::UInt128)
    if x == 0
        return UInt128(0)
    end

    remainder = UInt128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((x >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
        end
    end

    return remainder
end

function divi(x::UInt128, m::UInt128)
    if x == 0
        return UInt128(0)
    end

    quotient = UInt128(0)
    remainder = UInt128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((x >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
            quotient |= (UInt128(1) << (127 - i))
        end
    end

    return quotient
end

function Base.mod(x::Int128, m::Int128)
    if x == 0
        return Int128(0)
    end

    sign = 1
    if (x < 0) != (m < 0)
        sign = -1
    end

    n = abs(x)
    m = abs(m)

    remainder = Int128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((x >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
        end
    end
    result = remainder * sign
    return result < 0 ? result + m : result
end

function divi(x::Int128, m::Int128)
    if x == 0
        return Int128(0)
    end

    sign = 1
    if (x < 0) != (m < 0)
        sign = -1
    end

    x = abs(x)
    m = abs(m)

    quotient = Int128(0)
    remainder = Int128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((x >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
            quotient |= (Int128(1) << (127 - i))
        end
    end

    return quotient * sign
end

Base.:%(x::UInt128, m::UInt128) = mod(x, m)
# Base.:÷(x::UInt128, m::UInt128) = div(x, m)
Base.:%(x::Int128, m::Int128) = mod(x, m)
# Base.:÷(x::Int128, m::Int128) = div(x, m)
Base.:÷(x::UInt128, m::UInt128) = divi(x, m)
Base.:÷(x::Int128, m::Int128) = divi(x, m)