# Need this file because "Int128 isn’t natively supported, and 
# LLVM relies on intrinsics for many operations. We don’t currently 
# have these intrinsics (implemented or linked in from another 
# library), resulting in the error you encountered.
# (https://discourse.julialang.org/t/division-for-int128-not-defined-on-gpu/62797)
# And I don't understand enough about compilers yet to solve that.
function unchecked_mod(x::UInt128, m::Integer)
    m = UInt128(m)
    remainder = UInt128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((x >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
        end
    end

    return remainder
end

function unchecked_mod(x::Int128, m::Integer)
    m = Int128(m)
    sign = 1
    if (x < 0) != (m < 0)
        sign = -1
    end

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

function unchecked_div(x::UInt128, m::UInt128)
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

function unchecked_div(x::Int128, m::Int128)
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