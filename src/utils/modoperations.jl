"""
    unchecked_mod(x::T, m::T) where T<:Integer

Variation of mod that assumes m != 0.
In reality only here because 128 bit mod tries
to compile to a CUDA intrinsic `__modti3` which doesn't
actually exist (to my knowledge).

128 bit mod isn't used that often, so not very high up
on priority list. Using really slow long division iterative
algorithm right now.
"""
@inline function unchecked_mod(x::T, m::Integer) where T<:Integer
    return mod(x, T(m))
end

@inline function unchecked_div(x::T, m::Integer) where T<:Integer
    return div(x, T(m))
end

include("int128.jl")

function sub_mod(x::Unsigned, y::Unsigned, m::Unsigned)
    if y > x
        return (m - y) + x
    else
        return x - y
    end
end

function sub_mod(x::Signed, y::Signed, m::Signed)
    return mod(x - y, m)
end

function add_mod(x::Unsigned, y::Unsigned, m::Unsigned)
    result = x + y
    return result >= m ? result - m : result
end

function add_mod(x::Signed, y::Signed, m::Signed)
    result = x + y
    return result >= m ? result - m : result
end

function mywiden(x)
    throw(MethodError(mywiden, (typeof(x),)))
end

macro generate_widen()
    int_types = [Int8, Int16, Int32, Int64, Int128, Int256, Int512]
    uint_types = [UInt8, UInt16, UInt32, UInt64, UInt128, UInt256, UInt512]

    widen_methods = quote end
    for i in 1:length(int_types) - 1
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(int_types[i])) = $(int_types[i+1])(x)
        ))
        push!(widen_methods.args, :(
            Base.@eval mywiden(x::$(uint_types[i])) = $(uint_types[i+1])(x)
        ))
    end

    return widen_methods
end

@generate_widen()

"""
    mywidemul(x::T, y::T) where T<:Integer

Exists because Base.widen() widens Int128 to BigInt, which 
CUDA doesn't like.
"""
function mywidemul(x::T, y::T) where T<:Integer
    return mywiden(x) * mywiden(y)
end

function mul_mod(x::T, y::T, m::T) where T<:Integer
    return T(unchecked_mod(mywidemul(x, y), m))
end

function mul_mod(x::BigInt, y::BigInt, m::BigInt)
    return (x * y) % m
end

function power_mod(n::T, p::Integer, m::T) where T<:Integer
    result = eltype(n)(1)
    base = unchecked_mod(n, m)

    while p > 0
        if p & 1 == 1
            result = mul_mod(result, base, m)
        end
        base = mul_mod(base, base, m)
        p = p >> 1
    end

    return result
end