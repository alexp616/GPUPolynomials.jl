abstract type CuPolyRingElem end

function get_coeffs(poly::PolyRingElem, T::DataType = Nothing)
    if T === Nothing
        maxCoeff = BigInt(maximum(coefficients(poly)))

        if poly isa ZZPolyRingElem
            T = get_int_type(max(64, Base._nextpow2(Int(ceil(log2(maxCoeff))) + 1)))
        else
            throw("chilling")
        end
        return T.(coefficients(poly))
    else
        coeffsPtr = Base.unsafe_convert(Ptr{T}, poly.coeffs)
        return unsafe_wrap(Vector{T}, coeffsPtr, poly.length)
    end
end

function Base.length(poly::CuPolyRingElem)
    return poly.length
end

struct MulPlan{T<:Integer} <: OperationPlan
    len::Int
    nttMulPlans::Vector{NTTMulPlan{T}}
    crtPlan::CuArray
end

Base.eltype(::Type{MulPlan{T}}) where T = T

struct PowPlan{T<:Unsigned} <: OperationPlan
    len::Int
    nttPowPlans::Vector{NTTPowPlan{T}}
    crtPlan::CuArray
end

Base.eltype(::Type{PowPlan{T}}) where T = T