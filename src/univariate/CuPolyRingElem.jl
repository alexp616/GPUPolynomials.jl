abstract type CuPolyRingElem end

function get_coeffs(poly::PolyRingElem, T::DataType = Nothing)
    if T === Nothing
        maxCoeff = BigInt(maximum(coefficients(poly)))

        if poly isa CuZZPolyRingElem
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