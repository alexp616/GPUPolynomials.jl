abstract type CuMPolyRingElem end

function get_coeffs(poly::MPolyRingElem)
    maxCoeff = BigInt(maximum(coefficients(poly)))
    
    T = get_int_type(max(64, Int(ceil(log2(maxCoeff))) + 1))
    return T.(coefficients(poly; ordering = monomial_ordering(poly.parent, :lex)))
end

function get_coeffs(poly::MPolyRingElem, T::DataType)
    coeffsPtr = Base.unsafe_convert(Ptr{T}, poly.coeffs)
    return unsafe_wrap(Vector{T}, coeffsPtr, poly.length)
end

function get_exps(poly::MPolyRingElem)
    T = get_uint_type(Base._nextpow2(poly.bits * poly.parent.nvars))
    expsPtr = Base.unsafe_convert(Ptr{T}, poly.exps)
    expsVec = unsafe_wrap(Vector{T}, expsPtr, poly.length)

    return expsVec
end

function is_homog(poly::MPolyRingElem)
    expVecs = leading_exponent_vector.(terms(poly))
    if length(poly) == 0
        homog = true
        homogDegree = 0
    elseif length(poly) == 1
        homog = true
        homogDegree = sum(expVecs[1])
    else
        deg = sum(expVecs[1])
        for i in eachindex(expVecs)
            if sum(expVecs[i]) != deg
                homog = false
                homogDegree = -1
                break
            end
        end
        homog = true
        homogDegree = deg
    end

    return homog, homogDegree
end

struct MPowPlan <: OperationPlan
    key::Int
    len::Int
    nttPowPlans::Vector{NTTPowPlan}
    crtPlan::CuArray
    # memorysafe::Bool
end
