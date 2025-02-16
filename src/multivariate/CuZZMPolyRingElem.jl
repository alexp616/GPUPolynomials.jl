import Oscar.ZZMPolyRingElem

mutable struct CuZZMPolyRingElem{T<:Integer} <: CuMPolyRingElem
    coeffs::CuVector{T}
    exps::CuVector
    bits::Int
    homog::Bool
    homogDegree::Int
    parent::ZZMPolyRing
    opPlan::OperationPlan

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem)
        coeffs = CuArray(get_coeffs(poly))
        exps = CuArray(get_exps(poly))

        bits = poly.bits
        homog, homogDegree = is_homog(poly)
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, homog, homogDegree, parent, EmptyPlan())
    end

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem, T::DataType)
        coeffs = CuArray(get_coeffs(poly, T))
        exps = CuArray(get_exps(poly))

        bits = poly.bits
        homog, homogDegree = is_homog(poly)
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, homog, homogDegree, parent, EmptyPlan())
    end

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem, T::DataType, homogDegree::Int)
        coeffs = CuArray(get_coeffs(poly, T))
        exps = CuArray(get_exps(poly))
        
        bits = poly.bits
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, true, homogDegree, parent, EmptyPlan())
    end

    function CuZZMPolyRingElem(coeffs::CuVector, exps::CuVector, bits::Int, homog::Bool, homogDegree::Int, parent::ZZMPolyRing, opPlan::OperationPlan)
        return new{eltype(coeffs)}(coeffs, exps, bits, homog, homogDegree, parent, opPlan)
    end
end

function CUDA.cu(poly::ZZMPolyRingElem)
    return CuZZMPolyRingElem(poly)
end

function Oscar.ZZMPolyRingElem(ctx::ZZMPolyRing, a::Vector{T}, b::Matrix{UInt}) where T<:Integer
    a = get_ZZRingElem_vector(a)
    z = ZZMPolyRingElem(ctx)
    @ccall libflint.fmpz_mpoly_init2(z::Ref{ZZMPolyRingElem}, length(a)::Int, ctx::Ref{ZZMPolyRing})::Nothing
    z.parent = ctx

    for i in eachindex(a)
        @ccall libflint.fmpz_mpoly_push_term_fmpz_ui(z::Ref{ZZMPolyRingElem}, a[i]::Ref{ZZRingElem}, pointer(b, (i - 1) * ctx.nvars + 1)::Ptr{UInt}, ctx::Ref{ZZMPolyRing})::Nothing
    end

    sort_terms!(z)
    return z
end

function get_ZZRingElem_vector(a::Vector{T})::Vector{ZZRingElem} where T<:Integer
    result = zeros(ZZRingElem, length(a))
    # https://flintlib.org/doc/fmpz.html#c.fmpz
    cutoff = UInt(1) << 62
    for i in eachindex(a)
        if a[i] < cutoff
            result[i] = ZZRingElem(UInt(a[i]))
        else
            result[i] = ZZRingElem(BigInt(a[i]))
        end
    end

    return result
end

function Oscar.ZZMPolyRingElem(poly::CuZZMPolyRingElem{T}) where T<:Integer
    a = Array(poly.coeffs)
    b = Array(decode_exps(poly.exps, poly.bits, poly.parent.nvars))

    return ZZMPolyRingElem(poly.parent, a, b)
end

function MPowPlan(poly::CuZZMPolyRingElem, pow::Integer)
    if poly.homog
        bound = get_bound(poly.homogDegree, poly.parent.nvars, maximum(poly.coeffs) + 1, pow)
        resultDataType = get_uint_type(max(Base._nextpow2(Int(ceil(log2(bound)))), 64))

        resultTotalDegree = pow * poly.homogDegree
        key = resultTotalDegree + 1
        fftLen = Base._nextpow2(resultTotalDegree * key ^ (poly.parent.nvars - 2) + 1)
        possiblePrimes = find_ntt_primes(fftLen)
        primeArray = UInt64[]
        currTotal = BigInt(1)
        idx = 1
        while currTotal < bound
            prime = possiblePrimes[idx]
            idx += 1
            currTotal *= prime
            push!(primeArray, prime)
        end

        nttPowPlans = NTTPowPlan[]
        for p in primeArray
            nttPowPlan = NTTPowPlan(fftLen, pow, p)
            push!(nttPowPlans, nttPowPlan)
        end

        crtPlan = plan_crt(resultDataType.(primeArray))

        return MPowPlan(key, fftLen, nttPowPlans, crtPlan)
    else
        throw("")
    end
end

function Base.:^(a::CuZZMPolyRingElem, p::Integer)
    if !(a.opPlan isa MPowPlan)
        a.opPlan = MPowPlan(a, p)
    end
    if a.homog
        return homog_poly_pow(a, p)
    else
        throw(ArgumentError("Non-homogeneous polynomial arithmetic hasn't been implemented yet"))
    end
end

function homog_poly_pow(poly::CuZZMPolyRingElem, pow::Integer)
    type = typeof(poly.opPlan.nttPowPlans[1].forwardPlan.p)
    stackedVec = get_dense_representation(poly, poly.opPlan.len, poly.bits, type, poly.opPlan.key, length(poly.opPlan.nttPowPlans))

    currPtr = pointer(stackedVec)
    for planNum in eachindex(poly.opPlan.nttPowPlans)
        vec = CUDA.unsafe_wrap(CuVector{type}, currPtr, poly.opPlan.len)
        ntt_pow!(vec, poly.opPlan.nttPowPlans[planNum])
        currPtr += sizeof(type) * poly.opPlan.len
    end

    resultUnreducedCoeffs, resultEncodedDegs = sparsify(stackedVec)

    resultCoeffs = build_result(resultUnreducedCoeffs, poly.opPlan.crtPlan)

    bitsNeeded = Int(ceil(log2(poly.homogDegree * pow)))
    wordSize = 64
    bits = div(wordSize, poly.parent.nvars)
    while bits < bitsNeeded
        wordSize *= 2
        bits = div(wordSize, poly.parent.nvars)
    end
    resultDegs = kronecker_to_bitpacked(resultEncodedDegs, poly.opPlan.key, poly.parent.nvars, poly.homogDegree * pow, bits, get_uint_type(wordSize))
    
    return CuZZMPolyRingElem(resultCoeffs, resultDegs, bits, true, poly.homogDegree * pow, poly.parent, EmptyPlan())
end