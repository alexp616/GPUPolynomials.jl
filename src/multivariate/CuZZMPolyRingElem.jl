import Oscar.ZZMPolyRingElem

mutable struct CuZZMPolyRingElem{T<:Integer}
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

function decode_exps_kernel!(exps::CuDeviceVector{T}, bits::Int, nvars::Int, dest::CuDeviceArray{T}) where T<:Unsigned
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(exps)
        curr = exps[idx]
        mask = (T(1) << bits) - T(1)
        for i in nvars:-1:1
            dest[i, idx] = curr & mask
            curr >>= bits
        end
    end

    return nothing
end

function decode_exps(exps::CuVector{T}, bits::Int, nvars::Int) where T<:Unsigned
    result = CUDA.zeros(UInt, nvars, length(exps))

    kernel = @cuda launch=false decode_exps_kernel!(exps, bits, nvars, result)
    config = launch_configuration(kernel.fun)
    threads = min(length(exps), config.threads)
    blocks = cld(length(exps), threads)

    kernel(exps, bits, nvars, result; threads = threads, blocks = blocks)

    return result
end

function Oscar.ZZMPolyRingElem(poly::CuZZMPolyRingElem{T}) where T<:Integer
    a = Array(poly.coeffs)
    b = Array(decode_exps(poly.exps, poly.bits, poly.parent.nvars))

    return ZZMPolyRingElem(poly.parent, a, b)
end

function get_bound(homogDegree::Integer, numVars::Integer, coeffBound::Integer, pow::Integer)
    homogDegree = BigInt(homogDegree)
    numVars = BigInt(numVars)
    coeffBound = BigInt(coeffBound)
    
    return ((coeffBound - 1) * binomial(homogDegree + numVars - 1, numVars - 1)) ^ pow
end

function MPowPlan(poly::CuZZMPolyRingElem, pow::Integer)
    if poly.homog
        bound = get_bound(poly.homogDegree, poly.parent.nvars, maximum(poly.coeffs) + 1, pow)
        resultDataType = get_uint_type(max(Base._nextpow2(Int(ceil(log2(bound)))), 64))

        resultTotalDegree = pow * poly.homogDegree
        # display("resultTotalDegree: $resultTotalDegree")
        key = resultTotalDegree + 1
        # display("poly.parent.nvars: $(poly.parent.nvars)")
        fftLen = Base._nextpow2(resultTotalDegree * key ^ (poly.parent.nvars - 2) + 1)
        # display("fftLen: $fftLen")
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
    # i dont care anymore
    wordSize = 64
    bits = div(wordSize, poly.parent.nvars)
    while bits < bitsNeeded
        wordSize *= 2
        bits = div(wordSize, poly.parent.nvars)
    end
    resultDegs = kronecker_to_bitpacked(resultEncodedDegs, poly.opPlan.key, poly.parent.nvars, poly.homogDegree * pow, bits, get_uint_type(wordSize))
    
    return CuZZMPolyRingElem(resultCoeffs, resultDegs, bits, true, poly.homogDegree * pow, poly.parent, EmptyPlan())
end

function cpu_get_dense_representation(poly::CuZZMPolyRingElem, len::Int, bits::Int, type::DataType, key::Int, copies::Int)
    dest = zeros(type, len, copies)
    keyPowers = [key^i for i in 0:poly.parent.nvars - 2]

    coeffs = Array(poly.coeffs)
    exps = Array(poly.exps)

    mask = (one(eltype(exps)) << bits) - one(eltype(exps))
    for idx in eachindex(poly.coeffs)
        resultIdx = 1
        deg = exps[idx]
        for i in 1:nvars - 1
            resultIdx += (deg & mask) * keyPowers[i]
            deg >>= bits
        end

        for i in axes(dest, 2)
            dest[resultIdx, i] = coeffs[idx]
        end
    end

    return dest
end

function get_dense_representation(poly::CuZZMPolyRingElem, len::Int, bits::Int, type::DataType, key::Int, copies::Int)::CuMatrix
    result = CUDA.zeros(type, len, copies)
    keyPowers = CuArray([key^i for i in 0:poly.parent.nvars - 2])

    kernel = @cuda launch=false get_dense_representation_kernel!(poly.coeffs, poly.exps, bits, keyPowers, result, poly.parent.nvars)
    config = launch_configuration(kernel.fun)
    threads = min(length(poly.coeffs), config.threads)
    blocks = cld(length(poly.coeffs), threads)

    kernel(poly.coeffs, poly.exps, bits, keyPowers, result, poly.parent.nvars; threads = threads, blocks = blocks)

    return result
end

function get_dense_representation_kernel!(coeffs::CuDeviceVector, exps::CuDeviceVector, bits, keyPowers::CuDeviceVector, dest, nvars)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if idx <= length(coeffs)
        mask = (one(eltype(exps)) << bits) - one(eltype(exps))
        resultIdx = 1
        deg = exps[idx]
        for i in 1:nvars - 1
            resultIdx += (deg & mask) * keyPowers[i]
            deg >>= bits
        end

        for i in axes(dest, 2)
            dest[resultIdx, i] = coeffs[idx]
        end

    end
    return
end