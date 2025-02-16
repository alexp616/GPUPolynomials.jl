import Oscar.ZZPolyRingElem

mutable struct CuZZPolyRingElem{T<:Integer} <: CuPolyRingElem
    coeffs::CuVector{T}
    length::Int
    parent::ZZPolyRing
    opPlan::OperationPlan

    function CuZZPolyRingElem(poly::ZZPolyRingElem, T::DataType = Nothing)
        coeffs = CuArray(get_coeffs(poly, T))
        return new{eltype(coeffs)}(coeffs, poly.length, poly.parent, EmptyPlan())
    end

    function CuZZPolyRingElem(coeffs::CuVector{T}, length::Int, parent::ZZPolyRing, opPlan::OperationPlan) where T<:Integer
        return new{T}(coeffs, length, parent, opPlan)
    end
end

function CUDA.cu(poly::ZZPolyRingElem)
    return CuZZPolyRingElem(poly, Nothing)
end

function Oscar.ZZPolyRingElem(poly::CuZZPolyRingElem)
    result = ZZPolyRingElem(Array(poly.coeffs))
    result.parent = poly.parent
    return result
end

function Base.:+(a::CuZZPolyRingElem, b::Integer)
    CUDA.@allowscalar a.coeffs[1] += b
end

function Base.:+(a::CuZZPolyRingElem, b::CuZZPolyRingElem)
    if a.parent != b.parent
        throw(ArgumentError("Polynomials must have the same parent!"))
    end

    c = CUDA.zeros(promote_type(eltype(a.coeffs), eltype(b.coeffs)), max(length(a), length(b)))

    kernel = @cuda launch=false add_kernel(a.coeffs, b.coeffs, c)
    config = launch_configuration(kernel.fun)
    threads = min(length(c), config.threads)
    blocks = cld(length(c), threads)

    if length(a) > length(b)
        kernel(a.coeffs, b.coeffs, c; threads = threads, blocks = blocks)
    else
        kernel(b.coeffs, a.coeffs, c; threads = threads, blocks = blocks)
    end

    return CuZZPolyRingElem(c, length(c), a.parent, EmptyPlan())
end

function add_kernel(a::CuDeviceVector, b::CuDeviceVector, c::CuDeviceVector)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(b)
        c[idx] = a[idx] + b[idx]
    elseif idx <= length(a)
        c[idx] = a[idx]
    end
    
    return
end

function Base.:-(a::CuZZPolyRingElem)
    coeffs = map(x -> -x, a.coeffs)
    return CuZZPolyRingElem(coeffs, length(a), a.parent, EmptyPlan())
end

function Base.:-(a::CuZZPolyRingElem, b::CuZZPolyRingElem)
    return +(a, -b)
end

function Base.:-(a::CuZZPolyRingElem, b::Integer)
    coeffs = copy(a)
    CUDA.@allowscalar coeffs[1] -= b
    return CuZZPolyRingElem(coeffs, length(a), a.parent, EmptyPlan())
end

function MulPlan(a::CuZZPolyRingElem, b::CuZZPolyRingElem, type::DataType = Int64)
    resultLength = length(a) + length(b) - 1
    fftLen = Base._nextpow2(resultLength)

    # Commpetely arbitrary
    bound = typemax(type) >> 4

    primeArray = UInt64[]
    possiblePrimes = find_ntt_primes(fftLen)
    currTotal = BigInt(1)
    idx = 1
    while currTotal < bound
        prime = possiblePrimes[idx]
        idx += 1
        currTotal *= prime
        push!(primeArray, prime)
    end

    nttMulPlans = NTTMulPlan[]
    for p in primeArray
        nttMulPlan = NTTMulPlan(fftLen, p)
        push!(nttMulPlans, nttMulPlan)
    end

    crtPlan = plan_crt(type.(primeArray))

    return MulPlan{UInt64}(fftLen, nttMulPlans, crtPlan)
end

function Base.:*(a::CuZZPolyRingElem, b::CuZZPolyRingElem)
    @assert a.parent == b.parent
    if !(a.opPlan isa MulPlan)
        a.opPlan = MulPlan(a, b)
    end
    plan = a.opPlan

    type = eltype(plan)

    stackedveca = reshape(repeat(vcat(a.coeffs, CUDA.zeros(type, plan.len - length(a))), length(plan.nttMulPlans)), plan.len, length(plan.nttMulPlans))

    stackedvecb = reshape(repeat(vcat(b.coeffs, CUDA.zeros(type, plan.len - length(b))), length(plan.nttMulPlans)), plan.len, length(plan.nttMulPlans))

    # stackedveca = repeat(vcat(a.coeffs, CUDA.zeros(type, plan.len - length(a))), length(plan.nttMulPlans))

    # stackedvecb = repeat(vcat(b.coeffs, CUDA.zeros(type, plan.len - length(b))), length(plan.nttMulPlans))

    aptr = pointer(stackedveca)
    bptr = pointer(stackedvecb)
    for p in plan.nttMulPlans
        veca = CUDA.unsafe_wrap(CuVector{type}, aptr, plan.len)
        vecb = CUDA.unsafe_wrap(CuVector{type}, bptr, plan.len)
        
        ntt_mul!(veca, vecb, p)
        aptr += plan.len * sizeof(type)
        bptr += plan.len * sizeof(type)
    end

    nttResult = build_result(stackedveca, plan.crtPlan)
    resultLen = length(a) + length(b) - 1
    
    result = CUDA.unsafe_wrap(CuVector{eltype(nttResult)}, pointer(nttResult), resultLen)

    useless = CUDA.unsafe_wrap(CuVector{eltype(nttResult)}, pointer(nttResult) + sizeof(eltype(nttResult)) * resultLen, length(nttResult) - resultLen)

    CUDA.unsafe_free!(useless)

    return CuZZPolyRingElem(result, resultLen, a.parent, EmptyPlan())
end

function Base.:*(a::CuZZPolyRingElem, b::Integer)
    coeffs = map(t -> t * b, a.coeffs)
    return CuZZPolyRingElem(coeffs, length(coeffs), a.parent, EmptyPlan())
end

function PowPlan(a::CuZZPolyRingElem, pow::Integer, type::DataType = Int64)
    resultLength = (length(a) - 1) * pow + 1
    fftLen = Base._nextpow2(resultLength)

    # Commpetely arbitrary
    bound = typemax(type) >> 4

    primeArray = UInt64[]
    possiblePrimes = find_ntt_primes(fftLen)
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

    crtPlan = plan_crt(type.(primeArray))

    return PowPlan{UInt64}(fftLen, nttPowPlans, crtPlan)
end

function Base.:^(a::CuZZPolyRingElem, p::Integer)
    if !(a.opPlan isa PowPlan)
        a.opPlan = PowPlan(a, p)
    end
    plan = a.opPlan

    type = eltype(plan)

    stackedveca = reshape(repeat(vcat(a.coeffs, CUDA.zeros(type, plan.len - length(a))), length(plan.nttPowPlans)), plan.len, length(plan.nttPowPlans))

    aptr = pointer(stackedveca)
    for powplan in plan.nttPowPlans
        veca = CUDA.unsafe_wrap(CuVector{type}, aptr, plan.len)

        ntt_pow!(veca, powplan)
        aptr += plan.len * sizeof(type)
    end

    nttResult = build_result(stackedveca, plan.crtPlan)
    resultLen = (length(a) - 1) * p + 1

    result = CUDA.unsafe_wrap(CuVector{eltype(nttResult)}, pointer(nttResult), resultLen)

    useless = CUDA.unsafe_wrap(CuVector{eltype(nttResult)}, pointer(nttResult) + sizeof(eltype(nttResult)) * resultLen, length(nttResult) - resultLen)

    CUDA.unsafe_free!(useless)

    return CuZZPolyRingElem(result, resultLen, a.parent, EmptyPlan())
end