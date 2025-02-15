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
    CUDA.@allowscalar a[1] += b
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
    coeffs = map(x -> -x, a)
    return CuZZPolyRingElem(coeffs, length(a), a.parent, EmptyPlan())
end

function Base.:-(a::CuZZPolyRingElem, b::CuZZPolyRingElem)
    return +(a, -b)
end

struct MulPlan{T<:Integer}
    len::Int
    nttMulPlans::Vector{NTTMulPlan{T}}
    crtPlan::CuArray

    function MulPlan(a::CuZZPolyRingElem, b::CuZZPolyRingElem, type::DataType = Int64)
        resultTotalDegree = length(a) + length(b) - 1
        fftLen = Base._nextpow2(resultTotalDegree)

        bound = typemax(type)

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

        return new{UInt64}(fftLen, nttMulPlans, crtPlan)
    end
end

function Base.:*(a::CuZZPolyRingElem, b::CuZZPolyRingElem)
    plan = a.opPlan
    if !(a.opPlan isa MulPlan)
        plan = MulPlan(a, b)
    end

    type = eltype(plan)

    stackedveca = reshape(repeat(vcat(a.coeffs, CUDA.zeros(type, plan.fftLen - length(a)), fftLen, length(MulPlan.nttMulPlans))), 3)

    stackedvecb = reshape(repeat(vcat(b.coeffs, CUDA.zeros(type, plan.fftLen - length(b)), fftLen, length(MulPlan.nttMulPlans))), 3)

    aptr = pointer(stackedveca)
    bptr = pointer(stackedvecb)
    for p in plan.nttMulPlans
        veca = CUDA.unsafe_wrap(CuVector{type}, aptr, plan.len)
        vecb = CUDA.unsafe_wrap(CuVector{type}, bptr, plan.len)
        # TODO ejfiwjeiwiofjao;jefoiwjf
    end

    nttResult = build_result(stackedveca, plan.crtPlan)
    resultLen = length(a) + length(b) - 1
    
    CUDA.@sync result = CUDA.unsafe_wrap(Vector{eltype(nttResult)}, pointer(nttResult), resultLen)

    useless = CUDA.unsafe_wrap(Vector{eltype(nttResult)}, pointer(nttResult) + sizeof(eltype(nttResult)) * resultLen, length(nttResult) - resultLen)

    CUDA.unsafe_free(useless)

    return result
end

function Base.:*(a::CuZZPolyRingElem, b::Integer)

end

struct PowPlan

end

function Base.:^(a::CuZZPolyRingElem, p::Integer)

end