abstract type CuMPolyRingElem end

function get_coeffs(poly::MPolyRingElem)
    if poly isa fpMPolyRingElem
        return get_coeffs(poly, UInt)
    else
        maxCoeff = BigInt(maximum(coefficients(poly)))
    end

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

function get_bound(homogDegree::Integer, numVars::Integer, coeffBound::Integer, pow::Integer)
    homogDegree = BigInt(homogDegree)
    numVars = BigInt(numVars)
    coeffBound = BigInt(coeffBound)
    return ((coeffBound - 1) * binomial(homogDegree + numVars - 1, numVars - 1)) ^ pow
end

function cpu_get_dense_representation(poly::CuMPolyRingElem, len::Int, bits::Int, type::DataType, key::Int, copies::Int)
    dest = zeros(type, len, copies)
    keyPowers = [key^i for i in 0:poly.parent.nvars - 2]

    coeffs = Array(poly.coeffs)
    exps = Array(poly.exps)

    mask = (one(eltype(exps)) << bits) - one(eltype(exps))
    for idx in eachindex(poly.coeffs)
        resultIdx = 1
        deg = exps[idx]
        for i in 1:poly.parent.nvars - 1
            resultIdx += (deg & mask) * keyPowers[i]
            deg >>= bits
        end

        for i in axes(dest, 2)
            dest[resultIdx, i] = coeffs[idx]
        end
    end

    return dest
end

function get_dense_representation(poly::CuMPolyRingElem, len::Int, bits::Int, type::DataType, key::Int, copies::Int)::CuMatrix
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