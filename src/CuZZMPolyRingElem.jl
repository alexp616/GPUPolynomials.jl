import Oscar.ZZMPolyRingElem

struct CuZZMPolyRingElem{T<:Integer}
    coeffs::CuVector{T}
    exps::CuVector
    bits::Int
    homog::Bool
    homogDegree::Int
    parent::ZZMPolyRing

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem)
        coeffs = CuArray(get_coeffs(poly))
        exps = CuArray(get_exps(poly))

        bits = poly.bits
        homog, homogDegree = is_homog(poly)
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, homog, homogDegree, parent)
    end

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem, T::DataType)
        coeffs = CuArray(get_coeffs(poly, T))
        exps = CuArray(get_exps(poly))

        bits = poly.bits
        homog, homogDegree = is_homog(poly)
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, homog, homogDegree, parent)
    end

    function CuZZMPolyRingElem(poly::ZZMPolyRingElem, T::DataType, homogDegree::Int)
        coeffs = CuArray(get_coeffs(poly, T))
        exps = CuArray(get_exps(poly))
        
        bits = poly.bits
        parent = poly.parent

        return new{eltype(coeffs)}(coeffs, exps, bits, true, homogDegree, parent)
    end
end

function Base.convert(::CuZZMPolyRingElem, poly::ZZMPolyRingElem)
    return CuZZMPolyRingElem(poly)
end

function cu(poly::ZZMPolyRingElem)
    return CuZZMPolyRingElem(poly)
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

function get_coeffs(poly::ZZMPolyRingElem)
    maxCoeff = BigInt(maximum(coefficients(poly)))

    T = get_int_type(max(64, Int(ceil(log2(maxCoeff)))))
    return T.(coefficients(poly))
end

function get_coeffs(poly::ZZMPolyRingElem, T::DataType)
    coeffsPtr = Base.unsafe_convert(Ptr{T}, poly.coeffs)
    return unsafe_wrap(Vector{T}, coeffsPtr, poly.length)
end

function get_exps(poly::ZZMPolyRingElem)
    T = get_uint_type(Base._nextpow2(poly.bits * poly.parent.nvars))
    expsPtr = Base.unsafe_convert(Ptr{T}, poly.exps)
    expsVec = unsafe_wrap(Vector{T}, expsPtr, poly.length)

    return expsVec
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

function Oscar.ZZMPolyRingElem(ctx::ZZMPolyRing, a::Vector{T}, b::Matrix{UInt}) where T<:Integer
    a = get_ZZRingElem_vector(a)
    z = ZZMPolyRingElem(ctx)
    @ccall libflint.fmpz_mpoly_init2(z::Ref{ZZMPolyRingElem}, length(a)::Int, ctx::Ref{ZZMPolyRing})::Nothing
    z.parent = ctx

    for i in eachindex(a)
        @ccall libflint.fmpz_mpoly_push_term_fmpz_ui(z::Ref{ZZMPolyRingElem}, a[i]::Ref{ZZRingElem}, pointer(b, (i - 1) * ctx.nvars + 1)::Ptr{UInt}, ctx::Ref{ZZMPolyRing})::Nothing
    end

    sort_terms!(z)
    combine_like_terms!(z)
    return z
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

function convert(::ZZMPolyRingElem, poly::CuZZMPolyRingElem{T}) where T<:Integer
    return ZZMPolyRingElem(poly)
end

struct GPUPowPlan
    nttpowplans::Vector{NTTPowPlan}
    crtplan
    memorysafe::Bool
end

function ^(poly::CuZZMPolyRingElem, pow::Integer)
    @assert poly.homog
    
end