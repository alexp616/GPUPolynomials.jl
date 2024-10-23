using Oscar
using FLINT_jll

import Oscar.fpMPolyRingElem

function get_coeffs(poly::FqMPolyRingElem)
    coeffsDataType = get_uint_type((poly.data.coeffs_alloc ÷ poly.data.length) << 6)
    coeffsPtr = Base.unsafe_convert(Ptr{coeffsDataType}, poly.data.coeffs)
    coeffsVec = unsafe_wrap(Vector{coeffsDataType}, coeffsPtr, poly.data.length)

    return coeffsVec
end

function get_exps(poly::FqMPolyRingElem)
    expsDataType = get_uint_type((poly.data.exps_alloc ÷ poly.data.length) << 6)
    expsPtr = Base.unsafe_convert(Ptr{expsDataType}, poly.data.exps)
    expsVec = unsafe_wrap(Vector{expsDataType}, expsPtr, poly.data.length)

    return expsVec
end

function exp_matrix_to_vec(mat, bits)
    result = zeros(get_uint_type(Base._nextpow2(bits * size(mat, 1))), size(mat, 2))
    mat = reverse(mat, dims = 1)
    for i in axes(mat, 2)
        temp = zero(eltype(result))
        for j in axes(mat, 1)
            temp += mat[j, i] << (bits * (j - 1))
        end
        result[i] = temp
    end

    return result
end

function Oscar.fpMPolyRingElem(ctx::fpMPolyRing, a::Vector{UInt}, b::Matrix{UInt})
    z = fpMPolyRingElem(ctx)
    ccall((:nmod_mpoly_init2, libflint), Nothing,
          (Ref{fpMPolyRingElem}, Int, Ref{fpMPolyRing}),
          z, length(a), ctx)
    z.parent = ctx

    @inbounds for i in 1:length(a)
        if a[i] != zero(eltype(a))
            ccall((:nmod_mpoly_push_term_ui_ui, libflint), Nothing,
                (Ref{fpMPolyRingElem}, UInt, Ptr{UInt}, Ref{fpMPolyRing}),
                z, a[i], pointer(b, (i - 1) * ctx.nvars + 1), ctx)
        end
    end

    return z
end

function get_coeffs(poly::ZZMPolyRingElem)
    throw(ArgumentError("This hasn't been implemented yet"))
end

function get_exps(poly::ZZMPolyRingElem)
    throw(ArgumentError("This hasn't been implemented yet"))
end