module GetOscarData

export get_data, convert_data_to_oscar
using Oscar

# TODO
# Sort_to_kronecker_order
# ZZPolyRingElem

function get_int_type(n)
    return eval(Symbol("Int", n))
end

# get_data() will have to be stored away for a bit because f.data.bits
# sometimes doesn't convert to an Integer type, meaning the exponents
# vector is bitpacked. This makes direct manipulation a bit harder than normal
# Stuff about bit packing: 
# https://github.com/flintlib/flint/blob/3dbe2399e74f24d8a883a97d736aee8eaf331b1c/src/mpoly.h#L54C1-L59C3

# function get_data(poly::FqMPolyRingElem)
#     numCoeffs = length(poly)
#     numVars = poly.parent.data.nvars

#     # I think for Fq, coeffs types are fixed at Int64 and we pray the user doesn't
#     # want to use primes bigger than 2^63 - 1
#     coeffsDataType = Int64
#     expsDataType = get_int_type(poly.data.bits)

#     coeffsPtr = Base.unsafe_convert(Ptr{coeffsDataType}, poly.data.coeffs)
#     expsPtr = Base.unsafe_convert(Ptr{expsDataType}, poly.data.exps)

#     coeffsVec = unsafe_wrap(Vector{coeffsDataType}, coeffsPtr, numCoeffs)
#     expsVec = unsafe_wrap(Vector{expsDataType}, expsPtr, numCoeffs * numVars)
#     expsArray = reshape(expsVec, numVars, numCoeffs)

#     return coeffsVec, expsArray
# end

# FqMPolyRingElem.data is just the gr_mpoly_struct defined here:
# https://github.com/flintlib/flint/blob/main/src/gr_mpoly.h
function get_data(poly::FqMPolyRingElem)
    println("Running get_data")
    coeffsDataType = Int64
    expsDataType = get_int_type(Base._nextpow2(poly.data.bits))

    numCoeffs = length(poly)

    coeffsPtr = Base.unsafe_convert(Ptr{coeffsDataType}, poly.data.coeffs)
    coeffsVec = unsafe_wrap(Vector{coeffsDataType}, coeffsPtr, numCoeffs)

    expVecs = reverse!.(leading_exponent_vector.(terms(poly)))
    expMat = expsDataType.(reduce(hcat, expVecs))

    println("Finished get_data")
    return coeffsVec, expMat
end

# This method isn't perfect: FLINT optimizes FqMPolyRingElem.data.bits to minimize the machine
# words needed to store exps and packs exponent vectors as such, and I don't want to deal with 
# bit packing in Julia right now. Basically, this will take slightly more space, but is faster 
# than any other method to convert to Oscar I can think of
function convert_data_to_oscar(coeffsVec::Vector{Int}, expsArray::Matrix{T}, parentRing::FqMPolyRing; sorted = false) where T<:Integer
    println("Running convert_data")
    result = zero(parentRing)
    numVars = parentRing.data.nvars

    if length(coeffsVec) != size(expsArray, 2)
        throw(ArgumentError("coeffsVec doesn't have same number of terms as expsArray"))
    end
    if numVars != size(expsArray, 1)
        throw(ArgumentError("expsArray doesn't have the same number of variables as parentRing"))
    end

    result.data.coeffs = reinterpret(Ptr{nothing}, pointer(coeffsVec))
    result.data.exps = reinterpret(Ptr{nothing}, pointer(expsArray))
    result.data.length = length(coeffsVec)
    result.data.bits = sizeof(eltype(expsArray)) * 8
    result.data.coeffs_alloc = result.data.length
    result.data.exps_alloc = cld(result.data.length * result.data.bits * numVars, 8)

    println("Finished convert_data")
    return result
end

function convert_data_to_oscar(coeffsVec::Vector{Int}, expsArray::Matrix{T}, parentRing::ZZMPolyRing; sorted = false) where T<:Integer
    return nothing
end
end