include("get_oscar_data.jl")

module Polynomials
using CUDA
using Oscar
using ..GetOscarData

import Base: show

export HomogeneousPolynomial, DensePolynomial, SparsePolynomial, MultivariatePolynomial, 
remove_zeros, encode_degrees, cpu_encode_degrees, gpu_encode_degrees, pretty_string, show, 
decode_degrees, cpu_decode_degrees, gpu_decode_degrees, homogkron, convert_to_oscar, invhomogkron

mutable struct HomogeneousPolynomial{T<:Number}
    coeffs::Vector{T}
    degrees::Array{<:Integer}
    homogeneousDegree::Int

    function HomogeneousPolynomial(coeffs::Vector{T}, degrees; check = false) where T<:Number
        homDeg = sum(degrees[:, 1])
        if !(check)
            for i in axes(degrees, 2)
                if sum(degrees[:, i]) != homDeg
                    throw("Polynomial is not homogeneous: non-homogeneous index: $i")
                end
            end
        end
        @assert length(coeffs) == size(degrees, 2)
        
        return new{T}(coeffs, degrees, homDeg)
    end

    function HomogeneousPolynomial(p::FqMPolyRingElem)
        coeffs, degrees = convert_to_gpu_representation(p)
        return HomogeneousPolynomial(coeffs, degrees)
    end
end

# Just leaving this dead in a ditch for now since kronecker stuff isn't fun: 
# no easy way to standardize a key. GPU is also probably slower than Oscar for 
# this problem anyways.
mutable struct MultivariatePolynomial{T<:Number}
    coeffs::Vector{T}
    degrees::Array{UInt32, 2}
end

mutable struct SparsePolynomial{T<:Number}
    coeffs::Vector{T}
    degrees::Vector{UInt64}
end

mutable struct DensePolynomial{T<:Number}
    coeffs::Vector{T}
end

function remove_zeros(hp::HomogeneousPolynomial{T}) where T<:Integer
    nonzeroIndices = findall(c -> c != T(0), hp.coeffs)

    hp.coeffs = hp.coeffs[nonzeroIndices]
    hp.degrees = hp.degrees[:, nonzeroIndices]

    return nothing
end

function kron(vec, key::Int)
    result = 0
    curr = 1
    for var in eachindex(vec)
        result += vec[var] * curr
        curr *= key
    end

    return result
end

function homogkron(vec, key::Int)
    result = 1
    curr = 1
    for var in 1:length(vec) - 1
        result += vec[var] * curr
        curr *= key
    end

    return result
end

function invkron(num::T, key::Int, numVars, dest) where T<:Integer
    for i in 1:numVars
        num, r = divrem(num, key)
        dest[i] = r
    end

    return nothing
end

function invhomogkron(num::T, key::Int, numVars, dest, totalDegree) where T<:Integer
    # println("decoding $num with key $key and totalDegree $totalDegree")
    for i in 1:numVars - 1
        num, r = divrem(num, key)
        dest[i] = r
        totalDegree -= r
    end
    dest[numVars] = totalDegree

    return nothing
end

function decode_degrees(encodedDegrees::Vector{UInt64}, key, numVars, homog, totalDegree = -1)
    return cpu_decode_degrees(encodedDegrees, key, numVars, homog, totalDegree)
end

function cpu_decode_degrees(encodedDegrees::Vector{UInt64}, key, numVars, homog, totalDegree = -1)
    result = zeros(UInt32, numVars, length(encodedDegrees))

    invkronfun = if !homog
        (term) -> invkron(encodedDegrees[term], key, numVars, view(result, :, term))
    else
        @assert totalDegree >= 0 "totalDegree has to be greater than 0!"
        (term) -> invhomogkron(encodedDegrees[term], key, numVars, view(result, :, term), totalDegree)
    end

    @inbounds for term in axes(result, 2)
        invkronfun(term)
    end

    return result
end

function gpu_decode_degrees_kernel!(invkronfun, encodedDegrees, len, result)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= len
        invkronfun(encodedDegrees[idx], view(result, :, idx))
    end

    return nothing
end

function gpu_decode_degrees(encodedDegrees::CuArray{UInt64}, key, numVars, homog, totalDegree = -1)
    len = length(encodedDegrees)
    result = CUDA.zeros(UInt32, numVars, length(encodedDegrees))
    
    invkronfun = if !homog
        (a, vec) -> invkron(a, key, numVars, vec)
    else
        (a, vec) -> invhomogkron(a, key, numVars, vec, totalDegree)
    end

    kernel = @cuda launch = false gpu_decode_degrees_kernel!(invkronfun, encodedDegrees, len, result)
    config = launch_configuration(kernel.fun)
    threads = min(length(encodedDegrees), config.threads)
    blocks = cld(length(encodedDegrees), threads)

    kernel(invkronfun, encodedDegrees, len, result; threads = threads, blocks = blocks)

    return result
end

# function HomogeneousPolynomial(sp::SparsePolynomial, key::Int, numVars, homogDegree)
#     # TODO
#     degrees = decode_degrees(sp.degrees, key, numVars, true, totalDegree)
#     return HomogeneousPolynomial(sp.coeffs, degrees, sp.degrees, homogDegree, false)
# end

function encode_degrees(degrees::Array{T}, key, homog) where T<:Integer
    return cpu_encode_degrees(degrees, key, homog)
end

function cpu_encode_degrees(degrees::Array{T}, key::Int, homog) where T<:Integer
    result = zeros(UInt64, size(degrees, 2))

    kronfun = homog ? homogkron : kron
    @inbounds begin
        for termNum in axes(degrees, 2)
            result[termNum] = kronfun(view(degrees, :, termNum), key)
        end
    end

    return result
end

function gpu_encode_degrees_kernel!(degrees, key::Int, result, kronfun)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(result)
        @inbounds begin
            result[idx] = kronfun(view(degrees, :, idx), key)
        end
    end

    return nothing
end

function gpu_encode_degrees(degrees::CuArray{T}, key::Int, homog) where T<:Integer
    result = CUDA.zeros(UInt64, size(degrees, 2))

    kronfun = homog ? homogkron : kron

    kernel = @cuda launch = false gpu_encode_degrees_kernel!(degrees, key, result, kronfun)
    config = launch_configuration(kernel.fun)
    threads = min(length(result), config.threads)
    blocks = cld(length(result), threads)

    kernel(degrees, key, result, kronfun; threads = threads, blocks = blocks)

    return result
end

function pretty_string(hp::Union{HomogeneousPolynomial, MultivariatePolynomial}, variableNames::Array{String} = ["x", "y", "z", "w", "a", "b", "c", "d"])
    if length(hp.coeffs) == 0
        return ""
    end

    numVars = size(hp.degrees, 1)
    if numVars > length(variableNames)
        throw(ArgumentError("Not enough variable names provided: $(length(variableNames)) variable names provided, $numVars variables needed."))
    end

    resultstr = ""

    if hp.coeffs[1] != 0
        termstr = ""

        if hp.coeffs[1] != 1
            termstr *= string(hp.coeffs[1])
        end

        for variable in 1:numVars
            if hp.degrees[variable, 1] != 0
                if termstr != "" && !endswith(termstr, "*")
                    termstr *= "*"
                end
                if hp.degrees[variable, 1] == 1
                    termstr *= string(variableNames[variable])
                else
                    termstr *= string(variableNames[variable], "^", hp.degrees[variable, 1])
                end
            end
        end

        resultstr *= termstr
    end

    for term in 2:length(hp.coeffs)
        if hp.coeffs[term] != 0
            termstr = ""

            if hp.coeffs[term] != 1
                termstr *= string(hp.coeffs[term])
            end

            for variable in 1:numVars
                if hp.degrees[variable, term] != 0
                    if termstr != "" && !endswith(termstr, "*")
                        termstr *= "*"
                    end
                    if hp.degrees[variable, term] == 1
                        termstr *= string(variableNames[variable])
                    else
                        termstr *= string(variableNames[variable], "^", hp.degrees[variable, term])
                    end
                end
            end

            resultstr *= " + "

            resultstr *= termstr
        end
    end
    
    return resultstr
end

function Base.show(io::IO, hp::HomogeneousPolynomial)
    println(io, pretty_string(hp))
end

function convert_to_oscar(hp::HomogeneousPolynomial, ring::MPolyRing)
    vars = gens(ring)
    numVars = size(hp.degrees, 1)

    @assert length(vars) == numVars "Number of variables of hp and ring not compatible"

    result = zero(ring)

    for (i, coeff) in enumerate(hp.coeffs)
        expRow = hp.degrees[:, i]
        term = coeff * prod(vars[j] ^ expRow[j] for j in 1:numVars)
        result += term
    end

    return result
end

end

# function sort_terms(hp::HomogeneousPolynomial)
#     perm = sortperm(hp.encodedDegrees, lt = !isless)

#     hp.encodedDegrees = hp.encodedDegrees[perm]
#     hp.coeffs = hp.coeffs[perm]
#     hp.degrees = hp.degrees[:, perm]

#     hp.sorted = true

#     return nothing
# end

# function Base.:(==)(hp1::HomogeneousPolynomial{T}, hp2::HomogeneousPolynomial) where T<:Number
#     if hp1.homogeneousDegree != hp2.homogeneousDegree
#         return false
#     end
#     if !hp1.sorted
#         sort_terms(hp1)
#     end
#     if !hp2.sorted
#         sort_terms(hp2)
#     end
#     remove_zeros(hp1)
#     remove_zeros(hp2)
#     return hp1.coeffs == T.(hp2.coeffs) && hp1.degrees == hp2.degrees
# end

# function DensePolynomial(sp::SparsePolynomial{T}) where T<:Number
#     if length(sp.coeffs) == 0
#         return DensePolynomial([])
#     end
#     if sp.sorted
#         maxdeg = sp.degrees[1]
#     else
#         maxdeg = maximum(sp.degrees)
#     end
    
#     resultCoeffs = zeros(T, maxdeg + 1)

#     for i in eachindex(sp.degrees)
#         resultCoeffs[sp.degrees[i] + 1] = sp.coeffs[i]
#     end

#     return DensePolynomial(resultCoeffs)
# end


# function SparsePolynomial(coeffs::Vector{T}, degrees::Vector) where T<:Number
#     return SparsePolynomial(coeffs, UInt64.(degrees), false)
# end

# function sort_terms(sp::SparsePolynomial)
#     perm = sortperm(sp.degrees, lt = !isless)

#     sp.coeffs = sp.coeffs[perm]
#     sp.degrees = sp.degrees[perm]

#     sp.sorted = true
    
#     return nothing
# end

# function remove_zeros(sp::SparsePolynomial{T}) where T<:Integer
#     nonzeroIndices = findall(c -> c != T(0), sp.coeffs)

#     sp.coeffs = sp.coeffs[nonzeroIndices]
#     sp.degrees = sp.degrees[nonzeroIndices]

#     return nothing
# end

# function SparsePolynomial(hp::HomogeneousPolynomial)
#     return SparsePolynomial(hp.coeffs, hp.encodedDegrees)
# end

# function Base.:(==)(sp1::SparsePolynomial{T}, sp2::SparsePolynomial) where T<:Number
#     if !sp1.sorted
#         sort_terms(sp1)
#     end
#     if !sp2.sorted
#         sort_terms(sp2)
#     end
#     remove_zeros(sp1)
#     remove_zeros(sp2)
#     return sp1.coeffs == T.(sp2.coeffs) && sp1.degrees == sp2.degrees
# end

# function MultivariatePolynomial(coeffs::Vector{T}, degrees::Array) where T<:Integer
#     return MultivariatePolynomial(coeffs, UInt32.(degrees))
# end


