module Polynomials

export HomogeneousPolynomial, sort_to_kronecker_order, easy_print, kronecker, DensePolynomial, SparsePolynomial, random_homogeneous_polynomial, pretty_string, kronecker_substitution, decode_kronecker_substitution, change_encoding, convert_to_oscar

using CUDA
using Oscar

"""
    HomogeneousPolynomial

Struct that represents a homogeneous polynomial. Does nothing to check that the terms
are actually homogeneous.

# Arguments
- `coeffs`: Array of coefficients for each term of the polynomial
- `degrees`: 2d array, where each row represents the degrees of the variables of the term
- `homogeneousDegree`: The homogeneous degree of the polynomial
"""
mutable struct HomogeneousPolynomial{T<:Integer}
    coeffs::Vector{T}
    degrees::Array{Int, 2}
    homogeneousDegree::Int
end

function HomogeneousPolynomial(coeffs::Vector{Int}, degrees::Array{Int, 2})
    @assert length(coeffs) == size(degrees, 1)
    return HomogeneousPolynomial(coeffs, degrees, sum(degrees[1, :]))
end

"""
    HomogeneousPolynomial(p::FqMPolyRingElem)

Convert Oscar polynomial to HomogeneousPolynomial datastructure
"""
function HomogeneousPolynomial(p::FqMPolyRingElem, type::DataType = Int)
    coeffs = coefficients(p)
  
    # julia by default doesn't realize that "ZZ" is not
    # an array, so insert it as a one-element tuple "(ZZ,)"
    # so that julia will know not to broadcast along it.
    coeffs_as_int64arr = type.(lift.((ZZ,),coeffs))

    exp_vecs = leading_exponent_vector.(terms(p))

    # shamelessly taken from 
    # https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/2 
    exponent_mat = reduce(vcat,transpose.(exp_vecs))
  
    return HomogeneousPolynomial(coeffs_as_int64arr,exponent_mat)
end

function sort_to_kronecker_order(hp::HomogeneousPolynomial, key::Int)
    encodedDegrees = zeros(Int, size(hp.degrees, 1))
    
    for term in axes(hp.degrees, 1)
        encodedDegrees[term] = kronecker(hp.degrees[term, :], key)
    end
    perm = sortperm(encodedDegrees)
    encodedDegrees = encodedDegrees[perm]
    hp.coeffs = hp.coeffs[perm]
    hp.degrees = hp.degrees[perm, :]

    return
end

function easy_print(hp::HomogeneousPolynomial)
    for i in axes(hp.degrees, 1)
        println("$(hp.coeffs[i]) $(hp.degrees[i, :])")
    end
end

function kronecker(arr, key)
    result = 1
    for i in 1:length(arr) - 1
        result += arr[i] * key ^ (i - 1)
    end

    return result
end

"""
    DensePolynomial

Struct that represents a univariate, dense polynomial. For example,
DensePolynomial([a, b, c, d]) = a + bx + cx^2 + d^3
"""
mutable struct DensePolynomial{T<:Integer}
    coeffs::Vector{T}
    degree::Int
end

function DensePolynomial(coeffs::Vector{Int})
    return DensePolynomial(coeffs, length(coeffs) - 1)
end

"""
    SparsePolynomial

Struct that represents a univariate, sparse polynomial. For example,
SparsePolynomial([a, b, c, d], [1, 2, 5, 7]) = ax^7 + bx^5 + cx^2 + dx.

Figure out sorting stuff later
"""
mutable struct SparsePolynomial{T<:Integer}
    coeffs::Vector{Int}
    degrees::Vector{Int}
    numTerms::Int
end

function SparsePolynomial(coeffs::Vector{T}, degrees::Vector{Int}) where T<:Integer
    if length(coeffs) != length(degrees)
        throw("coeffs and degrees must be the same length! length(coeffs): $(length(coeffs)), length(degrees): $(length(degrees))")
    end
    new(coeffs, degrees, length(coeffs))
end

"""
    DensePolynomial(sp::SparsePolynomial, key::Int)

Constructs a DensePolynomial object from a sparse polynomial
"""
# function DensePolynomial(sp::SparsePolynomial{T}) where T<:Integer
#     polynomialDegree = sp.degrees[1]
#     result = zeros(T, polynomialDegree + 1)
#     for ()
# end

"""
    random_homogeneous_polynomial(numVars, numTerms, modulus)

Generates a random HomogeneousPolynomial object with numVars variables, 
numTerms non-zero terms, and non-zero coefficients less than modulus.
"""
function random_homogeneous_polynomial(numVars, numTerms, modulus)::HomogeneousPolynomial{Int}
    @assert numVars > 0 "number of variables must be greater than 0"
    @assert numTerms > 0 "number of terms must be greater than 0"
    @assert modulus > 0 "modulus must be greater than 0"
    if numVars == 4
        allPossibleMonomials = [4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]
    else
        allPossibleMonomials = generate_compositions(numVars, numVars)
    end
    @assert numTerms <= size(allPossibleMonomials, 1) "numTerms is greater than all possible monomials: $numTerms > $(size(allPossibleTerms, 1))"

    degrees2 = Array{Int}(undef, 1, numVars)
    random_set = randperm(size(allPossibleMonomials, 1))[1:numTerms]

    for i in eachindex(random_set)
        degrees2 = vcat(degrees2, allPossibleMonomials[random_set[i], :]')
    end

    degrees2 = degrees2[2:end, :]
    polynomial2 = HomogeneousPolynomial(rand(1:modulus - 1, numTerms), degrees2)
end

"""
    pretty_string(hp, variableNames = ["x", "y", "z", "w", "v", "p", "q", "r"])

Returns a pretty string representing the polynomial of a HomogeneousPolynomial object.
Pass in an array of strings of variables to customize.
"""
function pretty_string(hp::HomogeneousPolynomial, variableNames::Array{String} = ["x", "y", "z", "w", "v", "p", "q", "r"])
    if length(hp.coeffs) == 0
        return ""
    end

    numVars = size(hp.degrees, 2)
    resultstr = ""
    for term in eachindex(hp.coeffs)
        if hp.coeffs[term] != 0
            if hp.coeffs[term] != 1
                termstr = string(hp.coeffs[term], "*")
            else
                termstr = ""
            end

            for variable in 1:numVars - 1
                if hp.degrees[term, variable] != 0
                    if hp.degrees[term, variable] == 1
                        termstr *= string(variableNames[variable], "*")
                    else
                        termstr *= string(variableNames[variable], "^", hp.degrees[term, variable], "*")
                    end
                end
            end

            if hp.degrees[term, numVars] != 0
                if hp.degrees[term, numVars] == 1
                    termstr *= string(variableNames[numVars])
                else
                    termstr *= string(variableNames[numVars], "^", hp.degrees[term, numVars])
                end
            else
                termstr = termstr[1:end - 1]
            end

            resultstr *= string(termstr, " + ")
        end
    end
    
    return resultstr[1:end-3]
end

"""
    kronecker_substitution(hp::HomogeneousPolynomial, key::Int, length::Int)

Isomorphically polynomial represented by `hp` to 1-variate polynomial.
For example the variables x^11*y^2*z^1*w^2, with selecting a maximum degree of 16, encodes to:
x^(11 + 2 * 17 + 1 * 17^2)

We can ignore the last term because the terms have homogeneous degree
"""
function kronecker_substitution(hp::HomogeneousPolynomial, key::Int, length::Int = -1)::Vector{Int}
    if length == -1
        length = Base._nextpow2(key * (key + 1) ^ (size(hp.degrees, 2)))
    end
    result = zeros(Int, length)
    @inbounds begin
        for termidx in eachindex(hp.coeffs)
            # encoded = 1 because julia 1-indexing
            encoded = 1
            for d in 1:size(hp.degrees, 2) - 1
                encoded += hp.degrees[termidx, d] * key ^ (d - 1)
            end

            result[encoded] = hp.coeffs[termidx]
        end
    end

    return result
end

"""
    decode_kronecker_substitution(arr, key, numVars, totalDegree)

Maps 1-variate polynomial back to `numVars`-variate polynomial
"""
function decode_kronecker_substitution(arr::CuVector{T}, key::Int, numVars::Int, totalDegree::Int) where T<:Integer
    flags = map(x -> x != 0 ? 1 : 0, arr)
    indices = accumulate(+, flags)

    # there must be a faster way to do this
    CUDA.@allowscalar resultLen = indices[end]

    resultCoeffs = CUDA.zeros(eltype(arr), resultLen)
    resultDegrees = CUDA.zeros(Int, resultLen, numVars)

    function decode_kronecker_kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if idx <= length(arr)
            if flags[idx] != 0
                num = idx - 1
                termNum = indices[idx]
                for i in 1:numVars - 1
                    num, r = divrem(num, key)
                    resultDegrees[termNum, i] = r
                    totalDegree -= r
                end

                resultCoeffs[termNum] = arr[idx]
                resultDegrees[termNum, numVars] = totalDegree
            end
        end

        return
    end

    kernel = @cuda launch = false decode_kronecker_kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree)
    config = launch_configuration(kernel.fun)
    threads = min(length(arr), config.threads)
    blocks = cld(length(arr), threads)

    kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree; threads = threads, blocks = blocks)
    
    return HomogeneousPolynomial(Array(resultCoeffs), Array(resultDegrees), totalDegree)
end

"""
    change_encoding(num, e1, e2, numValues)

Return num, if it was encoded with e2 instead of e1
"""
function change_encoding(num::Int, e1::Int, e2::Int, numValues::Int)
    result = 0
    for i in 1:numValues
        num, r = divrem(num, e1)
        result += r * e2 ^ (i - 1)
    end

    return result
end

function convert_to_oscar(hp::HomogeneousPolynomial, ring::FqMPolyRing)
    vars = gens(ring)
    numVars = size(hp.degrees, 2)

    @assert length(vars) == numVars "Number of variables of hp and ring not compatible"

    result = zero(ring)

    for (i, coeff) in enumerate(hp.coeffs)
        expRow = hp.degrees[i, :]
        term = coeff * prod(vars[j] ^ expRow[j] for j in 1:numVars)
        result += term
    end

    return result
end

end