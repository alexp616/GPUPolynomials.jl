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
DensePolynomial([a, b, c, d]) = a + bx + cx^2 + dx^3
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

Struct that represents a univariate, sparse polynomial.
"""
mutable struct SparsePolynomial{T<:Integer}
    coeffs::Vector{Int}
    degrees::Vector{Int}
    numTerms::Int
end

function SparsePolynomial(coeffs::Vector{Int}, degrees::Vector{Int})
    @assert length(coeffs) == length(degrees)
    new(coeffs, degrees, length(coeffs))
end

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