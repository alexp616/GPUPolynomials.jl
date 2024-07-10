module Delta1

include("gpu_ntt_pow.jl")
using Random

export HomogeneousPolynomial, random_homogeneous_polynomial, pretty_string, Delta1Pregen, pregen_delta1, delta1


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

"""
    HomogeneousPolynomial(coeffs, degrees)

Slightly lazier constructor for creating HomogeneousPolynomial object
"""
function HomogeneousPolynomial(coeffs::Vector{Int}, degrees::Array{Int, 2})
    @assert length(coeffs) == size(degrees, 1)
    return HomogeneousPolynomial(coeffs, degrees, sum(degrees[1, :]))
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


"""
    Delta1Pregen

Struct that contains almost anything that can be computed for the Delta1 algorithm. See pregen_delta1() to actually generate this

# Fields
- `numVars`: number of variables of polynomial
- `prime`: chosen prime number
- `step1pregen`: pregen object for first gpu_pow() step
- `step2pregen`: pregen object for second gpu_pow() step
- `inputLen1`: Length of first gpu_pow()'s input
- `inputLen2`: Length of second gpu_pow()'s input
- `key1`: Key for encoding multiple variables into 1 for first step
- `key2`: Key for encoding multiple variables into 1 for second step
"""
struct Delta1Pregen
    numVars::Int
    prime::Int
    step1pregen::GPUPowPregen
    step2pregen::GPUPowPregen
    inputLen1::Int
    inputLen2::Int
    key1::Int
    key2::Int
end

"""
    pregen_delta1(numVars, prime)

Generates a Delta1Pregen object corresponding to numVars and prime.
"""
function pregen_delta1(numVars::Int, prime::Int)
    # TODO very temporary
    
    if (numVars == 4 && prime == 5)
        primeArray1 = [2654209]
        primeArray2 = [13631489, 23068673]
        crtType1 = Int64
        crtType2 = Int128
        resultType1 = Int64
        resultType2 = Int64
    elseif (numVars == 4 && prime == 7)
        primeArray1 = [65537, 114689, 147457]
        primeArray2 = [167772161, 377487361, 469762049]
        crtType1 = Int64
        crtType2 = Int128
        resultType1 = Int64
        resultType2 = Int128
    else
        throw("I havent figured out these bounds yet")
    end

    step1ResultDegree = numVars * (prime - 1)
    key1 = step1ResultDegree + 1

    inputLen1 = numVars * (step1ResultDegree + 1) ^ (numVars - 2) + 1
    fftSize1 = nextpow(2, (inputLen1 - 1) * (prime - 1) + 1)

    step1Pregen = pregen_gpu_pow(primeArray1, fftSize1, crtType1, resultType1)

    step2ResultDegree = step1ResultDegree * prime
    key2 = step2ResultDegree + 1
    inputLen2 = step1ResultDegree * (step2ResultDegree + 1) ^ (numVars - 2) + 1
    fftSize2 = nextpow(2, (inputLen2 - 1) * prime + 1)

    step2Pregen = pregen_gpu_pow(primeArray2, fftSize2, crtType2, resultType2)

    return Delta1Pregen(numVars, prime, step1Pregen, step2Pregen, inputLen1, inputLen2, key1, key2)
end

"""
    kronecker_substitution(hp, pow, pregen)

Isomorphically polynomial represented by `hp` to 1-variate polynomial.
For example the variables x^11*y^2*z^1*w^2, with selecting a maximum degree of 16, encodes to:
x^(11 + 2 * 17 + 1 * 17^2)

We can ignore the last term because the terms have homogeneous degree
"""
function kronecker_substitution(hp::HomogeneousPolynomial, pow::Int, pregen::Delta1Pregen)::Vector{Int}
    key = hp.homogeneousDegree * pow + 1

    result = zeros(Int, pregen.inputLen1)
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
function decode_kronecker_substitution(arr, key, numVars, totalDegree)
    flags = map(x -> x != 0 ? 1 : 0, arr)
    indices = accumulate(+, flags)

    # there must be a faster way to do this
    resultLen = Array(indices)[end]

    resultCoeffs = CUDA.zeros(Int, resultLen)
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

"""
    change_polynomial_encoding(p, pregen)

Intermediate step of Delta1: Instead of decoding the result of the first step, then re-encoding it, we can just map each term directly to the input of the next step
"""
function change_polynomial_encoding(p::CuVector{Int}, pregen::Delta1Pregen)
    result = CUDA.zeros(Int, pregen.inputLen2)

    kernel = @cuda launch = false change_polynomial_encoding_kernel(p, result, pregen.key1, pregen.key2, pregen.numVars)
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(p))
    blocks = cld(length(p), threads)

    kernel(p, result, pregen.key1, pregen.key2, pregen.numVars; threads = threads, blocks = blocks)

    return result
end

"""
    change_polynomial_encoding_kernel()

Kernel function for change_polynomial_encoding()
"""
function change_polynomial_encoding_kernel(source, dest, key1, key2, numVars)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds begin
        if tid <= length(source)
            resultidx = change_encoding(tid - 1, key1, key2, numVars - 1) + 1
            dest[resultidx] = source[tid]
        end
    end

    return
end

"""
    cpu_remove_pth_power_terms!(big,small)

Subtracts small .^ p from big as polynomials.

Here's the idea: small already in kronecker order
(we'll have to use decode_kronecker_substitution there, or else sort it)
So loop through each of the terms of gpuOutput, keeping a counter on cpuOutput.
When you find one that is the same, set it's coefficient to zero,
increment the cpuOutput counter, and then continue
By the end, we should have reached the end of both arrays.

This will have time complexity O(nTerms(gpuOutput))

This is really a modified version of Johnson's addition/subtraction algorithm. 
Should be fast on the cpu

Note that this means if the two polynomials don't have the same term order,
this can fail.

# Arguments:
- `big::HomogeneousPolynomial`
- `small::HomogeneousPolynomial`
- `p`: power to remove terms from
"""
function cpu_remove_pth_power_terms!(big,small,p)
    
    i = 1
    k = 1

    while i ≤ length(small.coeffs)
        # for a standard addition, remove the p
        smalldegs = p .* small.degrees[i,:]

        bigdegs = big.degrees[k,:]
        while smalldegs != bigdegs
            k = k + 1
            bigdegs = big.degrees[k,:]
        end
        # now we know that the term in row k of big is a pth power of 
        # the term in row i of small

        # this is a subtraction
        big.coeffs[k] -= (small.coeffs[i])^p

        i = i + 1
    end

    nothing
end

"""
    generate_compositions(n, k)

Escaped hypertriangle code because still useful for generating all possible monomials
"""
function generate_compositions(n, k, type::DataType = Int64)
    compositions = zeros(type, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(type, k)
    current_composition[1] = n
    idx = 1
    while true
        compositions[idx, :] .= current_composition
        idx += 1
        v = current_composition[k]
        if v == n
            break
        end
        current_composition[k] = 0
        j = k - 1
        while 0 == current_composition[j]
            j -= 1
        end
        current_composition[j] -= 1
        current_composition[j + 1] = 1 + v
    end

    return compositions
end

function delta1(hp::HomogeneousPolynomial, prime, pregen::Delta1Pregen)
    @assert prime == pregen.prime && size(hp.degrees, 2) == pregen.numVars "Current pregen isn't compatible with input"

    # Raising f ^ p - 1
    input1 = CuArray(kronecker_substitution(hp, prime - 1, pregen))
    output1 = gpu_pow(input1, prime - 1; pregen = pregen.step1pregen)

    # Reduce mod p
    output1 = map(num -> faster_mod(num, prime), output1)
    # Raising g ^ p
    input2 = CuArray(change_polynomial_encoding(output1, pregen))

    output2 = gpu_pow(input2, prime; pregen = pregen.step2pregen)
    result = decode_kronecker_substitution(output2, pregen.key2, pregen.numVars, pregen.key2 - 1)
    
    intermediate = decode_kronecker_substitution(output1, pregen.key1, pregen.numVars, pregen.key1 - 1)
    cpu_remove_pth_power_terms!(result,intermediate,prime)

    # Here for debug purposes
    # for i in 1:length(result.coeffs)
    # for i in 80250:80350
    #     if result.coeffs[i] % prime != 0
    #         println("WRONG COEFFICIENT $i: $(result.coeffs[i]), $(result.degrees[i, :])")
    #     end
    # end

    result.coeffs ./= prime

    result.coeffs .%= prime

    #return (intermediate, result, finalresult
    result
end

function test_delta1()
    coeffs = [1, 1, 1, 1]
    degrees = [
        4 0 0 0
        0 4 0 0
        0 0 4 0
        0 0 0 4
    ]

    polynomial1 = HomogeneousPolynomial(coeffs, degrees)

    println("Time to pregenerate everything")
    CUDA.@time pregen = pregen_delta1(4, 5)

    println("Time to raise 4-variate polynomial to the 4th and 5th power for the first time: ")
    CUDA.@time result = delta1(polynomial1, 5, pregen)


    allPossibleMonomials = [4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]
    println("Time to raise different 4-variate polynomials to the 4th and 5th power: ")
    for i in 1:10
        degrees2 = Array{Int}(undef, 1, 4)
        for monNum in axes(allPossibleMonomials, 1)
            if rand((0, 1)) == 1
                degrees2 = vcat(degrees2, allPossibleMonomials[monNum, :]')
            end
        end
        degrees2 = degrees2[2:end, :]
        polynomial2 = HomogeneousPolynomial(rand(1:4, size(degrees2, 1)), degrees2)
        println("Trial $i")
        CUDA.@time result2 = delta1(polynomial2, 5, pregen)
    end
    CUDA.@profile result = delta1(polynomial1, 5, pregen)
end

end