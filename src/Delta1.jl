module Delta1

include("gpu_ntt_pow.jl")
include("Polynomials.jl")
using Random

export Delta1Pregen, pregen_delta1, raise_to_p_minus_1, delta1


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
    # forgetIndices::CuVector
end

function generate_forget_indices(numVars, prime, key)
    arr = generate_compositions(numVars * (prime - 1), numVars)
    arr .*= prime
    display(arr)

    result = zeros(Int, size(arr, 1))
    for i in axes(arr, 1)
        encoded = 1
        for d in 1:size(arr, 2) - 1
            encoded += arr[i, d] * key ^ (d - 1)
        end

        result[i] = encoded
    end

    return CuArray(result)
end

"""
    pregen_delta1(numVars, prime)

Generates a Delta1Pregen object corresponding to numVars and prime.
"""
function pregen_delta1(numVars::Int, prime::Int)
    if (numVars, prime) == (4, 2)
        primeArray1 = [257]
        primeArray2 = [12289]
        crtType1 = Int64
        crtType2 = Int64
        resultType1 = Int64
        resultType2 = Int64
    elseif (numVars, prime) == (4, 3)
        primeArray1 = [12289]
        primeArray2 = [114689]
        crtType1 = Int64
        crtType2 = Int64
        resultType1 = Int64
        resultType2 = Int64
    elseif (numVars, prime) == (4, 5)
        primeArray1 = [2654209]
        primeArray2 = [13631489, 23068673]
        crtType1 = Int64
        crtType2 = Int128
        resultType1 = Int64
        resultType2 = Int64
    elseif (numVars, prime) == (4, 7)
        primeArray1 = [65537, 114689, 147457]
        primeArray2 = [167772161, 377487361, 469762049]
        crtType1 = Int128
        crtType2 = Int256
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

    # forgetIndices = generate_forget_indices(numVars, prime, key2)

    return Delta1Pregen(numVars, prime, step1Pregen, step2Pregen, inputLen1, inputLen2, key1, key2)
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
function kernel(source, dest, key1, key2, numVars)
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

    n = size(small.degrees,2)

    # pre-allocation
    smalldegs = zeros(eltype(small.degrees),n)
    bigdegs = zeros(eltype(big.degrees),n)

    function setslice_noalloc!(target,source,k)
        for j = 1:n
            target[j] = source[k,j]
        end
    end

    while i ≤ length(small.coeffs)
        # for a standard addition, remove the p
        
        #smalldegs = p .* small.degrees[i,:]
        for j = 1:n
            smalldegs[j] = p * small.degrees[i,j]
        end

        #bigdegs = big.degrees[k,:]
        setslice_noalloc!(bigdegs,big.degrees,k)
        while smalldegs != bigdegs
            k = k + 1
            setslice_noalloc!(bigdegs,big.degrees,k)
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


"""
    raise_to_p_minus_1(hp::HomogeneousPolynomial, prime::Int, pregen::Delta1Pregen)

Raises homogeneous polynomial `hp` to the `prime - 1` power and returns a vector representing
the input into delta1, a HomogeneousPolynomial object representing hp ^ (p - 1), and a boolean
representing if hp is F-split
"""
function raise_to_p_minus_1(hp::HomogeneousPolynomial, prime::Int, pregen::Delta1Pregen)
    input1 = CuArray(kronecker_substitution(hp, pregen.key1, pregen.inputLen1))
    output1 = gpu_pow(input1, prime - 1; pregen = pregen.step1pregen)
    output1 = map(coeff -> faster_mod(coeff, prime), output1)

    intermediate = decode_kronecker_substitution(output1, pregen.key1, pregen.numVars, hp.homogeneousDegree * (prime - 1))

    return output1, intermediate, in_power_of_variable_ideal(prime, intermediate)
end

# don't actually know if this is right
function in_power_of_variable_ideal(m, hp)
    if all(hp.degrees .< m)
        return false
    end
    return true
end

# """
#     delta1(intermediate::HomogeneousPolynomial, prime::Int, pregen::Delta1Pregen, output1::CuVector = nothing)

# Computes Δ_1(intermediate, prime).

# Arguments:
# `intermediate`: Homogeneous Polynomial to take delta1 of (f^(p - 1))
# `output1`: Univariate polynomial isomorphic to intermediate
# """
# function delta1(intermediate::HomogeneousPolynomial, prime::Int, pregen::Delta1Pregen, output = nothing)
#     if output === nothing
#         input2 = CuArray(kronecker_substitution(intermediate, intermediate.homogeneousDegree * prime + 1, pregen.inputLen2))
#     else
#         input2 = change_polynomial_encoding(output, pregen)
#     end

#     output2 = gpu_pow(input2, prime; pregen = pregen.step2pregen)
#     result = decode_kronecker_substitution(output2, pregen.key2, pregen.numVars, pregen.key2 - 1)

#     cpu_remove_pth_power_terms!(result, intermediate, prime)

#     result.coeffs ./= prime
#     result.coeffs .%= prime

#     return result
# end

function delta1(intermediate::HomogeneousPolynomial, prime::Int, pregen::Delta1Pregen, output = nothing)
    if output === nothing
        input2 = CuArray(kronecker_substitution(intermediate, intermediate.homogeneousDegree * prime + 1, pregen.inputLen2))
    else
        input2 = change_polynomial_encoding(output, pregen)
    end

    output2 = gpu_pow(input2, prime; pregen = pregen.step2pregen)
    # gpu_remove_pth_power_terms!(output2, pregen.forgetIndices)
    result = decode_kronecker_substitution(output2, pregen.key2, pregen.numVars, pregen.key2 - 1)

    cpu_remove_pth_power_terms!(result,intermediate,prime)

    result.coeffs ./= prime
    result.coeffs .%= prime

    return result
end

function gpu_remove_pth_power_terms!(output2, forgetIndices)
    function gpu_remove_pth_power_terms_kernel!(output2, forgetIndices)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if idx <= length(forgetIndices)
            output2[forgetIndices[idx]] = 0
        end

        return
    end

    kernel = @cuda launch=false gpu_remove_pth_power_terms_kernel!(output2, forgetIndices)
    config = launch_configuration(kernel.fun)
    threads = min(length(forgetIndices), config.threads)
    blocks = cld(length(forgetIndices), threads)

    kernel(output2, forgetIndices; threads = threads, blocks = blocks)
    return
end

function new_process(hp::HomogeneousPolynomial, prime, pregen::Delta1Pregen)
    output1, intermediate, isfsplit = raise_to_p_minus_1(hp, prime, pregen)
    result = delta1(intermediate, prime, pregen, output1)
    result
end

function old_process(hp::HomogeneousPolynomial, prime, pregen::Delta1Pregen)
    @assert prime == pregen.prime && size(hp.degrees, 2) == pregen.numVars "Current pregen isn't compatible with input"
    
    # Raising f ^ p - 1
    input1 = CuArray(kronecker_substitution(hp, hp.homogeneousDegree * (prime - 1) + 1, pregen.inputLen1))
    output1 = gpu_pow(input1, prime - 1; pregen = pregen.step1pregen)

    # Reduce mod p
    output1 = map(num -> faster_mod(num, prime), output1)
    # Raising g ^ p
    input2 = CuArray(change_polynomial_encoding(output1, pregen))

    output2 = gpu_pow(input2, prime; pregen = pregen.step2pregen)
    result = decode_kronecker_substitution(output2, pregen.key2, pregen.numVars, pregen.key2 - 1)

    intermediate = decode_kronecker_substitution(output1, pregen.key1, pregen.numVars, pregen.key1 - 1)

    cpu_remove_pth_power_terms!(result,intermediate,prime)

    # # Here for debug purposes
    # for i in 1:length(result.coeffs)
    # # for i in 80250:80350
    #     if result.coeffs[i] % prime != 0
    #         println("WRONG COEFFICIENT $i: $(result.coeffs[i]), $(result.degrees[i, :])")
    #     end
    # end
    # if result.coeffs[10000] % prime != 0
    #     println("WRONG COEFFICIENT 10000: $(result.coeffs[10000]), $(result.degrees[10000, :])")
    # end

    result.coeffs ./= prime

    result.coeffs .%= prime

    #return (intermediate, result, finalresult
    result
end

function test_delta1(prime = 5, numTrials = 10)
    coeffs = [1, 2, 2, 2, 1]
    degrees = [
        4 0 0 0
        0 4 0 0
        0 0 4 0
        0 0 0 4
        1 1 2 0
    ]

    polynomial1 = HomogeneousPolynomial(coeffs, degrees)

    println("Time to pregenerate everything")
    CUDA.@time pregen = pregen_delta1(4, prime)

    println("Time to raise 4-variate polynomial to the 4th and 5th power for the first time: ")
    CUDA.@time old_result = old_process(polynomial1, prime, pregen)
    new_result = new_process(polynomial1, prime, pregen)
    @test Array(new_result.coeffs) == Array(old_result.coeffs)
    @test Array(new_result.degrees) == Array(old_result.degrees)

    println("Time to raise different 4-variate polynomials to the 4th and 5th power: ")
    for i in 1:numTrials
        polynomial2 = random_homogeneous_polynomial(4, rand(1:35), prime)
        println("Trial $i")
        CUDA.@time old_result2 = old_process(polynomial2, prime, pregen)
        new_result2 = new_process(polynomial2, prime, pregen)
        @test Array(new_result2.coeffs) == Array(old_result2.coeffs)
        @test Array(new_result2.degrees) == Array(old_result2.degrees)
    end
end
end