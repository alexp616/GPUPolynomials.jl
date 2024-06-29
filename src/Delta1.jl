include("gpu_ntt_pow.jl")

"""
    HomogeneousPolynomial

Struct that represents a homogeneous polynomial. Does nothing to check that the terms
are actually homogeneous.

# Arguments
- `coeffs`: Array of coefficients for each term of the polynomial
- `degrees`: 2d array, where each row represents the degrees of the variables of the term
- `homogeneousDegree`: The homogeneous degree of the polynomial
"""
mutable struct HomogeneousPolynomial
    coeffs::Vector{Int}
    degrees::Array{Int, 2}
    homogeneousDegree::Int
end

"""
    HomogeneousPolynomial(coeffs, degrees)

Lazier for creating HomogeneousPolynomial object
"""
function HomogeneousPolynomial(coeffs::Vector{Int}, degrees::Array{Int, 2})
    @assert length(coeffs) == size(degrees, 1)
    return HomogeneousPolynomial(coeffs, degrees, sum(degrees[1, :]))
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
    for termidx in eachindex(hp.coeffs)
        # encoded = 1 because julia 1-indexing
        encoded = 1
        for d in 1:size(hp.degrees, 2) - 1
            encoded += hp.degrees[termidx, d] * key ^ (d - 1)
        end

        result[encoded] = hp.coeffs[termidx]
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

    function decode_kronecker_kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree, offset = 0)
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
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
        
        return
    end

    nthreads = min(512, length(arr))
    nblocks = fld(length(arr), nthreads)

    last_block_threads = length(arr) - nthreads * nblocks

    # I have this last_block_threads thing because it ran slightly faster than
    # padding to the next multiple of nthreads, might not be the same on other machines
    # Also, the gpu kernels I see in CUDA.jl have some if tid < length(arr) ... stuff,
    # that just index out of bounds for me if I do that, so I don't know how they coded those
    if last_block_threads > 0
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            decode_kronecker_kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        decode_kronecker_kernel(resultCoeffs, resultDegrees, arr, flags, indices, key, numVars, totalDegree)
    )
    
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

    nthreads = min(512, length(p))
    nblocks = fld(length(p), nthreads)
    lastBlockThreads = length(p) - nthreads * nblocks

    if lastBlockThreads > 0
        @cuda(
            threads = lastBlockThreads,
            blocks = 1,
            change_polynomial_encoding_kernel(p, result, pregen.key1, pregen.key2, pregen.numVars, nthreads * nblocks)
        )
    end

    @cuda(
        threads = nthreads,
        blocks = nblocks,
        change_polynomial_encoding_kernel(p, result, pregen.key1, pregen.key2, pregen.numVars)
    )

    return result
end

"""
    change_polynomial_encoding_kernel()

Kernel function for change_polynomial_encoding()
"""
function change_polynomial_encoding_kernel(source, dest, key1, key2, numVars, offset = 0)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    resultidx = change_encoding(tid - 1, key1, key2, numVars - 1) + 1
    dest[resultidx] = source[tid]

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

Escaped hypertriangle code because still useful for generating starting degrees
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
        # if result.coeffs[i] % 5 != 0
        #     println("WRONG COEFFICIENT $i: $(result.coeffs[i]), $(result.degrees[i, :])")
        # end
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

    degrees2 = [4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]
    coeffs2 = fill(4, size(degrees2, 1))

    polynomial = HomogeneousPolynomial(coeffs, degrees)
    polynomial2 = HomogeneousPolynomial(coeffs2, degrees2)

    pregen = pregen_delta1(4, 5)

    println("Time to raise 4-variate polynomial to the 4th and 5th power for the first time: ")
    CUDA.@time result = delta1(polynomial, 5, pregen)

    println("Time to raise different 4-variate polynomial to the 6th and 7th power: ")
    for i in 1:10
        println("Trial $i")
        CUDA.@time result2 = delta1(polynomial2, 5, pregen)
    end
end
    
function test_bug()

    coeffs = [4, 4, 4, 2, 4, 3, 1]
    degrees = [2 1 1 0; 0 3 1 0; 0 0 4 0; 1 1 1 1; 0 1 2 1; 0 2 0 2; 0 0 1 3]

    polynomial = HomogeneousPolynomial(coeffs, degrees)

    pregen = pregen_delta1(4,5)

    result = delta1(polynomial,5,pregen)

    println(result.coeffs[2877])
    println(result.degrees[2877,:]) # should be [11, 29, 15, 25]

    # @assert result.coeffs[2877] == 315105550
    # @assert result.coeffs[1040] == 24352765

end

