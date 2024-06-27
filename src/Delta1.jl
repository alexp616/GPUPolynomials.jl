include("gpu_ntt_pow.jl")
include("cpu_ntt_pow.jl")

mutable struct HomogeneousPolynomial
    coeffs::Array{Int, 1}
    degrees::Array{Int, 2}
    homogeneousDegree::Int
end

function HomogeneousPolynomial(coeffs::Array{Int, 1}, degrees::Array{Int, 2})
    @assert length(coeffs) == size(degrees, 1)
    return HomogeneousPolynomial(coeffs, degrees, sum(degrees[1, :]))
end

struct Delta1Pregen
    numVars::Int
    prime::Int
    cpuPrime::Int
    cpuNpru::Int
    cpuPregenButterfly::Array{Int, 1}
    gpuPrimeArray::Array{Int, 1}
    gpuNpruArray::Array{Int, 1}
    gpuPregenButterfly::CuArray{Int, 1}
    cpuInputLen::Int
    cpuOutputLen::Int
    gpuInputLen::Int
    gpuOutputLen::Int
    key1::Int
    key2::Int
end

function pregen_delta1(numVars::Int, prime::Int)
    # TODO very temporary
    @assert prime == 5 "I havent figured out upper bounds for other primes yet"
    @assert numVars == 4 "I haven't figured out upper bounds for 5 variables yet"
    
    # All of these as of now are prime numbers I hand-selected, 
    # I can hand-select primes for primes > 5 when the time comes. Though FFT might slow down a bit
    step1ResultDegree = numVars * (prime - 1)
    # len1 stores the size of the input into CPUNTT
    cpuInputLen = numVars * (step1ResultDegree + 1) ^ (numVars - 2) + 1
    # cpuOutputLen stores the size of the output of CPUNTT
    cpuOutputLen = nextpow(2, (cpuInputLen - 1) * (prime - 1) + 1)
    cpuPrime = 2654209
    cpuNpru = nth_principal_root_of_unity(cpuOutputLen, 2654209)
    cpuPregenButterfly = Array(generate_butterfly_permutations(cpuOutputLen))

    step2ResultDegree = step1ResultDegree * prime
    # len3 stores the size of the input into GPUNTT
    gpuInputLen = step1ResultDegree * (step2ResultDegree + 1) ^ (numVars - 2) + 1
    primearray2 = [330301441, 311427073]
    # gpuOutputLen stores the size of the output of GPUNTT
    gpuOutputLen = nextpow(2, (gpuInputLen - 1) * prime + 1)
    npruarray2 = npruarray_generator(primearray2, gpuOutputLen)
    gpu_pregenButterfly = generate_butterfly_permutations(gpuOutputLen)

    return Delta1Pregen(numVars, prime, cpuPrime, cpuNpru, cpuPregenButterfly, primearray2, npruarray2, gpu_pregenButterfly, cpuInputLen, cpuOutputLen, gpuInputLen, gpuOutputLen, step1ResultDegree + 1, step2ResultDegree + 1)
end

function kronecker_substitution(hp::HomogeneousPolynomial, pow::Int, pregen::Delta1Pregen)::Array{Int, 1}
    key = hp.homogeneousDegree * pow + 1

    result = zeros(Int, pregen.cpuInputLen)
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
    # that just index out of bounds for me if I do that
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

function change_encoding(num::Int, e1::Int, e2::Int, numValues::Int)
    result = 0
    for i in 1:numValues
        num, r = divrem(num, e1)
        result += r * e2 ^ (i - 1)
    end

    return result
end

function change_polynomial_encoding(p::Array{Int, 1}, pregen::Delta1Pregen)
    result = zeros(Int, pregen.gpuInputLen)
    for i in eachindex(p)
        if p[i] != 0
            result[change_encoding(i - 1, pregen.key1, pregen.key2, pregen.numVars - 1) + 1] = p[i]
        end
    end

    return result
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

Arguments:
big - HomogeneousPolynomial
small - HomogeneousPolynomial
p - power to remove terms from
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


function delta1(hp::HomogeneousPolynomial, prime, pregen::Delta1Pregen)
    @assert prime == pregen.prime && size(hp.degrees, 2) == pregen.numVars "Current pregen isn't compatible with input"

    # Raising f ^ p - 1
    cpuInput = kronecker_substitution(hp, prime - 1, pregen)
    cpuOutput = cpu_pow(cpuInput, prime - 1; prime = pregen.cpuPrime, npru = pregen.cpuNpru, len = pregen.cpuOutputLen, pregenButterfly = pregen.cpuPregenButterfly)

    # Reduce mod p
    cpuOutput .%= prime

    # println("CPU part took $(now() - start_time)")

    # Raising g ^ p
    gpuInput = CuArray(change_polynomial_encoding(cpuOutput, pregen))
    gpuOutput = GPUPow(gpuInput, prime; primearray = pregen.gpuPrimeArray, npruarray = pregen.gpuNpruArray, len = pregen.gpuOutputLen, pregenButterfly = pregen.gpuPregenButterfly)


    result = decode_kronecker_substitution(gpuOutput, pregen.key2, pregen.numVars, pregen.key2 - 1)

    println("Before subtracting at 2877: $(result.coeffs[2877]), $(result.degrees[2877,:])")
    
    # for now, do the rest of the steps on the cpu
    # TODO: uncomment this after fixing the bug
    
    #intermediate = decode_kronecker_substitution(CuArray(cpuOutput), pregen.key1, pregen.numVars, pregen.key1 - 1)
    #cpu_remove_pth_power_terms!(result,intermediate,prime)

    ##result.coeffs = divexact.(result.coeffs,5)

    #for i in 1:length(result.coeffs)
    #    i == 2877 && println("$i: $(result.coeffs[i]), $(result.degrees[i,:])")
    #    result.coeffs[i] = divexact(result.coeffs[i],5)
    #end
    #result.coeffs .%= prime

    #return (intermediate, result, finalresult
    result
end

function test_delta1()

    coeffs = [1, 2, 3, 4]
    degrees = [
        4 0 0 0
        0 4 0 0
        0 0 4 0
        0 0 0 4
    ]
    
    degrees2 = [4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]
    coeffs2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    polynomial = HomogeneousPolynomial(coeffs, degrees)
    polynomial2 = HomogeneousPolynomial(coeffs2, degrees2)
    
    pregen = pregen_delta1(4, 5)
    
    println("Time to raise 4-variate polynomial to the 4th and 5th power for the first time: ")
    CUDA.@time result = delta1(polynomial, 5, pregen)
    
    #println(length(result.coeffs))
    

    println("Time to raise different 4-variate polynomial to the 4th and 5th power: ")
    CUDA.@time result2 = delta1(polynomial2, 5, pregen)

    #println(length(result2.coeffs))

    return (result, result2)
end

function test_bug()

    coeffs = [4, 4, 4, 2, 4, 3, 1]
    degrees = [2 1 1 0; 0 3 1 0; 0 0 4 0; 1 1 1 1; 0 1 2 1; 0 2 0 2; 0 0 1 3]

    polynomial = HomogeneousPolynomial(coeffs, degrees)

    pregen = pregen_delta1(4,5)

    result = delta1(polynomial,5,pregen)

    println(result.coeffs[2877])
    println(result.degrees[2877,:]) # shouldb e [11, 29, 15, 25]

    @assert result.coeffs[2877] == 315105550

end
