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
    # primearray2 = [330301441, 311427073]
    primearray2 = [13631489, 23068673]
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
    return result

    # Haven't added other steps yet
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

    println("Time to raise different 4-variate polynomial to the 4th and 5th power: ")
    CUDA.@time result2 = delta1(polynomial2, 5, pregen)
    @assert all(x -> x > 0, result2.coeffs)
end