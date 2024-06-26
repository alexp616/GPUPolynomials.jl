include("gpu_ntt_pow.jl")
include("cpu_ntt_pow.jl")

mutable struct HomogeneousPolynomial
    coeffs::Vector{Int}
    degrees::Array{Int, 2}
    homogeneousDegree::Int
end

function HomogeneousPolynomial(coeffs::Vector{Int}, degrees::Array{Int, 2})
    @assert length(coeffs) == size(degrees, 1)
    return HomogeneousPolynomial(coeffs, degrees, sum(degrees[1, :]))
end

struct Delta1Pregen
    numVars::Int
    prime::Int
    primeArray1::Vector{Int}
    npruArray1::Vector{Int}
    pregenButterfly1::CuVector{Int}
    primeArray2::Vector{Int}
    npruArray2::Vector{Int}
    pregenButterfly2::CuVector{Int}
    inputLen1::Int
    fftSize1::Int
    inputLen2::Int
    fftSize2::Int
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
    inputLen1 = numVars * (step1ResultDegree + 1) ^ (numVars - 2) + 1
    # fftSize1 stores the size of the output of CPUNTT
    fftSize1 = nextpow(2, (inputLen1 - 1) * (prime - 1) + 1)
    primeArray1 = [2654209]
    npruArray1 = npruarray_generator(primeArray1, fftSize1)
    pregenButterfly1 = generate_butterfly_permutations(fftSize1)

    step2ResultDegree = step1ResultDegree * prime
    # len3 stores the size of the input into GPUNTT
    inputLen2 = step1ResultDegree * (step2ResultDegree + 1) ^ (numVars - 2) + 1
    # primeArray2 = [330301441, 311427073]
    primeArray2 = [13631489, 23068673]
    # fftSize2 stores the size of the output of GPUNTT
    fftSize2 = nextpow(2, (inputLen2 - 1) * prime + 1)
    npruArray2 = npruarray_generator(primeArray2, fftSize2)
    pregenButterfly2 = generate_butterfly_permutations(fftSize2)

    return Delta1Pregen(numVars, prime, primeArray1, npruArray1, pregenButterfly1, primeArray2, npruArray2, pregenButterfly2, inputLen1, fftSize1, inputLen2, fftSize2, step1ResultDegree + 1, step2ResultDegree + 1)
end


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

function change_polynomial_encoding_kernel(source, dest, key1, key2, numVars, offset = 0)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    resultidx = change_encoding(tid - 1, key1, key2, numVars - 1) + 1
    dest[resultidx] = source[tid]

    return
end

function delta1(hp::HomogeneousPolynomial, prime, pregen::Delta1Pregen)
    @assert prime == pregen.prime && size(hp.degrees, 2) == pregen.numVars "Current pregen isn't compatible with input"

    # Raising f ^ p - 1
    input1 = CuArray(kronecker_substitution(hp, prime - 1, pregen))
    output1 = gpu_pow(input1, prime - 1; primearray = pregen.primeArray1, npruarray = pregen.npruArray1, len = pregen.fftSize1, pregenButterfly = pregen.pregenButterfly1)

    # Reduce mod p
    output1 .%= prime

    # Raising g ^ p
    input2 = CuArray(change_polynomial_encoding(output1, pregen))

    output2 = gpu_pow(input2, prime; primearray = pregen.primeArray2, npruarray = pregen.npruArray2, len = pregen.fftSize2, pregenButterfly = pregen.pregenButterfly2)

    result = decode_kronecker_substitution(output2, pregen.key2, pregen.numVars, pregen.key2 - 1)
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
    @benchmark result2 = delta1(polynomial2, 5, pregen)
end