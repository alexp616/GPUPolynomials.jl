import Base: convert, ==

# Fields of FqMPolyRingElem: 
# - parent
#     - data
#         - nvars (obvious)
# - data
#     - coeffs (Ptr{nothing}) for coeffs
#     - exps (Ptr{nothing}) for exps
#     - length (number of terms)
#     - bits (number of bits each element of a degree vector takes up)
#     - coeffs_alloc (number of machine words taken up by all coeffs)
#     - exps_alloc (number of machine words taken up by all exps)

mutable struct HomogeneousPolynomial
    poly::MPolyRingElem
    homogDegree::Int

    function HomogeneousPolynomial(poly::MPolyRingElem)
        new(poly, get_homog_degree(poly))
    end

    function HomogeneousPolynomial(poly::MPolyRingElem, homogDegree::Int)
        new(poly, homogDegree)
    end
end

# Not meant for use internally, hence lazy implementation
function exps_matrix(hp::HomogeneousPolynomial)
    return exps_matrix(hp.poly)
end

function exps_matrix(p::MPolyRingElem)
    expvecs = leading_exponent_vector.(terms(p))
    expmat = reduce(hcat, expvecs)
    return expmat
end

function get_coeffs(hp::HomogeneousPolynomial)
    return get_coeffs(hp.poly)
end

function get_exps(hp::HomogeneousPolynomial)
    return get_exps(hp.poly)
end

function get_homog_degree(poly::MPolyRingElem)::Int
    exps = get_exps(poly)
    num = exps[1]
    deg = 0
    mask = (one(eltype(exps)) << poly.data.bits) - 1
    for i in 1:poly.parent.data.nvars
        deg += num & mask
        num >>= poly.data.bits
    end

    return deg
end

function nvars(hp::HomogeneousPolynomial)::Int
    return length(gens(hp.poly.parent))
end

function Base.convert(::Type{FqMPolyRingElem}, hp::HomogeneousPolynomial)
    degs = get_exps(hp)
    if issorted(degs, rev = true)
        return hp.poly
    else
        coeffs = get_coeffs(hp)
        perm = sortperm(degs, rev = true)
        coeffs .= coeffs[perm]
        degs .= degs[perm]
        return hp.poly
    end
end

function Base.convert(::Type{HomogeneousPolynomial}, poly::FqMPolyRingElem)
    return HomogeneousPolynomial(poly)
end

function get_dense_representation(hp::HomogeneousPolynomial, length, bits, type, key)
    result = zeros(type, length)
    coeffs = get_coeffs(hp)
    exps = get_exps(hp)
    mask = (one(eltype(exps)) << bits) - 1
    keyPowers = [key ^ i for i in 0:nvars(hp) - 2]
    for termIdx in eachindex(exps)
        resultIdx = 1
        deg = exps[termIdx]
        for i in 1:nvars(hp) - 1
            resultIdx += (deg & mask) * keyPowers[i]
            deg >>= bits
        end

        result[resultIdx] = coeffs[termIdx]
    end
    
    return result
end

function kron_to_bitpacked(idx, numVars, key, totalDegree, bits)
    num = idx - 1
    resultexp = zero(UInt64)
    for i in 0:numVars - 2
        num, r = divrem(num, key)
        resultexp += r << (bits * i)
        totalDegree -= r
    end

    resultexp += totalDegree << (bits * (numVars - 1))

    return resultexp
end

function get_bound(n, m, pow)
    n = BigInt(n)
    m = BigInt(m)
    return (m - 1) ^ pow * binomial(n * m + n - 1, n - 1) ^ pow
end

function get_fft_size(homogDegree, n, pow)
    resultTotalDegree = pow * homogDegree
    key = resultTotalDegree + 1
    finalLength = resultTotalDegree * key ^ (n - 2) + 1

    return Base._nextpow2(finalLength)
end

function pregen_gpu_pow(hp::HomogeneousPolynomial, pow::Int)
    @assert hp.poly isa FqMPolyRingElem "Haven't implemented bound finding for ZZPolyRingElem yet"
    bound = get_bound(nvars(hp), Int(hp.poly.parent.data.n), pow)
    fftSize = get_fft_size(hp.homogDegree, nvars(hp), pow)

    possiblePrimes = find_ntt_primes(fftSize)
    primeArray = UInt[]
    currTotal = BigInt(1)
    idx = 1
    while currTotal < bound
        prime = possiblePrimes[idx]
        idx += 1
        currTotal *= prime
        push!(primeArray, prime)
    end

    return pregen_gpu_pow(primeArray, fftSize)
end

function gpu_pow(hp::HomogeneousPolynomial, pow::Int, pregen::GPUPowPregen{T}) where T<:Integer
    resultDegree = hp.homogDegree * pow
    key = (resultDegree) + 1
    len = hp.homogDegree * (key) ^ (nvars(hp) - 2) + 1

    if pregen === nothing
        throw(ArgumentError("Default pregeneration has not been implemented yet."))
    end

    if pregen.nttpregen.butterfly isa Vector
        vect = get_dense_representation(hp, len, hp.poly.data.bits, T, key)
        denseMultimodCoeffs = memorysafe_gpu_ntt_pow(vect, pow; pregen = pregen)
        sparseMultimodCoeffs, encodedDegs = sparsify(denseMultimodCoeffs)
        resultCoefs = build_result(sparseMultimodCoeffs, pregen.crtpregen, pregen.resultType, true)
        if hp.poly isa FqMPolyRingElem
            resultCoefs .%= hp.poly.parent.data.n
        end
        resultCoefs = UInt64.(resultCoefs)
        resultDegs = decode_kronecker_substitution(encodedDegs, key, nvars(hp), resultDegree)

        result = zero(hp.poly.parent)
        result.data = Oscar.fpMPolyRingElem(hp.poly.parent.data, resultCoefs, resultDegs)
    else
        vect = CuArray(get_dense_representation(hp, len, hp.poly.data.bits, T, key))
        denseMultimodCoeffs = gpu_ntt_pow(vect, pow; pregen = pregen, docrt = false)
        sparseMultimodCoeffs, encodedDegs = sparsify(denseMultimodCoeffs)
        resultCoefs = build_result(sparseMultimodCoeffs, pregen.crtpregen, pregen.resultType)
        resultCoefs = Array(resultCoefs)
        if hp.poly isa FqMPolyRingElem
            resultCoefs .%= hp.poly.parent.data.n
        end
        resultCoefs = UInt64.(resultCoefs)
        resultDegs = decode_kronecker_substitution(encodedDegs, key, nvars(hp), resultDegree)
        
        result = zero(hp.poly.parent)
        result.data = Oscar.fpMPolyRingElem(hp.poly.parent.data, resultCoefs, resultDegs)
    end

    return HomogeneousPolynomial(result, resultDegree)
end

function decode_kronecker_substitution(encodedDegs, key, numVars, totalDegree)
    result = CUDA.zeros(UInt64, numVars, length(encodedDegs))

    kernel = @cuda launch=false decode_kronecker_substitution_kernel!(encodedDegs, key, numVars, totalDegree, result)
    config = launch_configuration(kernel.fun)
    threads = min(length(encodedDegs), config.threads)
    blocks = cld(length(encodedDegs), threads)

    kernel(encodedDegs, key, numVars, totalDegree, result; threads = threads, blocks = blocks)

    return Array(result)
end

function decode_kronecker_substitution_kernel!(encodedDegs::CuDeviceVector, key::Int, numVars::Int, totalDegree::Int, dest::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(encodedDegs)
        num = encodedDegs[idx] - 1
        for i in 1:numVars - 1
            num, r = divrem(num, key)
            dest[i, idx] = r
            totalDegree -= r
        end
        dest[numVars, idx] = totalDegree
    end

    return nothing
end