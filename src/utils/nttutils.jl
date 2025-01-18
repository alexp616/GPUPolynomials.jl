include("getinttype.jl")
include("modoperations.jl")
include("modsqrt.jl")

function intlog2(x::Int64)
    return 64 - leading_zeros(x - 1)
end

function intlog2(x::Int32)::Int32
    return Int32(32) - leading_zeros(x - Int32(1))
end

function is_primitive_root(npru::T, p::T, order::Integer) where T<:Integer
    temp = npru
    for i in 1:order - 1
        if temp == 1
            return false
        end

        temp = mul_mod(temp, npru, p)
    end

    return temp == 1
end

"""
    primitive_nth_root_of_unity(n::Integer, p::Integer)

Return a primitive n-th root of unity of the field 𝔽ₚ
"""
function primitive_nth_root_of_unity(n::Integer, p::Integer)
    @assert ispow2(n)
    if (p - 1) % n != 0
        throw("n must divide p - 1")
    end

    g = p - typeof(p)(1)

    a = intlog2(n)

    while a > 1
        a -= 1
        original = g
        g = modsqrt(g, p)
        @assert powermod(g, 2, p) == original
    end

    @assert is_primitive_root(g, p, n)
    return g
end

"""
    generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer

Returns array containing powers 0 -> n-1 of npru mod p. Accessed as:
arr[i] = npru ^ (i - 1)
"""
function generate_twiddle_factors(npru::T, p::T, n::Int) where T<:Integer
    @assert is_primitive_root(npru, p, n)

    result = zeros(T, n)
    curr = T(1)
    for i in eachindex(result)
        result[i] = curr
        curr = mul_mod(curr, npru, p)
    end

    return result
end

function find_ntt_primes(len::Int, T = UInt32, num = 10)
    prime_list = []
    k = fld(typemax(T), len)
    while length(prime_list) < num
        candidate = k * len + 1
        if isprime(candidate)
            push!(prime_list, candidate)
        end
        k -= 1
    end

    return prime_list
end

function sparsify(dense::Array)
    resultLen = 0
    zerovec = zeros(eltype(dense), size(dense, 2))
    for i in axes(dense, 1)
        if view(dense, i, :) != zerovec
            resultLen += 1
        end
    end

    resultCoeffs = zeros(eltype(dense), resultLen, size(dense, 2))
    resultDegrees = zeros(Int64, resultLen)

    curridx = 1
    for i in axes(dense, 1)
        if dense[i, :] != zerovec
            resultCoeffs[curridx, :] .= dense[i, :]
            resultDegrees[curridx] = i
            curridx += 1
        end
    end

    return CuArray(resultCoeffs), CuArray(resultDegrees)
end

function sparsify(dense::CuArray)
    flags = CUDA.zeros(Int32, size(dense, 1))

    kernel1 = @cuda launch=false generate_flags_kernel!(dense, flags)
    config = launch_configuration(kernel1.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel1(dense, flags; threads = threads, blocks = blocks)

    indices = accumulate(+, flags)

    CUDA.@allowscalar resultLen = indices[end]

    resultCoeffs = CUDA.zeros(eltype(dense), resultLen, size(dense, 2))
    resultDegrees = CUDA.zeros(Int64, resultLen)

    kernel2 = @cuda launch=false generate_result_kernel!(dense, flags, indices, resultCoeffs, resultDegrees)
    config = launch_configuration(kernel2.fun)
    threads = min(length(flags), config.threads)
    blocks = cld(length(flags), threads)

    kernel2(dense, flags, indices, resultCoeffs, resultDegrees; threads = threads, blocks = blocks)

    return resultCoeffs, resultDegrees
end

function generate_flags_kernel!(dense, flags)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(flags)
        for i in axes(dense, 2)
            if dense[idx, i] != zero(eltype(dense))
                flags[idx] = one(Int32)
            end
        end
    end

    return nothing
end

function generate_result_kernel!(dense, flags, indices, resultCoeffs, resultDegrees)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(flags)
        if flags[idx] != 0
            termNum = indices[idx]
            resultDegrees[termNum] = idx
            for i in axes(dense, 2)
                resultCoeffs[termNum, i] = dense[idx, i]
            end
        end
    end

    return nothing
end

function build_result(multimodvec, crtpregen, cpureturn = false)
    # result = CUDA.zeros(eltype(crtpregen), finalLength)
    result = CuArray(zeros(eltype(crtpregen), size(multimodvec, 1)))

    # zerovec = CUDA.zeros(eltype(multimodvec), size(multimodvec, 2))
    kernel = @cuda launch=false build_result_kernel!(multimodvec, crtpregen, result)
    config = launch_configuration(kernel.fun)
    threads = min(length(result), config.threads)
    blocks = cld(length(result), threads)

    kernel(multimodvec, crtpregen, result; threads = threads, blocks = blocks)

    if cpureturn
        cpuresult = Array(result)
        CUDA.unsafe_free!(result)
        return cpuresult
    else
	    return result
    end
end

function build_result_kernel!(multimodvec, crtpregen, result)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx <= length(result)
        @inbounds result[idx] = crt(view(multimodvec, idx, :), crtpregen)
    end

    return nothing
end

function crt(vec, pregen)
    x = eltype(pregen)(vec[1])
    # @cuprintln(x)
    for i in axes(pregen, 2)
        a = mul_mod(x, pregen[2, i], pregen[3, i])
        b = mul_mod(eltype(pregen)(vec[i + 1]), pregen[1, i], pregen[3, i])
        x = add_mod(a, b, pregen[3, i])
        # @cuprintln(x)
    end

    return x
end

function plan_crt(primeArray::Vector{T}) where T<:Integer
    # There really shouldn't be any overflow behavior, but 
    # I'm doing it in BigInt just to be safe. This is all for
    # the pregeneration step anyways.
    primeArray = BigInt.(primeArray)

    result = zeros(BigInt, 3, length(primeArray) - 1)

    currmod = primeArray[1]
    for i in 2:length(primeArray)
        m1, m2 = extended_gcd_iterative(currmod, primeArray[i])
        result[1, i - 1] = m1 * currmod
        currmod *= primeArray[i]
        result[2, i - 1] = m2 * primeArray[i]
        result[3, i - 1] = currmod
        if result[1, i - 1] < 0
            result[1, i - 1] += currmod
        end
        if result[2, i - 1] < 0
            result[2, i - 1] += currmod
        end
    end

    @assert all([i > 0 for i in result]) display(result)
    return T.(result)
end

function decode_kronecker_substitution(encodedDegs, key, numVars, totalDegree, type)
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
        for i in numVars:-1:2
            num, r = divrem(num, key)
            dest[i, idx] = r
            totalDegree -= r
        end
        dest[1, idx] = totalDegree
    end

    return nothing
end