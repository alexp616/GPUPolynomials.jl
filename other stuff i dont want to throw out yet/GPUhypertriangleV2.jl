using Test
using BenchmarkTools
using CUDA

include("utilsV2.jl")


"""
    raise_to_n(p, m, pregen = nothing)

Return p^m, where p is a polynomial and m is an integer power. Will probably overflow when m gets big.

p is a polynomial represented as 2d array
Example: 2xyz + 3x^2yz^2 is represented as

    2 1 1 1
    3 2 1 2

Data structure subject to change

(I will never use this)
(Will probably error too)
"""
function raise_to_n(p, n, pregen = nothing)
    num_vars = size(p, 2) - 1
    num_terms = size(p, 1)

    if pregen === nothing
        pregen = pregenerate(num_terms, n)
    end

    # My laptop explodes when I try to do the following:
    # nthreads = min(CUDA.attribute(
    #     device(),
    #     CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    # ), size(pregen[1]))

    nthreads = min(512, size(pregen[1], 1))
    nblocks = cld(size(pregen[1], 1), nthreads)

    cu_p = CuArray(p)

    result = CUDA.fill(zero(Float64), size(pregen[1], 1), 1 + num_vars)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        raise_to_m_kernel!(cu_p, result, pregen[1], pregen[2], num_vars, num_terms)
    )

    return view(result, 1:pregen[3], :)
end

function raise_to_m_kernel!(cu_p, result, cu_termPowers, multinomial_coeffs, num_vars, num_terms)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    result[idx, 1] = multinomial_coeffs[idx]

    for j in 1:num_terms
        result[idx, 1] *= cu_p[j, 1] ^ cu_termPowers[idx, j]
        for k in 2:num_vars + 1
            result[idx, k] += cu_termPowers[idx, j] * cu_p[j, k]
        end
    end

    return
end


"""
    raise_to_mminus1_mod_m(p, m)

Return p^(m-1) mod m, where p is a polynomial and m is a prime integer.
"""
function raise_to_mminus1_mod_m(p, m, pregen = nothing)
    num_vars = size(p, 2) - 1
    num_terms = size(p, 1)

    if pregen === nothing
        pregen = pregenerate_mod_m(num_terms, m)
    end

    # pregen is a tuple of (termPowers, multinomial_coeffs, num_of_ending_terms)
    # termPowers is all possible combinations of powers of the terms of the polynomial
    # if the polynomial has 3 terms to be raised to the 4th then termPowers contains
    # [4, 0, 0], [0, 4, 0], [0, 0, 4], [3, 1, 0] ... etc
    nthreads = min(512, size(pregen[1], 1))
    nblocks = cld(size(pregen[1], 1), nthreads)

    cu_p = CuArray(p)

    result = CUDA.fill(zero(Int32), size(pregen[1], 1), 1 + num_vars)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        raise_to_mminus1_mod_m_kernel!(cu_p, m, result, pregen[1], pregen[2], num_vars, num_terms)
    )

    # how in the world do you efficiently just remove rows
    # result = result[setdiff(1:end, (pregen[3]+1:end)), :]
    # return result;
    return view(result, 1:pregen[3], :)
end

function raise_to_mminus1_mod_m_kernel!(cu_p, m, result, cu_termPowers, multinomial_coeffs, num_vars, num_terms)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    result[idx, 1] = 1
    result[idx, 1] *= multinomial_coeffs[idx]
    for j in 1:num_terms
        result[idx, 1] *= raise_n_to_p_mod_m(cu_p[j, 1], cu_termPowers[idx, j], m)
        result[idx, 1] = result[idx, 1] % m
        # # CuArrays don't like p:q indexing I think
        for k in 2:num_vars + 1
            result[idx, k] += cu_termPowers[idx, j] * cu_p[j, k]
        end
    end

    return
end


"""
    raise_to_n_mod_m(p, n, m, pregen = nothing)

Return p ^ n mod m, where p is a polynomial, n is an integer, and m is a prime integer
"""
function raise_to_n_mod_m(p, n, m, pregen = nothing)
    num_vars = size(p, 2) - 1
    num_terms = size(p, 1)

    if pregen === nothing
        pregen = pregenerate_to_n_mod_m(num_terms, n, m)
    end
end

"""
    pregenerate_to_n_mod_m(num_terms, n, m)


"""
function pregenerate_to_n_mod_m(num_terms, n, m)
    factorials = CuArray(generate_n_factorials_mod_m)
    # Wouldn't most of these not be invertible mod m? So then how would division work?
    inverse = nothing
end


polynomial = [ 
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
    1 1 1 1 1
    2 4 0 0 0
    3 2 2 0 0
    1 1 1 2 0
    4 1 0 1 2
]

println("Time to pregen (35 terms, 4 degree)")
pregen = pregenerate_mod_m(35, 5)
@btime pregenerate_mod_m(35, 5)

println("Time to compute power")
result = raise_to_mminus1_mod_m(polynomial, 5, pregen)
@btime raise_to_mminus1_mod_m(polynomial, 5, pregen)





