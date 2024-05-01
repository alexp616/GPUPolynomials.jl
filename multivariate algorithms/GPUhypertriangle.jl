using Test
using BenchmarkTools
using CUDA

include("utils.jl")

"""
    get_num_variables(p)

Return number of variables in polynomial represented by p
"""
function get_num_variables(p)
    return length(p[1][2])
end

"""
    polynomial_to_arr(p)

Convert p into 2d array
"""
function polynomial_to_arr(p)
    num_cols = get_num_variables(p) + 1
    result = zeros(Int, length(p), num_cols)
    for i in eachindex(p)
        result[i, 1] = p[i][1]
        result[i, 2:num_cols] .= p[i][2]
    end

    return result
end



"""
    raise_to_mminus1_mod_m(p, m)

Return p^(m-1) mod m
"""
function raise_to_mminus1_mod_m(p, m, pregen = nothing)
    # termPowers is all possible combinations of powers of the terms of the polynomial
    # if the polynomial has 3 terms to be raised to the 4th then termPowers contains
    # [4, 0, 0], [0, 4, 0], [0, 0, 4], [3, 1, 0] ... etc
    num_vars = get_num_variables(p)
    num_terms = length(p)

    if pregen === nothing
        pregen = pregenerate(num_terms, m)
    end

    # pregen returns (cu_termPowers, multinomial_coeffs, num_of_ending_terms)
    nthreads = min(512, size(pregen[1], 1))
    nblocks = cld(size(pregen[1], 1), nthreads)

    cu_p = CuArray(polynomial_to_arr(p))

    result = CUDA.fill(zero(Int32), size(pregen[1], 1), 1 + num_vars)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        power_kernel!(cu_p, m, result, pregen[1], pregen[2], num_vars, num_terms)
    )

    return view(result, 1:pregen[3], :)
end


"""
kernel for raise to power thing
"""
function power_kernel!(cu_p, m, result, cu_termPowers, multinomial_coeffs, num_vars, num_terms)
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
    generate_compositions(n, k)

Return all possible ways to distribute n identical balls into k distinct boxes.

No idea how to parallelize this, maybe dynamic parallelism?
"""
function generate_compositions(n, k)
    compositions = zeros(Int32, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(Int32, k)
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
    generate_termPowers(p, m)

Pre-generate resulting representations of p^m-1 mod m
"""
function generate_termPowers(p, m)
    return generate_compositions(m - 1, length(p))
end

function pregenerate(num_terms, m)
    factorials = CuArray(compute_factorials_mod_m(m))
    inverse = CuArray(compute_inverses_mod_m(m))
    termPowers = generate_compositions(m - 1, num_terms)

    nthreads = min(512, size(termPowers, 1))
    nblocks = cld(size(termPowers, 1), nthreads)

    num_paddedrows = cld(size(termPowers, 1), nthreads) * nthreads - size(termPowers, 1)
    cu_termPowers = CuArray(vcat(termPowers, fill(zero(Int32), (num_paddedrows, size(termPowers, 2)))))

    multinomial_coeffs = CUDA.fill(zero(Int32), size(cu_termPowers, 1))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_multinomial_coeffs(cu_termPowers, multinomial_coeffs, num_terms, m, factorials, inverse)
    )

    CUDA.unsafe_free!(factorials)
    CUDA.unsafe_free!(inverse)
    return (cu_termPowers, multinomial_coeffs, size(termPowers, 1))
end

function generate_multinomial_coeffs(cu_termPowers, multinomial_coeffs, num_terms, m, factorials, inverse)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    multinomial_coeffs[idx] = m - 1

    for j in 1:num_terms
        multinomial_coeffs[idx] *= inverse[factorials[cu_termPowers[idx, j] + 1]]
        multinomial_coeffs[idx] = multinomial_coeffs[idx] % m
    end

    return
end

polynomial1 = [[2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [5, [1, 1, 4, 2]],
               [4, [4, 1, 1, 2]], 
               [2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [5, [1, 1, 4, 2]],
               [4, [4, 1, 1, 2]], 
               [2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [5, [1, 1, 4, 2]],
               [4, [4, 1, 1, 2]], 
               [2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [4, [4, 1, 1, 2]],]

nterms = length(polynomial1)
m = 5

CUDA.memory_status()
# println("Time to pre-generate compositions & coefficients:")
pregen = pregenerate(nterms, m)
# @btime pregenerate(nterms, m)

mem = 4 * (binomial(m + nterms - 2, nterms - 1) * (nterms + 1))
println("Expected memory gain: $mem bytes")

CUDA.memory_status()
# raise_to_mminus1_mod_m(polynomial1, m, pregen)

# CUDA.memory_status()

# mem = 4 * (binomial(m + nterms - 2, nterms - 1) * nterms + 2 + get_num_variables(polynomial1)) + 4 * (nterms * (get_num_variables(polynomial1)))
# println("Expected memory gain: $mem bytes")
# @btime raise_to_mminus1_mod_m(polynomial1, m, pregen)