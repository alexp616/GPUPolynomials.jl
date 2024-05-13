using Test
using BenchmarkTools
using CUDA



include("utils.jl")


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

    # result = result[setdiff(1:end, (pregen[3]+1:end)), :]
    # return result;
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

function sort_by_col!(arr, col)
    arr .= arr[sortperm(arr[:, col]), :]
end

polynomial = [[2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [5, [1, 1, 4, 2]],
               [4, [4, 1, 1, 2]], 
               [2, [2, 3, 1, 2]],
               [2, [1, 2, 3, 2]],
               [3, [3, 2, 1, 2]],
               [5, [1, 1, 4, 2]],
               [4, [4, 1, 1, 2]], 
]


println("Time to pregen (10 terms, 10 degree)")
pregen = pregenerate(10, 11)
@btime pregenerate(10, 11)

println("Time to compute power")
result = raise_to_mminus1_mod_m(polynomial, 11, pregen)
@btime raise_to_mminus1_mod_m(polynomial, 11, pregen)

# println("Time to sort result")
# @btime sort_by_col!(result)

# primes = [3, 5, 7, 11]
# open("hypertrianglebenchmarks.txt", "w") do io
#     redirect_stdout(io)
#     for i in 1:19
#         nterms = i
#         polynomial1 = polynomial[1:nterms]
#         for m in eachindex(primes)
#             println("Time to pregen ($nterms terms, $(m-1) degree)")
#             pregen = pregenerate(nterms, primes[m])
#             @btime pregenerate(nterms, primes[m])
    
#             prineln("Time to compute power")
#             @btime raise_to_mminus1_mod_m(polynomial1, primes[m], pregen)
#             CUDA.memory_status()
#             CUDA.reclaim()
#         end
#     end
# end



