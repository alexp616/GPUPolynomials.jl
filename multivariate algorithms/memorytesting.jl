using Test
using BenchmarkTools
using CUDA


"""
    compute_inverses_mod_m(m)

Return array of multiplicative inverses of a mod m for 1 <= a < m

After assigning the return array to arr, simply do arr[a] for the inverse of a mod m
"""
function compute_inverses_mod_m(m)
    result = zeros(Int, m - 1)
    for i in 1:m-1
        if result[i] == 0
            for j in 1:m-1
                if (i * j) % m == 1
                    result[i] = j
                    result[j] = i
                end
            end
        end
    end
    return result
end
@test compute_inverses_mod_m(7) == [1, 4, 5, 2, 3, 6]

function compute_factorials_mod_m(m)
    result = zeros(Int32, m)
    result[1] = 1
    prod = 1
    for i in 2:m
        prod *= i - 1
        prod = prod % m
        result[i] = prod
    end
    
    return result
end

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

function raise_n_to_p_mod_m(n, p, m)
    result = 1
    for i in 1:p
        result *= n
        result = result % m
    end
    return Int32(result)
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
        multinomial_coeffs[idx] *= 1
        multinomial_coeffs[idx] = multinomial_coeffs[idx] % m
    end

    return
end

degree = 5
terms = 20

CUDA.memory_status()
println(binomial(degree + terms - 1, terms - 1))
println("expected memory: $(4 * binomial(terms + degree - 1, terms - 1) * (terms + 1)) bytes")
pregen = pregenerate(terms, degree + 1)
CUDA.memory_status()
# for terms in 1:5
#     for degree in 1:5
#         println("memory after terms=$terms, degree=$degree ($(binomial(terms + degree - 1, terms - 1)) partitions of length $terms):")
#         println("$(4 * binomial(terms + degree - 1, terms - 1) * (terms + 1))")
#         println()
#         pregen = pregenerate(terms, degree + 1)
#         CUDA.memory_status()
#         CUDA.unsafe_free!(pregen[1])
#         CUDA.unsafe_free!(pregen[2])
#         println()
#     end
# end