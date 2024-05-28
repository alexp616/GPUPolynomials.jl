using CUDA
using Test

"""
METHODS ALL FUNCTIONS USE
"""

function sort_by_col!(arr, col)
    arr .= arr[sortperm(arr[:, col]), :]
end


"""
    generate_compositions(n, k)

Return all possible ways to distribute n identical balls into k distinct boxes.

No idea how to parallelize this, maybe dynamic parallelism?
"""
function generate_compositions(n, k, type::DataType = Int32)
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


"""
METHODS FOR polynomial ^ n
"""

"""
    pregenerate(num_terms, n)

Return tuple of all pre-computed information to avoid recomputation.
(cu_weakIntegerCompositions, multinomial, num_of_ending_terms)
"""
function pregenerate(num_terms, n)
    factorials = CuArray(generate_factorials(n))
    
    weakIntegerCompositions = generate_compositions(n, num_terms)

    nthreads = min(512, size(weakIntegerCompositions, 1))
    nblocks = cld(size(weakIntegerCompositions, 1), nthreads)

    # Pad to multiple of nthreads if needed
    num_paddedrows = cld(size(weakIntegerCompositions, 1), nthreads) * nthreads - size(weakIntegerCompositions, 1)
    cu_weakIntegerCompositions = CuArray(vcat(weakIntegerCompositions, fill(zero(Int32), (num_paddedrows, size(weakIntegerCompositions, 2)))))

    cu_multinomial_coeffs = CUDA.fill(zero(Int64), size(cu_weakIntegerCompositions, 1))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_multinomial_coeffs!(cu_weakIntegerCompositions, cu_multinomial_coeffs, n, num_terms, factorials)
    )

    return (cu_weakIntegerCompositions, cu_multinomial_coeffs, size(weakIntegerCompositions, 1))
end

"""
    generate_multinomial_coeffs(cu_weakIntegerCompositions, cu_multinomial_coeffs, n, num_terms, factorials)

Kernel function to put all multinomial coefficients in multinomial_coeffs
"""
function generate_multinomial_coeffs!(cu_weakIntegerCompositions, cu_multinomial_coeffs, n, num_terms, factorials)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    cu_multinomial_coeffs[idx] = factorials[n + 1]

    for j in 1:num_terms
        cu_multinomial_coeffs[idx] /= factorials[cu_weakIntegerCompositions[idx, j] + 1]
    end

    return
end

"""
    generate_factorials(m)

Return array of factorials up until m. Weirdly indexed to account for 0!:

generate_factorials(5)
[1, 1, 2, 6, 24, 120]

To get x!, factorials[x + 1]
"""
function generate_factorials(n)
    result = zeros(Int64, n + 1)
    result[1] = 1
    prod = 1

    for i in 2:n + 1
        prod *= i - 1
        result[i] = prod
    end

    return result
end

"""
METHODS FOR polynomial ^ p-1 mod p, WHERE p IS PRIME
"""

"""
    generate_inverses_mod_m(m)

Return array of multiplicative inverses of a mod m for 1 <= a < m

After assigning the return array to arr, simply do arr[a] for the inverse of a mod m
"""
function generate_inverses_mod_m(m)
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
@test generate_inverses_mod_m(7) == [1, 4, 5, 2, 3, 6]


"""
    generate_factorials_mod_m(m)

Return array of factorials mod m

factorials = generate_factorials_mod_m(m)
To get n! mod m, call factorials[n + 1]. Only works for n < m.
"""
function generate_factorials_mod_m(m)
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
    raise_n_to_p_mod_m(n, p, m)

Find n ^ p % m

Method is necessary because n ^ p may overflow.
"""
function raise_n_to_p_mod_m(n, p, m)
    result = 1
    for i in 1:p
        result *= n
        result = result % m
    end
    return Int32(result)
end

"""
    pregenerate_mod_m(num_terms, m)


"""
function pregenerate_mod_m(num_terms, m)
    factorials = CuArray(generate_factorials_mod_m(m))
    inverse = CuArray(generate_inverses_mod_m(m))
    weakIntegerCompositions = generate_compositions(m - 1, num_terms)

    nthreads = min(512, size(weakIntegerCompositions, 1))
    nblocks = cld(size(weakIntegerCompositions, 1), nthreads)

    num_paddedrows = cld(size(weakIntegerCompositions, 1), nthreads) * nthreads - size(weakIntegerCompositions, 1)
    cu_weakIntegerCompositions = CuArray(vcat(weakIntegerCompositions, fill(zero(Int32), (num_paddedrows, size(weakIntegerCompositions, 2)))))

    multinomial_coeffs = CUDA.fill(zero(Int32), size(cu_weakIntegerCompositions, 1))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_multinomial_coeffs_mod_m(cu_weakIntegerCompositions, multinomial_coeffs, num_terms, m, factorials, inverse)
    )

    return (cu_weakIntegerCompositions, multinomial_coeffs, size(weakIntegerCompositions, 1))
end

function generate_multinomial_coeffs_mod_m(cu_weakIntegerCompositions, multinomial_coeffs, num_terms, m, factorials, inverse)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    multinomial_coeffs[idx] = m - 1

    for j in 1:num_terms
        multinomial_coeffs[idx] *= inverse[factorials[cu_weakIntegerCompositions[idx, j] + 1]]
        multinomial_coeffs[idx] = multinomial_coeffs[idx] % m
    end

    return
end