using Test
using BenchmarkTools
using CUDA

"""
    factorial_mod_m(n, m)

Calculate n! mod m
"""
function factorial_mod_m(n, m)
    result = 1
    for i in 2:n
        result *= i
        result %= m
    end
    return result
end
@test factorial_mod_m(6, 7) == 6


"""
    compute_inverses_mod_m(m)

Return array of multiplicative inverses of a mod m for 1 <= a < m

After assigning the return array to arr, simply do arr[a] for the inverse of a mod m
"""
function compute_inverses_mod_m(m)
    result = zeros(Int, m-1)
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
function raise_to_mminus1_mod_m(p, m, termPowers = nothing)
    # termPowers is all possible combinations of powers of the terms of the polynomial
    # if the polynomial has 3 terms to be raised to the 4th then termPowers contains
    # [4, 0, 0], [0, 4, 0], [0, 0, 4], [3, 1, 0] ... etc
    num_vars = get_num_variables(p)
    num_terms = length(p)

    if termPowers === nothing
        termPowers = generate_partitions(m - 1, num_terms)
    else
        try
            if size(termPowers) != (binomial(m + length(p) - 2, num_terms - 1), num_vars)
                throw("termPowers not valid")
            end
        catch e
            throw("termPowers not valid")
        end
    end

    nthreads = min(512, size(termPowers, 1))
    nblocks = cld(size(termPowers, 1), nthreads)

    cu_p = CuArray(polynomial_to_arr(p))
    inverse = CuArray(compute_inverses_mod_m(m))
    num_paddedrows = cld(size(termPowers, 1), nthreads) * nthreads - size(termPowers, 1)
    cu_termPowers = CuArray(vcat(termPowers, fill(zero(Int32), (num_paddedrows, size(termPowers, 2)))))
    result = CUDA.fill(zero(Int32), size(cu_termPowers, 1), 1 + num_vars)
    
    # nthreads = min(CUDA.attribute(
    #     device(),
    #     CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    # ), size(cu_termPowers, 1))

    println("nthreads: $nthreads")
    println("nblocks: $nblocks")
    println("cu_termPowers size: $(size(cu_termPowers))")
    println("result size: $(size(result))")

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        power_kernel!(cu_p, m, result, cu_termPowers, inverse, num_vars, num_terms)
    )

    return view(result, 1:size(termPowers, 1), :)
end


"""
kernel for raise to power thing
"""
function power_kernel!(cu_p, m, result, cu_termPowers, inverse, num_vars, num_terms)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    # stride = blockDim().x * gridDim().x
    result[idx, 1] = m - 1
    for j in 1:num_terms
        result[idx, 1] *= inverse[factorial_mod_m(cu_termPowers[idx, j], m)]
        result[idx, 1] *= cu_p[j, 1] ^ cu_termPowers[idx, j]
        result[idx, 1] = result[idx, 1] % m
        # CuArrays don't like p:q indexing I think
        for k in 2:num_vars + 1
            result[idx, k] += cu_termPowers[idx, j] * cu_p[j, k]
        end
    end

    return
end

"""
    generate_partitions(n, k)

Return all possible ways to distribute n identical balls into k distinct boxes.

No idea how to parallelize this, maybe dynamic parallelism?
"""
function generate_partitions(n, k)
    partitions = zeros(Int32, binomial(n + k - 1, k - 1), k)  # Initialize 2D array to store partitions
    idx = 1
    stack = [(n, k, [])]

    while !isempty(stack)
        n, k, current_partition = pop!(stack)
        if n == 0 && k == 0
            partitions[idx, :] = current_partition
            idx += 1
        elseif n >= 0 && k > 0
            for i in 0:n
                new_partition = copy(current_partition)
                push!(new_partition, i)
                push!(stack, (n - i, k - 1, new_partition))
            end
        end
    end

    return partitions
end

"""
    generate_termPowers(p, m)

Pre-generate resulting representations of p^m-1 mod m
"""
function generate_termPowers(p, m)
    return generate_partitions(m - 1, length(p))
end

testpoly = [[1, [1, 0]], [1, [0, 1]], [2, [2, 2]], [1, [3, 4]]]
# testpoly = [[1, [1, 0]], [1, [0, 1]]]
raise_to_mminus1_mod_m(testpoly, 14)
# Only errors when (n + k - 1) choose (k - 1) > 512

# polynomial1 = [[2, [2, 3, 1]], [2, [1, 2, 3]], [3, [3, 2, 1]], [5, [1, 1, 4]], [4, [4, 1, 1]], [3, [6, 0, 0]]]
# m = 17

# println("Time to compute p^m-1 mod m without pre-generating coefficients")
# @btime raise_to_mminus1_mod_m(polynomial1, m)

# println("Time to pre-generate coefficients")
# pregen = generate_termPowers(polynomial1, m)
# @btime generate_termPowers(polynomial1, m)

# println("Time to compute p^m-1 mod m with pre-generated coefficients")
# @btime raise_to_mminus1_mod_m(polynomial1, m, pregen)

