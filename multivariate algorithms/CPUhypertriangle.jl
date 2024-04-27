using Test
using BenchmarkTools

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
    raise_to_mminus1_mod_m(p, m)

Return p^(m-1) mod m
"""
function raise_to_mminus1_mod_m(p, m, termPowers = nothing)
    inverses = compute_inverses_mod_m(m)

    # termPowers is all possible combinations of powers of the terms of the polynomial
    # if the polynomial has 3 terms to be raised to the 4th then termPowers contains
    # [4, 0, 0], [0, 4, 0], [0, 0, 4], [3, 1, 0] ... etc
    
    if termPowers === nothing
        termPowers = generate_partitions(m - 1, length(p))
    else
        try
            if length(termPowers) != binomial(m + length(p) - 2, length(p) - 1)
                throw("termPowers not valid")
            end
        catch e
            throw("termPowers not valid")
        end
    end

    result = [copy(p[1]) for i in eachindex(termPowers)]

    for i in eachindex(termPowers)
        # coeff = factorial_mod_m(n, m)
        degrees = zeros(Int, length(p[1][2]))
        coeff = m - 1
        for j in eachindex(termPowers[i])
            @inbounds coeff *= inverses[factorial_mod_m(termPowers[i][j], m)]
            
            @inbounds coeff *= p[j][1] ^ termPowers[i][j] % m
            # println("idx: $i, j: $j, coeff: $coeff")
            println("idx: $i, j: $j, $(p[j][1]) ^ $(termPowers[i][j]) mod $m: $(p[j][1] ^ termPowers[i][j] % m)")
            coeff = coeff % m
            @inbounds degrees += termPowers[i][j] .* p[j][2]

        end
        result[i][1] = coeff
        result[i][2] = degrees
    end

    return result
end

"""
    generate_partitions(n, k)

Return all possible ways to distribute n identical balls into k distinct boxes.
"""
function generate_partitions(n, k)
    partitions = []
    stack = [(n, k, [])]

    while !isempty(stack)
        n, k, current_partition = pop!(stack)
        if n == 0 && k == 0
            push!(partitions, current_partition)
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

generate_partitions(3,5)

"""
    generate_termPowers(p, m)

Pre-generate resulting representations of p^m-1 mod m
"""
function generate_termPowers(p, m)
    return generate_partitions(m - 1, length(p))
end

polynomial1 = [
    [1, [1, 0]],
    [2, [0, 1]],
    [3, [1, 1]]
]
m = 5
# pregen = generate_termPowers(testpoly, 3)
println(raise_to_mminus1_mod_m(polynomial1, m))

# @test raise_to_mminus1_mod_m(testpoly, 5, pregen) == raise_to_mminus1_mod_m(testpoly, 5)


# polynomial1 = [[2, [2, 3, 1]], [2, [1, 2, 3]], [3, [3, 2, 1]], [5, [1, 1, 4]], [4, [4, 1, 1]], [3, [6, 0, 0]]]

# println("Time to compute p^m-1 mod m without pre-generating coefficients")
# @btime raise_to_mminus1_mod_m(polynomial1, 31)

# println("Time to pre-generate coefficients")
# pregen = generate_termPowers(polynomial1, 31)
# @btime generate_termPowers(polynomial1, 31)

# println("Time to compute p^m-1 mod m with pre-generated coefficients")
# @btime raise_to_mminus1_mod_m(polynomial1, 31, pregen)