using Test

function factorial_mod_m(n, m)
    result = 1
    for i in 2:n
        result *= i
        result %= m
    end
    return result
end
@test factorial_mod_m(6, 7) == 6

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

function get_num_variables(p)
    return length(p[1][2])
end
@test get_num_variables([[1, [1, 0, 0, 0]]]) == 4

function raise_to_n_mod_m(p, n, m)
    inverses = compute_inverses_mod_m(m)

    # termPowers is all possible combinations of powers of the polynomial
    # if the polynomial has 3 terms to be raised to the 4th then termPowers contains
    # [4, 0, 0], [0, 4, 0], [0, 0, 4], [3, 1, 0] ... etc
    termPowers = generate_partitions(n, length(p), [])
    result = [copy(p[1]) for i in eachindex(termPowers)]
    nfac = factorial(n)

    for i in eachindex(termPowers)
        coeff = nfac
        # coeff = factorial_mod_m(n, m)
        degrees = zeros(Int, length(p[1][2]))
        # coeff = m - 1
        println("---------------------")
        for j in eachindex(termPowers[i])
            coeff ÷= factorial(termPowers[i][j])
            # coeff *= inverses[factorial_mod_m(termPowers[i][j], m)]
            
            coeff *= p[j][1] ^ termPowers[i][j]
        
            # coeff %= m
            degrees += termPowers[i][j] .* p[j][2]

        end
        result[i][1] = coeff
        result[i][2] = degrees
    end

    return result
end



function generate_partitions(n, k, current_partition)
    if n == 0 && k == 0
        return [current_partition]
    elseif n >= 0 && k > 0
        partitions = []
        for i in 0:n
            new_partition = push!(copy(current_partition), i)
            partitions = vcat(partitions, generate_partitions(n - i, k - 1, new_partition))
        end
        return partitions
    end
    return []
end

polynomial = [[2, [1, 1]], [3, [0, 1]]]
raise_to_n_mod_m(polynomial, 4, 5)

# partitions = generate_partitions(n, k, [])

# println(partitions)