using CUDA
using Test

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


"""
    compute_factorials_mod_m(m)

Return array of factorials mod m

factorials = compute_factorials_mod_m(m)
To get n! mod m, call factorials[n + 1]. Only works for n < m.
"""
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
    get_num_variables(p)

Return number of variables in polynomial represented by p
"""
function get_num_variables(p)
    return length(p[1][2])
end

"""
    polynomial_to_arr(p)

Convert p into array 
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