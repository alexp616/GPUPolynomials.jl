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