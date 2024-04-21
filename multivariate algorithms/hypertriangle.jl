using Test

function polynomial_to_pow(p, n)

end

function get_num_variables(p)
    return length(p[1][2])
end
@test get_num_variables([[1, [1, 0, 0, 0]]]) == 4


function polynomial_to_arr(p)
    result = zeros(Int, length(p) * (get_variables(p) + 1))

    for i in 0:length(p)-1
        result[i * 4 + 1] = p[i + 1][1]
        for j in eachindex(p[i + 1][2])
            result[i * 4 + 1 + j] = p[i + 1][2][j]
        end
    end

    return result
end
@test polynomial_to_arr([(1, (1, 0, 0)), (2, (0, 1, 0)), (3, (0, 0, 1))]) == [1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1]

p1 = [(1, (1, 0, 0)), (2, (0, 1, 0)), (3, (0, 0, 1))]