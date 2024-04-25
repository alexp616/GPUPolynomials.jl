using Test

function polynomial_to_pow(p, n)

end

function get_num_variables(p)
    return length(p[1][2])
end
@test get_num_variables([[1, [1, 0, 0, 0]]]) == 4


function polynomial_to_arr(p)
    num_cols = get_num_variables(p) + 1
    result = zeros(Int, length(p), num_cols)
    for i in eachindex(p)
        result[i, 1] = p[i][1]
        result[i, 2:num_cols] .= p[i][2]
    end

    return result
end

p1 = [(1, [1, 0, 0]), (2, [0, 1, 0]), (3, [0, 0, 1])]

println(polynomial_to_arr(p1))