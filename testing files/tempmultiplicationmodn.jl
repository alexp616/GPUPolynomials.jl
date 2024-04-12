function add_row_of_zeros(matrix)
    num_cols = size(matrix, 2)
    zero_row = zeros(1, num_cols)
    return vcat(matrix, zero_row)
end

# Example usage
matrix = [1 2 3; 4 5 6]
matrix_with_zeros = add_row_of_zeros(matrix)