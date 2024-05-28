using CUDA
using BenchmarkTools

function sort_keys_with_values(keys, values)
    # Get permutation indices to sort keys array
    perm = sortperm(keys)

    # Rearrange keys and values arrays according to permutation indices
    sorted_keys = keys[perm]
    sorted_values = values[perm]

    return sorted_keys, sorted_values
end

keys = CuArray(rand(1:10, 100))
values = CuArray([1 for _ in 1:100])

keys, values = sort_keys_with_values(keys, values)

println(keys)
println(values)
# println("Sorted keys: ", sorted_keys)
# println("Corresponding sorted values: ", sorted_values)