using CUDA
using BenchmarkTools

array = rand(Int32, 10000000, 2)
cu_array = CuArray(array)

function sort_by_col!(arr, col)
    arr .= arr[sortperm(arr[:, col]), :]
end

@btime sort_by_col!(cu_array, 1)