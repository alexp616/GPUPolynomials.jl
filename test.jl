using CUDA
using BenchmarkTools


result = CuArray(rand(Int32, (10000000, 6)))

function sortByCol(arr, col)
    arr = arr[sortperm(arr[:, col]), :]
end



@btime sortByCol(result, 6)