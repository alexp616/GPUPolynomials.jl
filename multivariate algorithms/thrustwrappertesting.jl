using CUDA

function sort_by_col(arr, col)
    arr .= arr[sortperm(arr[:, col]), :]
end

cuarr = CuArray([
    2 1
    3 2
    3 3
    3 4
    2 5
    2 6
    4 7
    3 8
])

sort_by_col(cuarr, 1)

println(cuarr)