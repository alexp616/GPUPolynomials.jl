using CUDA
using BenchmarkTools
# arr = CuArray(rand(1:1000, 100000000))
# CUDA.@time arr2 = reshape(arr, 10000, 10000)

function new_sort!(arr)
    matrixDim = ceil(Int, sqrt(length(arr)))
    numPaddedZeros = matrixDim ^ 2 - length(arr)
    matrix = Array(reshape(vcat(arr, CUDA.zeros(Int, numPaddedZeros)), matrixDim, matrixDim))

    Threads.@threads for row in axes(matrix, 1)
        subarr = @view matrix[row, :]
        sort!(subarr)
    end

    Threads.@threads for col in axes(matrix, 2)
        subarr = @view matrix[:, col]
        sort!(subarr)
    end

    return matrix
end

arr = rand(1:100000000, 100000000)

@time new_sort!(arr)