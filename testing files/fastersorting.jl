using CUDA

# arr = CuArray(rand(1:1000, 100000000))
# CUDA.@time arr2 = reshape(arr, 10000, 10000)

# function sus_sort!(arr::AnyCuArray{T}) where {T :> Real}
#     matrixDim = ceil(Int, sqrt(length(arr)))
#     numPaddedZeros = matrixDim ^ 2 - length(arr)
#     matrix = reshape(vcat(arr, CUDA.zeros(T, numPaddedZeros)), matrixDim, matrixDim)

#     nthreads = min(512, matrixDim)
#     nblocks = fld(matrixDim, nthreads)

#     last_block_threads = matrixDim - nthreads * nblocks

#     if last_block_threads > 0
#         @cuda(
#             threads = last_block_threads,
#             blocks = 1,
#             sort_row_kernel!(arr, nthreads * nblocks)
#         )
#     end

#     CUDA.@sync @cuda(
#         threads = nthreads,
#         blocks = nblocks,
#         sort_row_kernel!(arr)
#     )

# end

# function sort_row_kernel!(arr, offset = 0)
#     tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

#     sort!()
# end

# function sort_col_kernel!(arr, offset = 0)

# end

# ceil(Int, sqrt(100000000))


arr = CuArray([
    1 3 2 4
    2 3 7 3
])

function sus_sort(arr)
    sort!(CuArray([1, 3, 2, 4]))
    CUDA.@sync @cuda(
        threads = 2,
        blocks = 1,
        sus_sort_kernel(arr)
    )
end

function sus_sort_kernel(arr)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    sort(@view arr[tid, :])
    return
end

sus_sort(arr)

sort!(rand(1:1000, 10000))
@time sort!(rand(1:1000, 100000000))
@time sort!(rand(1:1000, 10000))
# arr2 = CuArray(rand(1:1000, 4000))
# CUDA.@time sort!(arr2)