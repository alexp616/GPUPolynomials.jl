using CUDA

# arr = CuArray(rand(1:1000, 100000000))
# CUDA.@time arr2 = reshape(arr, 10000, 10000)

function 2d_sort!(arr::CuArray{Int, 1})
    matrixDim = ceil(Int, sqrt(length(arr)))
    numPaddedZeros = matrixDim ^ 2 - length(arr)
    matrix = reshape(vcat(arr, CUDA.zeros(Int, numPaddedZeros)), matrixDim, matrixDim)

    nthreads = min(512, matrixDim)
    nblocks = fld(matrixDim, nthreads)

    last_block_threads = matrixDim - nthreads * nblocks

    if last_block_threads > 0
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            # dynamic = true,
            sort_row_kernel!(matrix, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        # dynamic = true,
        sort_row_kernel!(matrix)
    )

    if last_block_threads > 0
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            # dynamic = true,
            sort_col_kernel!(matrix, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        # dynamic = true,
        sort_col_kernel!(matrix)
    )

    return matrix

end

function sort_row_kernel!(matrix::CuDeviceMatrix{Int, 1}, offset = 0)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    # subarr = @view arr[tid, :]
    
    iterative_merge_sort!(matrix[tid, :])
    return
end

function sort_col_kernel!(matrix::CuDeviceMatrix{Int, 1}, offset = 0)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset

    # subarr = @view arr[:, tid]
    
    iterative_merge_sort!(matrix[:, tid])
    return
end

function merge!(arr, temp, left, mid, right)
    i = left
    j = mid + 1
    k = left
    
    while i <= mid && j <= right
        if arr[i] <= arr[j]
            temp[k] = arr[i]
            i += 1
        else
            temp[k] = arr[j]
            j += 1
        end
        k += 1
    end
    
    while i <= mid
        temp[k] = arr[i]
        i += 1
        k += 1
    end
    
    while j <= right
        temp[k] = arr[j]
        j += 1
        k += 1
    end
    
    for i in left:right
        arr[i] = temp[i]
    end
end

function iterative_merge_sort!(arr::CuArray{Int})
    n = length(arr)
    temp = similar(arr)
    
    curr_size = 1
    while curr_size < n
        left_start = 1
        while left_start <= n - curr_size
            mid = left_start + curr_size - 1
            right_end = min(left_start + 2 * curr_size - 1, n)
            
            merge!(arr, temp, left_start, mid, right_end)
            
            left_start += 2 * curr_size
        end
        curr_size *= 2
    end
    return arr
end

arr = CuArray(rand(1:100, 100))

2d_sort!(arr)
# arr2 = CuArray(rand(1:1000, 4000))
# CUDA.@time sort!(arr2)



# arr = rand(1:1000, 10000)
# # println("Original array: ", arr)
# @time iterative_merge_sort!(arr)

# arr = rand(1:1000, 10000)
# @time sort!(arr)
# println("Sorted array: ", arr)