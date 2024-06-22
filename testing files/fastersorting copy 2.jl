using CUDA

# arr = CuArray(rand(1:1000, 100000000))
# CUDA.@time arr2 = reshape(arr, 10000, 10000)

function 2d_sort!(arr)
    nthreads = min(512, size(arr, 1))
    nblocks = cld(size(arr, 1), nthreads)

    func = CUDA.bitonic_sort!

    @cuda(
        threads = nthreads,
        blocks = nblocks,
        sort_row_kernel!(arr, func)
    )

    return arr
end

function sort_row_kernel!(arr, func)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= size(arr, 1)
        row = @view arr[tid, :]
        func(row)
    end

    return
end


arr = reshape(CuArray(rand(1:100, 100)), 10, 10)

2d_sort!(arr)
# arr2 = CuArray(rand(1:1000, 4000))
# CUDA.@time sort!(arr2)



# arr = rand(1:1000, 10000)
# # println("Original array: ", arr)
# @time iterative_merge_sort!(arr)

# arr = rand(1:1000, 10000)
# @time sort!(arr)
# println("Sorted array: ", arr)