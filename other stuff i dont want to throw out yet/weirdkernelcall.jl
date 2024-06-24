using CUDA
using BenchmarkTools


p1 = CuArray([i for i in 1:5000])
p2 = CuArray([i*10 for i in 1:5000])

function blah(p1, p2)

    result = CUDA.zeros(length(p1) * length(p2))

    nthreads = min(512, length(result))
    nblocks = fld(length(result), nthreads)

    last_block_threads = length(result) - nthreads * nblocks

    function kernel(poly1, poly2, length2, result, offset)
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
        idx1 = cld(tid, length2)
        idx2 = tid - (idx1 - 1) * length2

        result[tid] = poly1[idx1] + poly2[idx2]
        return
    end

    if last_block_threads > 0
        @cuda(
            threads = last_block_threads,
            blocks = 1,
            kernel(p1, p2, length(p2), result, nthreads * nblocks)
        )
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        kernel(p1, p2, length(p2), result, 0)
    )
    
    return result
end

function blah2(p1, p2)
    numPaddedRows = cld(length(p1), 512) * 512 - length(p1)
    cu_p1 = vcat(p1, CUDA.zeros(Int, numPaddedRows))

    result = CUDA.zeros(length(cu_p1) * length(p2))

    nthreads = min(512, length(result))
    nblocks = cld(length(result), nthreads)

    function kernel(poly1, poly2, length2, result, offset)
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x + offset
        idx1 = cld(tid, length2)
        idx2 = tid - (idx1 - 1) * length2

        result[tid] = poly1[idx1] + poly2[idx2]
        return
    end

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        kernel(cu_p1, p2, length(p2), result, 0)
    )
    
    return result
end

jit = CuArray([1, 2, 3])
blah(jit, jit)
blah2(jit, jit)


CUDA.@time blah(p1, p2)
CUDA.@time blah2(p1, p2)

# for 5000 x 5000:
# blah() 23 ms
# blah2() 36 ms




# function get_last_element(arr)
#     return Array(arr)[end]
# end

# a = 0
# @btime a = Array(p1[end:end])[1]
# @btime a = get_last_element(p1)

# function set_value(arr, idx, num)
#     arr[idx] = num
#     return
# end

# @btime @cuda set_value(p1, 400, 2)