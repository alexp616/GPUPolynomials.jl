using CUDA

inputarr = CuArray([
    0 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0
])


function mystery(arr)
    nthreads = size(arr, 1)
    nblocks = 1

    println("this ran")
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        mysterykernel1(arr)
    )
    println("this ran2")

    return arr
end

function mysterykernel1(arr)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nthreads = size(arr, 2)
    nblocks = 1

    subarr = @view arr[tid, :]
    @cuda(
        threads = nthreads,
        blocks = nblocks,
        dynamic = true,
        mysterykernel2(subarr)
    )

    return
end

function mysterykernel2(arr)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    arr[tid] += tid
    
    return
end 

println(mystery(inputarr))
# println(Array(mystery(inputarr)))

# using CUDA


# #this is an example parent kernel, every thread calls child kernel
# function example_parent()
#     tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     @cuda(
#         threads = 5,
#         blocks = 1,
#         dynamic = true,
#         example_child(tidx)
#     )
#     return nothing
# end

# #this is an example child kernel, every thread calls a print function
# function example_child(tidx)
#     tidxx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     @cuprintln("the message from: parent thread $tidx and child thread $tidxx")
#     return nothing
# end

# function rundynamic()
#     @cuda(
#         threads = 8,
#         blocks = 1,
#         example_parent()
#     )

#     return
# end

# rundynamic()