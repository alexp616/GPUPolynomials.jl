using CUDA

#this is an example parent kernel, every thread calls child kernel
function example_parent()
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tidx <= 5
        @cuda threads = (32, 1, 1) dynamic = true example_child(tidx)
    end
    return nothing
end



#this is an example child kernel, every thread calls a print function
function example_child(tidx)
    tidxx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tidxx <= 3
        @cuprintln("the message from: parent thread $tidx and child thread $tidxx")
    end
    return nothing
end

@cuda threads = (32, 1, 1) example_parent()



# arr = CuArray(rand(1:10, 10))
# function sus_sort!(arr)
#     sort!(arr)
#     return nothing
# end
# @cuda sus_sort!(arr)