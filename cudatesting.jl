using CUDA
using Test
using BenchmarkTools

function slowMultiply1(p1, p2)
    temp = fill(0, length(p1) + length(p2)-1)
    for i in eachindex(p1) 
        for j in eachindex(p2)
            @inbounds temp[i + j - 1] += p1[i] * p2[j]
        end
    end
    return temp
end

# function slowMultiply2Kernel!(temp, p1, p2)
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     stride = blockDim().x * gridDim().x

#     for i = idx:stride:length(temp)
#         for j in eachindex(p1)
#             if i >= j && (i - j + 1) <= length(p2)
#                 @inbounds temp[i] += p1[j] * p2[i - j + 1]
#             end
#         end
#     end

#     return nothing
# end

function slowMultiply2Kernel!(temp, p1, p2)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x * gridDim().x

    for i = idx:stride:length(temp)
        for j in eachindex(p1)
            if i >= j && (i - j + 1) <= length(p2)
                @cuprintln("ThreadID: $idx")
                @inbounds temp[i] += p1[j] * p2[i - j + 1]
            end
        end
    end

    return nothing
end

function slowMultiply2(p1, p2)
    temp = CUDA.fill(zero(Int64), length(p1) + length(p2) - 1)
    t1 = CuArray(p1)
    t2 = CuArray(p2)

    nthreads = min(CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    ), length(p1)*length(p2))

    nblocks = cld(length(p1)*length(p2), nthreads)

    println("Threads: $nthreads, Blocks: $nblocks")
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        slowMultiply2Kernel!(temp, t1, t2)
    )

    return temp
end


polynomial1 = [1 for i in 1:5]
polynomial2 = [1 for i in 1:5]

println("-----------------start------------------")
# @btime slowMultiply1(polynomial1, polynomial2)
# btime slowMultiply2(polynomial1, polynomial2)
# # slowMultiply2 catches up and becomes faster at squaring a 1700-degree polynomial (on my NVIDIA GeForce RTX 3050 Laptop GPU)
slowMultiply2(polynomial1, polynomial2)
# @test (slowMultiply1(polynomial1, polynomial2) == Array(slowMultiply2(polynomial1, polynomial2)))
println("------------------end-------------------")
