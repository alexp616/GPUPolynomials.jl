using CUDA
using BenchmarkTools

function cpu_merge(arr1::Vector{T}, arr2::Vector{T}) where T<:Integer
    result = zeros(T, length(arr1) + length(arr2))
    idx1, idx2 = 1, 1
    for i in eachindex(result)
        if idx2 > length(arr2)
            result[i] = arr1[idx1]
            idx1 += 1
        elseif idx1 > length(arr1)
            result[i] = arr2[idx2]
            idx2 += 1
        else
            if arr1[idx1] < arr2[idx2]
                result[i] = arr1[idx1]
                idx1 += 1
            else
                result[i] = arr2[idx2]
                idx2 += 1
            end
        end
    end

    return result
end

function gpu_merge(arr1::CuVector{T}, arr2::CuVector{T}) where T<:Integer
    resultLen = length(arr1) + length(arr2)
    result = CUDA.zeros(T, resultLen)
    totalThreads = 10
    Adiag = zeros(Int, totalThreads)
    Bdiag = zeros(Int, totalThreads)

    function gpu_merge_kernel(a, b, result, resultLen, totalThreads)
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        index = tid * (length(a) + length(b)) / totalThreads
        atop = index > length(a) ? length(a) : index
        btop = index > length(a) ? index - length(A) : 0

        while true
            offset = div(atop + btop, 2)
            aidx = atop - offset
            bidx = btop + offset
            if a[aidx] > b[bidx - 1]

            end
        end
    end
end



# @benchmark C = cpu_merge(A, B)
len = 100000

for t in 1:10
    A = CuArray(rand(1:len, len))
    B = CuArray(rand(1:len, len))
    sort!(A)
    sort!(B)
    println("trial $t:")
    CUDA.@time gpu_merge(A, B)
end