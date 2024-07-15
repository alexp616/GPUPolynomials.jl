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
    # 10 values per thread because why not
    totalThreads = div(resultLen, 10)

    kernel = CUDA.@sync @cuda launch = false gpu_merge_kernel(arr1, arr2, result, resultLen, totalThreads)
    config = launch_configuration(kernel.fun)
    threads = min(totalThreads, config.threads)
    blocks = cld(totalThreads, threads)

    kernel(arr1, arr2, result, resultLen, totalThreads; threads = threads, blocks = blocks)

    return result
end

# 1 indexed
function gpu_merge_kernel(a, b, result, resultLen, totalThreads)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1

    if tid <= totalThreads
        aptr = 0
        bptr = 0
        diag = cld(tid * (resultLen), totalThreads)
        right = min(diag, length(a))
        left = diag - right
        while true
            mid = (right + left) >> 1

            bcoord = diag - mid
            if (mid == length(a) || bcoord == length(b)) || (mid == 0 || bcoord == 0)
                aptr = mid + 1
                bptr = bcoord + 1
                break
            end

            if a[mid + 1] > b[bcoord]
                if a[mid] <= b[bcoord + 1]
                    aptr = mid + 1
                    bptr = bcoord + 1
                    break
                else
                    right = mid - 1
                end
            else
                left = mid + 1
            end
        end

        step = 0
        steps = min(div(resultLen, totalThreads), resultLen - diag)
        switch = 0
        while step < steps
            if aptr > length(a)
                switch = 1
                break
            end
            if bptr > length(b)
                switch = 2
                break
            end
            if a[aptr] < b[bptr]
                result[step + diag + 1] = a[aptr]
                aptr += 1
            else
                result[step + diag + 1] = b[bptr]
                bptr += 1
            end
            step += 1
        end

        if switch == 2
            while step < steps && aptr <= length(a)
                result[step + diag + 1] = a[aptr]
                step += 1
                aptr += 1
            end
        end
        if switch == 1
            while step < steps && bptr <= length(b)
                result[step + diag + 1] = b[bptr]
                step += 1
                bptr += 1
            end
        end
    end
    return 
end

function test_gpu_merge()

    for num in [1000, 10000, 1000000, 10000000, 100000000, 1000000000]
        arr1 = rand(1:num, num)
        arr2 = rand(1:num, num)

        sort!(arr1)
        sort!(arr2)
        
        println("Time to merge 2 * 10 ^ $(Int(round(log(10, num)))) elements on CPU: ")
        @time arr3 = cpu_merge(arr1, arr2)

        cu_arr1 = CuArray(arr1)
        cu_arr2 = CuArray(arr2)

        println("Time to merge 2 * 10 ^ $(Int(round(log(10, num)))) elements on GPU: ")
        CUDA.@time cu_arr3 = gpu_merge(cu_arr1, cu_arr2)

        @assert arr3 == Array(cu_arr3)
    end
    
end

# @benchmark C = cpu_merge(A, B)
# len = 100000

# for t in 1:10
#     A = CuArray(rand(1:len, len))
#     B = CuArray(rand(1:len, len))
#     sort!(A)
#     sort!(B)
#     println("trial $t:")
#     CUDA.@time gpu_merge(A, B)
# end