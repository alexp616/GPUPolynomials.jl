using CUDA
using BenchmarkTools

function cpu_merge(arr1::Vector{T}, arr2::Vector{T}) where T<:Real
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

function gpu_merge(arr1::CuVector{T}, arr2::CuVector{T}) where T<:Real
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

"""
    gpu_merge_by_key(aKeys, aValues, bKeys, bValues)

Merge 2 key-value pairs of arrays by their keys. Example:
ak = [3, 5, 6, 8]
av = [6, 2, 3, 5]
bk = [1, 4, 5, 7]
bv = [5, 2, 4, 7]

rk, rv = gpu_merge_by_key(ak, av, bk, bv)
rk: [1, 3, 4, 5, 5, 6, 7, 8]
rv: [5, 6, 2, 2, 4, 3, 7, 5]
"""
function gpu_merge_by_key(aKeys::CuVector{K}, aValues::CuVector{V}, bKeys::CuVector{K}, bValues::CuVector{V}) where {K, V}<:Real
    @assert length(aKeys) == length(aValues) && length(bKeys) == length(bValues) "length of key-value array pairs must match!"
    resultLen = length(aKeys) + length(bKeys)
    resultKeys = CUDA.zeros(K, resultLen)
    resultValues = CUDA.zeros(V, resultLen)
    totalThreads = div(resultLen, 10)

    kernel = @cuda launch = false gpu_merge_by_key_kernel(aKeys, aValues, bKeys, bValues, resultKeys, resultValues, resultLen, totalThreads)
    threads = min(totalThreads, config.threads)
    blocks = cld(totalThreads, threads)

    CUDA.@sync kernel(aKeys, aValues, bKeys, bValues, resultKeys, resultValues, resultLen, totalThreads; threads = threads, blocks = blocks)

    return resultKeys, resultValues
end

function gpu_merge_by_key_kernel(aKeys, aValues, bKeys, bValues, resultKeys, resultValues, resultLen, totalThreads)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1

    if tid <= totalThreads
        aptr = 0
        bptr = 0
        diag = cld(tid * (resultLen), totalThreads)
        right = min(diag, length(aKeys))
        left = diag - right
        while true
            mid = (right + left) >> 1

            bcoord = diag - mid
            if (mid == length(aKeys) || bcoord == length(bKeys)) || (mid == 0 || bcoord == 0)
                aptr = mid + 1
                bptr = bcoord + 1
                break
            end

            if aKeys[mid + 1] > bKeys[bcoord]
                if aKeys[mid] <= bKeys[bcoord + 1]
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
            if aptr > length(aKeys)
                switch = 1
                break
            end
            if bptr > length(bKeys)
                switch = 2
                break
            end
            if aKeys[aptr] < bKeys[bptr]
                resultKeys[step + diag + 1] = aKeys[aptr]
                resultValues[step + diag + 1] = aValues[aptr]
                aptr += 1
            else
                resultKeys[step + diag + 1] = bKeys[bptr]
                resultValues[step + diag + 1] = bValues[bptr]
                bptr += 1
            end
            step += 1
        end

        if switch == 2
            while step < steps && aptr <= length(aKeys)
                resultKeys[step + diag + 1] = aKeys[aptr]
                resultValues[step + diag + 1] = aValues[aptr]
                step += 1
                aptr += 1
            end
        end
        if switch == 1
            while step < steps && bptr <= length(bKeys)
                resultKeys[step + diag + 1] = bKeys[bptr]
                resultValues[step + diag + 1] = bValues[bptr]
                step += 1
                bptr += 1
            end
        end
    end
    return 
end

function test_gpu_merge()

    for num in [10000000, 100000000, 1000000000]
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