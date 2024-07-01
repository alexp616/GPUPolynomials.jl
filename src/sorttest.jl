using CUDA

function quicksort!(arr, lo=1, hi=length(arr))
    if hi <= lo
        return
    end
    p = partition!(arr, lo, hi)
    quicksort!(arr, lo, p - 1)
    quicksort!(arr, p + 1, hi)

    return
end

function partition!(arr, lo, hi)
    pivot = arr[hi]
    i = lo - 1
    for j in lo:hi-1
        if arr[j] <= pivot
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
        end
    end
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1
end

function thread_sort!(arr)
    @cuda(
        threads = 1,
        blocks = 1,
        thread_sort_kernel!(arr)
    )
    return
end

function thread_sort_kernel!(arr)
    quicksort!(arr)
    return
end

function test_sort!()
    arr = CuArray(rand(1:100, 10000))

    CUDA.@time thread_sort!(arr)
    arr
end