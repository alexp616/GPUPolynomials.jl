using CUDA

# arr = CuArray(rand(1:1000, 100000000))
# CUDA.@time arr2 = reshape(arr, 10000, 10000)

function matrix_sort(arr::CuVector{T}) where T<:Integer
    matrixDim = ceil(Int, sqrt(length(arr)))
    padding = matrixDim ^ 2 - length(arr)
    matrix = reshape(vcat(arr, CUDA.fill(typemax(T), padding)), matrixDim, matrixDim)
    result = CUDA.zeros(T, matrixDim^2)
    
    # Sort rows
    kernel1 = @cuda launch=false sort_row_kernel!(matrix)
    config1 = launch_configuration(kernel1.fun)
    threads1 = min(size(matrix, 1), config1.threads)
    blocks1 = cld(size(matrix, 1), threads1)
    CUDA.@sync blocking = true begin
        kernel1(matrix; threads = threads1, blocks = blocks1)
    end
    
    
    # Sort columns
    kernel2 = @cuda launch=false sort_col_kernel!(matrix)
    CUDA.@sync blocking = true begin
        kernel2(matrix; threads = threads1, blocks = blocks1)
    end

    # println("matrix: ")
    # display(matrix)
    # println()

    # Counting sort thing
    kernel3 = @cuda launch=false count_less_than_kernel!(matrix, result)
    config3 = launch_configuration(kernel3.fun)
    temp = floor(Int, sqrt(config3.threads))
    threads3 = min(size(matrix), (temp, temp))
    blocks3 = cld.(size(matrix), threads3)

    CUDA.@sync blocking = true begin
        kernel3(matrix, result; threads = threads3, blocks = blocks3)
    end

    return result[1:length(arr)]
end


function sort_row_kernel!(matrix)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= size(matrix, 1)
        row = @view matrix[tid, :]
        quicksort!(row)
    end

    return
end

function sort_col_kernel!(matrix)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= size(matrix, 2)
        col = @view matrix[:, tid]
        quicksort!(col)
    end

    return
end

function count_less_than_kernel!(matrix, result)
    idxcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idxrow = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if idxrow <= size(matrix, 1) && idxcol <= size(matrix, 2)
        less = idxcol * idxrow - 1
        X = matrix[idxrow, idxcol]

        r1, c1 = idxrow - 1, idxcol + 1
        while r1 >= 1 && c1 <= size(matrix, 2)
            if X <= matrix[r1, c1]
                r1 -= 1
            else
                less += r1
                c1 += 1
            end
        end

        r3, c3 = idxrow + 1, idxcol - 1
        while r3 <= size(matrix, 1) && c3 >= 1
            if X < matrix[r3, c3]
                c3 -= 1
            else
                less += c3
                r3 += 1
            end
        end
        result[less + 1] = X
    end

    return
end

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

function test_sort(length)
    jit = CuArray(rand(1:9, 9))
    jit2 = matrix_sort(jit)
    jit3 = CuArray(rand(1:9, 9))
    jit4 = sort(jit3)

    arr = CuArray(rand(1:length, length))
    println("jelqsort time to sort array of length $length")
    CUDA.@time arr2 = matrix_sort(arr)

    arr3 = CuArray(rand(1:length, length))
    println("cuda.jl time to sort array of length arr $length")
    CUDA.@time stblibarr4 = sort(stdlibarr)
    
end