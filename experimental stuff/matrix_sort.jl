module MatrixSort

export matrix_sort!, matrix_sortperm
using CUDA
using Statistics

function matrix_sort!(arr::CuVector{T}) where T<:Real
    matrixDim = ceil(Int, sqrt(length(arr)))
    padding = matrixDim ^ 2 - length(arr)
    matrix = reshape(vcat(arr, CUDA.fill(typemax(T), padding)), matrixDim, matrixDim)
    result = CUDA.zeros(T, matrixDim^2)

    # Sort columns and rows of matrix
    sort!(matrix; dims = 1)
    sort!(matrix; dims = 2)

    # Counting sort thing
    kernel = @cuda launch=false count_less_than_kernel!(matrix, result)
    config = launch_configuration(kernel.fun)
    temp = floor(Int, sqrt(config.threads))
    threads = min(size(matrix), (temp, temp))
    blocks = cld.(size(matrix), threads)

    CUDA.@sync kernel(matrix, result; threads = threads, blocks = blocks)

    arr .= result[1:length(arr)]
end

# Because this uses a counting sort, there needs to be another function that tells if two elements are equal
# otherwise two elements will try to go into the same index
function count_less_than_kernel!(matrix::CuDeviceArray{T}, result::CuDeviceVector{T}) where T<:Real
    idxcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idxrow = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    @inbounds if idxrow <= size(matrix, 1) && idxcol <= size(matrix, 2)
        less = idxcol * idxrow - 1
        X = matrix[idxrow, idxcol]

        # Counting elements less than X in quadrant 1
        r1, c1 = idxrow - 1, idxcol + 1
        while r1 >= 1 && c1 <= size(matrix, 2)
            if X <= matrix[r1, c1]
                r1 -= 1
            else
                less += r1
                c1 += 1
            end
        end

        # Counting elements less than X in quadrant 3
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

function run_benchmarks()
    len = 10000
    for _ in 1:4
        cudajltimes = []
        matrixsorttimes = []
        unifiedmemorytimes = []
        cputimes = []
        for i in 1:15
            C = CuArray(rand(1:len, len))
            dfdf = CUDA.@timed matrix_sort!(C)
            @assert issorted(Array(C))
            push!(matrixsorttimes, dfdf.time)

            B = CuArray(rand(1:len, len))
            sfsf = CUDA.@timed sort!(B)
            push!(cudajltimes, sfsf.time)

            D = cu(rand(1:len, len); unified = true)
            E = unsafe_wrap(Array, D)
            afaf = @timed sort!(E)
            push!(unifiedmemorytimes, afaf.time)

            F = rand(1:len, len)
            qfqf = @timed sort!(F)
            push!(cputimes, qfqf.time)
        end

        # get rid of priming jitter value
        deleteat!(matrixsorttimes, 1)
        deleteat!(cudajltimes, 1)
        deleteat!(unifiedmemorytimes, 1)
        deleteat!(cputimes, 1)

        println("average time to sort $len elements with matrix_sort!(): ")
        println("$(mean(matrixsorttimes)) s")

        println("average time to sort $len elements with CUDA.sort!(): ")
        println("$(mean(cudajltimes)) s")

        println("average time to sort $len elements with cpu and unified memory: ")
        println("$(mean(unifiedmemorytimes)) s")

        println("average time to sort $len elements with cpu: ")
        println("$(mean(cputimes)) s")
        println("")
        len *= 10
    end
end

function test_sort(length = 10)
    jitter = CuArray(rand(1:9, 9))
    matrix_sort!(jitter)

    arr = CuArray(rand(1:length, length))
    CUDA.@profile trace=true matrix_sort!(arr)
end

end