using CUDA
using Statistics

function matrix_sort(arr::CuVector{T}) where T<:Real
    matrixDim = ceil(Int, sqrt(length(arr)))
    padding = matrixDim ^ 2 - length(arr)
    matrix = reshape(vcat(arr, CUDA.fill(typemax(T), padding)), matrixDim, matrixDim)
    result = CUDA.zeros(T, matrixDim^2)
    
    sort!(matrix, dims = 1)
    sort!(matrix, dims = 2)

    # Counting sort thing
    kernel = @cuda launch=false count_less_than_kernel!(matrix, result)
    config = launch_configuration(kernel.fun)
    temp = floor(Int, sqrt(config.threads))
    threads = min(size(matrix), (temp, temp))
    blocks = cld.(size(matrix), threads)

    CUDA.@sync blocking = true begin
        kernel(matrix, result; threads = threads, blocks = blocks)
    end

    return result[1:length(arr)]
end

function count_less_than_kernel!(matrix::CuDeviceArray{T}, result::CuDeviceVector{T}) where T<:Real
    idxcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idxrow = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    @inbounds if idxrow <= size(matrix, 1) && idxcol <= size(matrix, 2)
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


function run_benchmarks()
    len = 10000
    for _ in 1:4
        cudajltimes = []
        matrixsorttimes = []
        cputimes = []
        for i in 1:15
            B = CuArray(rand(1:len, len))
            sfsf = CUDA.@timed sort!(B)
            push!(cudajltimes, sfsf.time)

            C = CuArray(rand(1:len, len))
            dfdf = CUDA.@timed matrix_sort(C)
            push!(matrixsorttimes, dfdf.time)

            D = cu(rand(1:len, len); unified = true)
            E = unsafe_wrap(Array, D)
            afaf = @timed sort!(E)
            push!(cputimes, afaf.time)
        end

        # get rid of priming jitter value
        deleteat!(cudajltimes, 1)
        deleteat!(matrixsorttimes, 1)
        deleteat!(cputimes, 1)

        println("average time to sort $len elements with matrix_sort!(): ")
        println("$(mean(cudajltimes)) s")

        println("average time to sort $len elements with CUDA.sort!(): ")
        println("$(mean(matrixsorttimes)) s")

        println("average time to sort $len elements with cpu and unified memory: ")
        println("$(mean(cputimes)) s")
        println("")
        len *= 10
    end
end

run_benchmarks()