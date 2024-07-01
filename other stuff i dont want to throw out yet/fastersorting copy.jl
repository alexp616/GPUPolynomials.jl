using CUDA



function sus(arr)
    kernel = @cuda launch=false sus_kernel(arr)
    config = launch_configuration(kernel.fun)
    temp = floor(Int, sqrt(config.threads))
    threads = min(size(arr), (temp, temp))
    blocks = cld.(size(arr), threads)
    println("threads: $threads")
    println("blocks: $blocks")

    kernel(arr; threads = threads, blocks = blocks)
end

function sus_kernel(arr)
    idxcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idxrow = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if idxrow <= size(arr, 1) && idxcol <= size(arr, 2)
        arr[idxrow, idxcol] = idxrow + idxcol
    end

    return
end

arr = reshape(CUDA.zeros(1000^2), 1000, 1000)

sus(arr)

arr