using CUDA
using BenchmarkTools

arr = [
    1 2 3 4
    5 6 7 8
    9 10 11 12
    13 14 15 16
]

function bitReverse(x, log2n)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

function parallelBitReverseCopyKernel(p, dest, len, log2n)
    idx1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    idx2 = idx1 + Int(len / 2)

    rev1 = bitReverse(idx1, log2n)
    rev2 = bitReverse(idx2, log2n)

    dest[idx1 + 1] = p[rev1 + 1]
    dest[idx2 + 1] = p[rev2 + 1]
    return nothing
end

function parallelBitReverseCopy(p)
    @assert ispow2(length(p)) "p must be an array with length of a power of 2"
    len = length(p)

    result = CUDA.zeros(eltype(p), len)

    nthreads = min(512, len ÷ 2)

    nblocks = cld(len ÷ 2, nthreads)

    log2n = Int(log2(len))

    # CUDA.@sync @cuda(
    @cuda(
        threads = nthreads,
        blocks = nblocks,
        parallelBitReverseCopyKernel(p, result, len, log2n)
    )
    
    return result
end


# doesn't include necessary bit reverse step, threw it in multiply already
function GPUDFT2!(arr, inverted = 1)
    numRows = size(arr, 1)
    numCols = size(arr, 2)

    @assert ispow2(numRows) && ispow2(numCols) "you messed up"

    nthreadsrow = min(512, numRows)

    nblocksrow = cld(numRows, nthreadsrow)

    nthreadsrowchild = min(512, numCols >> 1)

    nblocksrowchild = cld(numCols >> 1, nthreadsrowchild)

    log2numCols = Int(log2(numCols))

    # CUDA.@sync @cuda(
    @cuda(
        threads = nthreadsrow,
        blocks = nblocksrow,
        GPUDFTRowParentKernel!(arr, nthreadsrowchild, nblocksrowchild, log2numCols, inverted)
    )

    nthreadscol = min(512, numCols)

    nblockscol = cld(numCols, nthreadscol)

    nthreadscolchild = min(512, numRows >> 1)

    nblockscolchild = cld(numRows >> 1, nthreadscolchild)

    log2numRows = Int(log2(numRows))

    # CUDA.@sync @cuda(
    @cuda(
        threads = nthreadscol,
        blocks = nblockscol,
        GPUDFTColParentKernel!(arr, nthreadscolchild, nblockscolchild, log2numRows, inverted)
    )

    return
end

function GPUDFTRowParentKernel!(arr, nthreadsrowchild, nblocksrowchild, log2numCols, inverted)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    row = @view arr[idx, :]
    for i in 1:log2numCols
        m2 = 1 << (i - 1)
        theta_m = cis(inverted * pi / m2)
        
        # magic because its magic how i figured it out
        magic = 1 << (log2numCols - i)
        # magic = (log2n / 2) / m2
        @cuda(
            threads = nthreadsrowchild,
            blocks = nblocksrowchild,
            dynamic = true,
            GPUDFTKernel!(row, m2, theta_m, magic)
        )
    end
end

function GPUDFTColParentKernel!(arr, nthreadscolchild, nblockscolchild, log2numRows, inverted)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    col = @view arr[:, idx]
    for i in 1:log2numRows
        m2 = 1 << (i - 1)
        theta_m = cis(inverted * pi / m2)
        
        # magic because its magic how i figured it out
        magic = 1 << (log2numRows - i)
        # magic = (log2n / 2) / m2
        @cuda(
            threads = nthreadscolchild,
            blocks = nblockscolchild,
            dynamic = true,
            GPUDFTKernel!(col, m2, theta_m, magic)
        )
    end
end

function GPUDFTKernel!(result, m2, theta_m, magic)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = Int(2 * m2 * (idx % magic) + floor(idx/magic))

    theta = (theta_m) ^ (floor(idx/magic))

    t = theta * result[k + m2 + 1]
    u = result[k + 1]

    result[k + 1] = u + t
    result[k + m2 + 1] = u - t
    return 
end

function generate_butterfly_permutations(rows, cols)
    @assert ispow2(rows) && ispow2(cols) "rows and cols must both be powers of 2"
    rowperm = parallelBitReverseCopy(CuArray([i for i in 1:rows]))
    colperm = parallelBitReverseCopy(CuArray([i for i in 1:cols]))
    return (rowperm, colperm)
end

function multiply_2d!(p1, p2, pregen_butterfly = nothing)
    # size(p1, 1) > size(p2, 1) ? numPaddedColumns = nextpow(2, size(p1, 1)) : numPaddedColumns = nextpow(2, size(p2, 1))
    # p1 = pad_columns(p1, numPaddedColumns)
    # p2 = pad_columns(p2, numPaddedColumns)
    
    # size(p1, 2) > size(p2, 2) ? numPaddedRows = nextpow(2, size(p1, 2)) : numPaddedRows = nextpow(2, size(p2, 2))
    # p1 = pad_rows(p1, numPaddedRows)
    # p2 = pad_rows(p2, numPaddedColumns)

    @assert size(p1) == size(p2) "p1 and p2 aren't the same size"

    if pregen_butterfly === nothing
        throw("you messed up")
    end

    p1 = (p1[pregen_butterfly[1], :])[:, pregen_butterfly[2]]
    p2 = (p2[pregen_butterfly[1], :])[:, pregen_butterfly[2]]


    GPUDFT2!(p1)
    GPUDFT2!(p2)

    c = p1 .* p2
    c .÷= length(p1)

    GPUDFT2!(c, -1)

    return c
end

function pad_rows(arr, numRows)
    return vcat(arr, CUDA.zeros(eltype(arr), numRows, size(arr, 2)))
end

function pad_columns(arr, numColumns)
    return hcat(arr, CUDA.zeros(eltype(arr), size(arr, 1), numColumns))
end

# Change these to any powers of 2, our goal is 8192 x 8192
rows = 512
cols = 512
p1 = CUDA.fill(1, rows, cols)

pregen_butterfly = generate_butterfly_permutations(rows, cols)

println("Time to multiply a $rows x $cols by a $rows x $cols array")
pregen_butterfly2 = generate_butterfly_permutations(4, 4)
multiply_2d!(CUDA.fill(1, 4, 4), CUDA.fill(1, 4, 4), pregen_butterfly2)
CUDA.@profile multiply_2d!(p1, p1, pregen_butterfly)