using CUDA
using Test
include("../utils.jl")

"""
This file includes 3 main algorithms and their implementation on the GPU.
    - GPUSlowMultiply
    - GPUDFT
    - GPUMultiply

All of these run using the CUDA library, the main way to interface with NVIDIA 
GPUs in Julia.

The CPU-parallelized versions are in CPUAlgorithms.jl
"""



"""
    GPUSlowMultiply(p1, p2)

Multiply two polynomials represented by vectors p1 and p2 and return
the resulting polynomial as an vector.

GPU-Parallelized version of O(mn) algorithm.
"""
function GPUSlowMultiply(p1, p2)
    temp = CUDA.fill(zero(Int64), length(p1) + length(p2) - 1)
    t1 = CuArray(p1)
    t2 = CuArray(p2)

    nthreads = min(CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    ), length(p1)*length(p2))

    nblocks = cld(length(p1)*length(p2), nthreads)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        GPUSlowMultiplyKernel!(temp, t1, t2)
    )

    return temp
end


"""
    GPUSlowMultiplyKernel!(temp, p1, p2)

Modifies temp for each thread launched by parent algorithm.
"""
function GPUSlowMultiplyKernel!(temp, p1, p2)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x * gridDim().x

    for i = idx:stride:length(temp)
        for j in eachindex(p1)
            if i >= j && (i - j + 1) <= length(p2)
                @inbounds temp[i] += p1[j] * p2[i - j + 1]
            end
        end
    end

    return nothing
end


"""
    GPUMultiply(p1, p2)

Multiply two polynomials represented by vectors p1 and p2 and return
the resulting polynomial as an vector. Assumes coefficients are integers.

GPU-Parallelized version of cpuMultiply in CPUAlgorithms.jl
"""
function GPUMultiply(p1, p2)
    # TODO Try putting everything into one method to see if significant change in speed
    n = Int(2^ceil(log2(length(p1) + length(p2) - 1)))
    log2n = UInt32(log2(n));
    finalLength = length(p1) + length(p2) - 1

    # TODO Decide whether throwing an if statement in the DFTs is faster or
    # if all of this memory copying is necessary.
    cudap1 = CuArray(append!(copy(p1), zeros(Int, n - length(p1))))
    cudap2 = CuArray(append!(copy(p2), zeros(Int, n - length(p2))))

    cudap1 = GPUDFT(cudap1, n, log2n)
    cudap2 = GPUDFT(cudap2, n, log2n)

    return Int.(round.(real.(Array(GPUIDFT(cudap1 .* cudap2, n, log2n))[1:finalLength])))
end


"""
    GPUDFT(p, n, log2n, inverted = 1)

Return the DFT of vector p as a vector. Output can be complex.

Does not work when log2(length(p)) is not an integer
"""
function GPUDFT(p, n, log2n, inverted = 1)
    result = CUDA.fill(ComplexF32(0), n)

    nthreads = min(CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    ), Int(n/2))

    nblocks = cld(Int(n/2), nthreads)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        parallelBitReverseCopy(p, result, n, log2n)
    )

    for i in 1:log2n
        m2 = 1 << (i - 1)
        theta_m = cis(inverted * pi / m2)
        
        # magic because its magic how i figured it out
        magic = 1 << (log2n - i)
        # magic = (log2n / 2) / m2
        CUDA.@sync @cuda(
            threads = nthreads,
            blocks = nblocks,
            GPUDFTKernel!(result, m2, theta_m, magic)
        )
    end

    return result
end


"""
    parallelBitReverseCopy(p, result, n, log2n)

Copies bit-reversed indices in p to result
"""
function parallelBitReverseCopy(p, result, n, log2n)
    idx1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    idx2 = idx1 + Int(n/2)

    rev1 = bitReverse(idx1, log2n)
    rev2 = bitReverse(idx2, log2n)

    result[idx1 + 1] = p[rev1 + 1]
    result[idx2 + 1] = p[rev2 + 1]
    return nothing
end


"""
    GPUDFTKernel!(result, m2, theta_m, magic)

Kernel function of GPUDFT() that builds DFT bottom up
"""
function GPUDFTKernel!(result, m2, theta_m, magic)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = Int(2 * m2 * (idx % magic) + floor(idx/magic))

    # Don't know if raising complex numbers to powers is efficient.
    theta = (theta_m) ^ (floor(idx/magic))

    t = theta * result[k + m2 + 1]
    u = result[k + 1]

    result[k + 1] = u + t
    result[k + m2 + 1] = u - t
    return 
end


"""
    GPUIDFT(y, n, log2n)

Return the inverse DFT of vector p as a vector.

Does not work when length(p) != 2^k for k∈Z
"""
function GPUIDFT(y, n, log2n)
    return GPUDFT(y, n, log2n, -1) ./ n
end

polynomial1 = [1, 2, 3, 4]

result = GPUMultiply(polynomial1, polynomial1)

result


# TEST CASE
# polynomial1 = rand(-100:100, rand(10:50))
# polynomial2 = rand(-100:100, rand(10:50))

# @test Array(GPUMultiply(polynomial1, polynomial2)) == Array(GPUSlowMultiply(polynomial1, polynomial2))