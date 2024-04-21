using CUDA
CUDA.allowscalar(false)


function trivialMultiply(p1, p2)
    result = Dict{Vector{Int}, Int}()
    
    for (coeff1, exp1) in p1
        for (coeff2, exp2) in p2
            coeff = coeff1 * coeff2
            exp = exp1 .+ exp2
            if haskey(result, exp)
                result[exp] += coeff
            else
                result[exp] = coeff
            end
        end
    end
    
    combined_result = [(coeff, exp) for (exp, coeff) in result]
    return combined_result
end

p1 = [(1, [1, 0, 0]), (2, [0, 1, 0]), (3, [0, 0, 1])]
p2 = [(1, [2, 0, 0]), (2, [0, 2, 0]), (3, [0, 0, 2])]

result = trivialMultiply(p1, p2)
println(result)


function GPUtrivialMultiply!(p1, p2)
    result = CUDA.fill(Tuple, length(p1) * length(p2))

    if !(p1 isa CuArray)
        p1 = CuArray(p1)
    end
    if !(p2 isa CuArray)
        p2 = CuArray(p2)
    end
    
    nthreads = min(CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    ), length(p1)*length(p2))

    nblocks = cld(length(p1) * length(p2), nthreads)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        GPUtrivialMultiplyKernel!(result, p1, p2)
    )
    
    return result
end

function GPUtrivialMultiplyKernel!(result, p1, p2)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    idx1 = floor(idx / length(p2))
    idx2 = idx - length(p2) * idx1 # idx2 = idx % length(p2)

    result[idx] = (p1[idx1][1] * p2[idx2][1], p1[idx1][2] .+ p2[idx2][2])

    return 
end



p1 = [(1, [1, 0, 0]), (2, [0, 1, 0]), (3, [0, 0, 1])]
p2 = [(1, [2, 0, 0]), (2, [0, 2, 0]), (3, [0, 0, 2])]

result = Array(GPUtrivialMultiply!(p1, p2))

println(result)