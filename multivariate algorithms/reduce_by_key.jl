using CUDA
using CUDA.CUSPARSE
using SparseArrays
include("utils.jl")
# keys = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# values = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# a = sparsevec(keys, values, 5, +)

# println(a)

degree = 5
numvars = 3

compositions = generate_compositions(degree, numvars)
println(compositions)
keys = zeros(Int32, size(compositions, 1))

function encode_composition(arr, maxValue)
    encoded = 0
    for i in eachindex(arr)
        encoded += arr[i] * maxValue ^ i
    end

    return encoded
end

for i in axes(compositions, 1)
    keys[i] = encode_composition(compositions[i, :], degree)
end

println(keys ./ 5)

keys = Int32[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
values = Int32[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

keys_out = Int32[1, 2, 3, 4, 5]
println(typeof(keys_out))
# 
# keys is encoded degrees of terms
# values is coefficients of terms 
# keys_out is encoded compositions vector. n degrees into k variables
function reduce_by_key!(keys, values, keys_out)
    reduced = sparsevec(keys_out, zeros(Int32, length(keys_out)))
    cu_reduced = CuSparseVector(reduced)

    cu_keys = CuArray(keys)
    cu_values = CuArray(values)

    nthreads = min(512, length(keys))
    nblocks = cld(length(keys), nthreads)

    # TODO make sure padded zeroes don't add anything, will probably index out of 
    # bounds in the kernel when actually called

    println(typeof(cu_keys))
    println(typeof(cu_values))
    println(typeof(cu_reduced))

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        reduce_by_key_kernel!(cu_keys, cu_values, cu_reduced)
    )

    return typeof(cu_reduced)
end

function reduce_by_key_kernel!(cu_keys, cu_values, cu_reduced)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    # a = cu_keys[idx]
    # @cuprintln(a)
    # b = cu_values[idx]
    @cuprintln(cu_reduced[1])
    # cu_reduced[cu_keys[idx]] += cu_values[idx]

    return
end

reduced_arr = reduce_by_key!(keys, values, keys_out)

println(reduced_arr)