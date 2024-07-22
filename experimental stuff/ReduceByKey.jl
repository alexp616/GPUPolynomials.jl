using CUDA
using BenchmarkTools

function segmented_scan_upsweep_kernel(data, flags_tmp, d)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = i * 2^(d + 1)

    if flags_tmp[k + 2^(d + 1)] == 0
        data[k + 2^(d + 1)] = data[k + 2^(d)] + data[k + 2^(d + 1)]
    end

    flags_tmp[k + 2^(d + 1)] = flags_tmp[k + 2^d] | flags_tmp[k + 2^(d + 1)]

    return nothing
end

function segmented_scan_downsweep_kernel(data, flags_original, flags_tmp, d)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    k = i * 2^(d + 1)

    temp = data[k + 2^d]
    data[k + 2^d] = data[k + 2^(d + 1)]
    if (flags_original[k + 2^d + 1] != 0)
        data[k + 2^(d + 1)] = 0
    elseif (flags_tmp[k + 2^d] != 0)
        data[k + 2^(d + 1)] = temp
    else
        data[k + 2^(d + 1)] += temp
    end
    flags_tmp[k + 2^d] = 0

    return nothing
end

function generate_start_flags_kernel(keys, flags)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if keys[tid + 1] != keys[tid]
        flags[tid + 1] = true
    end

    return nothing
end

# function reduce_by_key(cu_keys::CuArray{T}, cu_values::CuArray{V}) where {T, V<:Integer}
function reduce_by_key(cu_keys::CuVector{K}, cu_values::CuVector{V}) where {K, V<:Real}
    @assert length(cu_keys) == length(cu_values) "Keys and values cannot be different lengths"
    if !ispow2(length(cu_keys) - 1)
        len = nextpow(2, length(cu_keys)) + 1
        cu_keys = vcat(cu_keys, CUDA.zeros(K, len))
        cu_values = vcat(cu_values, CUDA.zeros(V, len))
    end

    # Flags must be type Int32 because we run a prefix scan on it to get result indices
    flags = CUDA.fill(Int32(0), length(cu_keys))

    flagskernel = @cuda launch=false generate_start_flags_kernel(cu_keys, flags)
    flagsconfig = launch_configuration(flagskernel.fun)
    threads = min(length(cu_keys) - 1, prevpow(2, flagsconfig.threads))
    blocks = cld(length(cu_keys) - 1, threads)

    flagskernel(cu_keys, flags; threads = threads, blocks = blocks)

    CUDA.@allowscalar flags[1] = 1
    
    key_indices = accumulate(+, flags)

    CUDA.@allowscalar length_of_reduced_keys = key_indices[end]

    cu_seg_reduced = segmented_scan(cu_values, flags)

    reduced_keys = CUDA.zeros(K, length_of_reduced_keys)
    reduced_values = CUDA.zeros(V, length_of_reduced_keys)
    
    reducebykeykernel = @cuda launch=false reduce_by_key_kernel(flags, key_indices, cu_keys, reduced_keys, reduced_values, cu_seg_reduced)
    reduceconfig = launch_configuration(reducebykeykernel.fun)
    threads = min(length(cu_keys) - 1, prevpow(2, reduceconfig.threads))
    blocks = cld(length(cu_keys) - 1, threads)

    reducebykeykernel(flags, key_indices, cu_keys, reduced_keys, reduced_values, cu_seg_reduced; threads = threads, blocks = blocks)

    return reduced_keys[1:end-1], reduced_values[1:end-1]
end

function reduce_by_key_kernel(flags, key_indices, cu_keys, reduced_keys, reduced_values, cu_seg_reduced)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if flags[tid + 1] != 0
        reduced_keys[key_indices[tid]] = cu_keys[tid]
        reduced_values[key_indices[tid]] = cu_seg_reduced[tid]
    end

    return nothing
end

function segmented_scan(cu_data, cu_flags_original)
    @assert length(cu_data) == length(cu_flags_original) "data and flags not same length"
    copy_original_data = copy(cu_data)
    cu_flags_tmp = copy(cu_flags_original)
    n = length(cu_data) - 1
    log2n = Int(floor(log2(n)))

    for d in 0:(log2n - 1)
        total_threads = div(length(cu_data), 2^(d + 1))
        nthreads = min(total_threads, 512)
        nblocks = cld(total_threads, nthreads)

        CUDA.@sync @cuda(
            threads = nthreads,
            blocks = nblocks,
            segmented_scan_upsweep_kernel(cu_data, cu_flags_tmp, d)
        )
    end

    CUDA.@allowscalar cu_data[n] = 0

    for d in (log2n - 1):-1:0
        total_threads = div(length(cu_data), 2^(d + 1))
        nthreads = min(total_threads, 512)
        nblocks = cld(total_threads, nthreads)

        CUDA.@sync @cuda(
            threads = nthreads,
            blocks = nblocks,
            segmented_scan_downsweep_kernel(cu_data, cu_flags_original, cu_flags_tmp, d)
        )
    end

    cu_data .+= copy_original_data

    CUDA.unsafe_free!(copy_original_data)
    CUDA.unsafe_free!(cu_flags_tmp)
    return cu_data
end

function sort_keys_with_values(keys, values)
    perm = sortperm(keys)

    sorted_keys = keys[perm]
    sorted_values = values[perm]

    return sorted_keys, sorted_values
end



function reduce_arr(cu_keys, cu_values)
    sorted_keys, sorted_values = sort_keys_with_values(cu_keys, cu_values)
    CUDA.unsafe_free!(cu_keys)
    CUDA.unsafe_free!(cu_values)

    reduced_keys, reduced_values = reduce_by_key(sorted_keys, sorted_values)
    CUDA.unsafe_free!(sorted_keys)
    CUDA.unsafe_free!(sorted_values)

    return reduced_keys, reduced_values
end



