using CUDA

# Returns padded array to next power of 2
function pad_to_next_power_of_2_plus_1(arr::Vector{T}) where T
    current_length = length(arr)
    next_power_of_2 = 2^ceil(Int, log2(current_length))
    padding_length = next_power_of_2 - current_length + 1
    padded_array = vcat(arr, zeros(T, padding_length))
    
    return padded_array
end

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

# idk any better way to do this
function set_value(arr, idx, num)
    arr[idx] = num
    return
end

function generate_start_flags_kernel(keys, flags)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if keys[tid + 1] != keys[tid]
        flags[tid + 1] = true
    end

    return nothing
end

function reduce_by_key(keys::Array{T}, values::Array{V}) where {T, V<:Integer}
    @assert length(keys) == length(values) "Keys and values cannot be different lengths"
    cu_keys = CuArray(pad_to_next_power_of_2_plus_1(keys))
    cu_values = CuArray(pad_to_next_power_of_2_plus_1(values))
    flags = CUDA.fill(Int32(0), length(cu_keys))

    nthreads = min(512, length(cu_keys) - 1)
    nblocks = cld(length(cu_keys) - 1, nthreads)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_start_flags_kernel(cu_keys, flags)
    )

    CUDA.@sync @cuda threads = 1 blocks = 1 set_value(flags, 1, 1)
    
    key_indices = accumulate(+, flags)

    length_of_reduced_keys = Array(key_indices)[end]

    cu_seg_reduced = segmented_scan(cu_values, flags)

    reduced_keys = CUDA.zeros(T, length_of_reduced_keys)
    reduced_values = CUDA.zeros(V, length_of_reduced_keys)
    
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        reduce_by_key_kernel(flags, key_indices, cu_keys, reduced_keys, reduced_values, cu_seg_reduced)
    )

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

    CUDA.@sync @cuda threads = 1 blocks = 1 set_value(cu_data, n, 0)

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
    perm = sortperm(keys, rev = true)

    sorted_keys = keys[perm]
    sorted_values = values[perm]

    return sorted_keys, sorted_values
end

function reduce_arr(cu_keys, cu_values)
    sorted_keys, sorted_values = sort_keys_with_values(cu_keys, cu_values)
    reduced_keys, reduced_values = reduce_by_key(sorted_keys, sorted_values)

    return reduced_keys, reduced_values
end

#just threw this in here to avoid adding another import
function generate_compositions(n, k, type::DataType = Int32)
    compositions = zeros(type, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(type, k)
    current_composition[1] = n
    idx = 1
    while true
        compositions[idx, :] .= current_composition
        idx += 1
        v = current_composition[k]
        if v == n
            break
        end
        current_composition[k] = 0
        j = k - 1
        while 0 == current_composition[j]
            j -= 1
        end
        current_composition[j] -= 1
        current_composition[j + 1] = 1 + v
    end

    return compositions
end