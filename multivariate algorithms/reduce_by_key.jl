using CUDA

include("utils.jl")
# keys = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# values = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
# 
# function encode_composition(arr, maxValue)
#     encoded = 0
#     for i in eachindex(arr)
#         encoded += arr[i] * maxValue ^ i
#     end

#     return encoded
# end

# for i in axes(compositions, 1)
#     keys[i] = encode_composition(compositions[i, :], degree)
# end

# need to add dummy row to end to avoid index out of bounds
keys = CuArray([1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 9999999])
values = CuArray([10, 20, 30, 40, 50, 10, 10, 20, 30, 10, 20, 30, 9999999])

expected_flags = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
expected_carryout = [100, 30, 0]
expected_result_keys = [1, 2, 3, 4, 9999999]
expected_result_values = [150, 10, 60, 60, 9999999]

"""
generate_flags_kernel!() generates segment end flags in segment_flags, as well as marks which
thread indices have a segment end in thread_level_segment_end_flags.
"""
function generate_flags_kernel!(keys, segment_flags, thread_level_segment_end_flags, VALUES_PER_THREAD)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    # This and the for loop designate which values the thread covers
    startindex = (tid - 1) * VALUES_PER_THREAD + 1
    
    for i in startindex:startindex + VALUES_PER_THREAD - 1
        if (keys[i] != keys[i + 1])
            # Mark segment end
            segment_flags[i] = 1

            # Mark that this thread contains a segment end
            thread_level_segment_end_flags[tid] = 1
        end
    end
    return
end

# This thing only works when keys and values are padded to length (multiple of VALUES_PER_THREAD * MAX_THREADS_PER_BLOCK) + 1
function seg_reduce(keys, values)
    VALUES_PER_THREAD = 4
    total_threads = 0

    # worry about the alternative later
    if ((length(keys) - 1) % VALUES_PER_THREAD == 0)
        total_threads = ÷((length(keys) - 1), VALUES_PER_THREAD) 
    end

    nthreads = min(total_threads, 128)
    nblocks = cld(total_threads, nthreads)

    segment_flags = CUDA.zeros(UInt16, length(keys))
    thread_level_segment_end_flags = CUDA.zeros(UInt16, total_threads)

    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_flags_kernel!(keys, segment_flags, thread_level_segment_end_flags, VALUES_PER_THREAD)
    )
    
    thread_level_carryout = CUDA.zeros(Int32, total_threads)
    
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        reduce_kernel_1!(values, segment_flags, thread_level_carryout, VALUES_PER_THREAD)
    )

    return (values, thread_level_carryout)
end

# This thing is "applied" to each block
# numWarps = MAX_THREADS_PER_BLOCK / 32
function DeviceFindSegScanDelta(tid, flag, delta_shared, numWarps)
    warp = div(tid, 32)
    lane = 31 & tid
    warpMask = UInt32(0xffffffff >> (31 - lane))
    ctaMask = UInt32(0x7fffffff >> (31 - lane))
    warpBits = vote_ballot_sync(0xffffffff, flag != 0)
    delta_shared[warp] = warpBits

    CUDA.sync_threads()

    if (tid < numWarps) 
        ctaBits = vote_ballot_sync(0xffffffff, 0 != delta_shared[tid])
        warpsegment = 31 - leading_zeros(ctaMask & ctaBits);
        start = (-1 != warpSegment) ? (31 - leading_zeros(delta_shared[warpSegment]) + 32 * warpsegment) : 0
        delta_shared[numWarps] = start
    end

    CUDA.sync_threads()

    start = 31 - leading_zeros(warpMask & warpBits)
    if (-1 != start)
        start += ~31 & tid
    else
        start = delta_shared[numWarps + warp]
    end

    CUDA.sync_threads()

    return tid - start
end


function reduce_kernel_1!(values, segment_flags, thread_level_carryout, VALUES_PER_THREAD)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    startindex = (tid - 1) * VALUES_PER_THREAD + 1
    accumulator = 0

    for i in startindex:startindex + VALUES_PER_THREAD - 1
        accumulator += values[i]
        if (segment_flags[i] != 0x0000)
            values[i] = accumulator
            accumulator = 0
        end
    end

    thread_level_carryout[tid] = accumulator
    return
end


result = seg_reduce(keys, values)

println(result[1])
println(result[2])




# keys = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
# values = [10, 20, 30, 40, 50, 10, 10, 20, 30, 10, 20, 30, 40, 50, 60, 70]
# flags = BitArray([0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])


# function mystery_segmented_scan(values, flags)
#     n = length(flags)
#     for d in 0:Int32(round(log2(n))) - 1
#         for k in 0:2 ^ (d + 1):n - 1
            
#             if flags[k + 2 ^ (d + 1)] == 0
#                 values[k + 2 ^ (d + 1)] = values[k + 2 ^ (d)] + values[k + 2 ^ (d + 1)]
#             end
#             flags[k + 2 ^ (d + 1)] = flags[k + 2 ^ d] || flags[k + 2 ^ (d + 1)]
#         end
#     end
# end

# mystery_segmented_scan(values, flags)

# println(values)
# println(flags)
