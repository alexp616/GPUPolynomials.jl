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
# keys = CuArray([1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 9999999])
# keys = CuArray(vcat([i for i in 1:20], 9999999))
# values = CuArray(vcat([1 for _ in 1:20], 9999999))
# keys = CuArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9999999])
# values = CuArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9999999])
# values = CuArray([10, 20, 30, 40, 50, 10, 10, 20, 30, 10, 20, 30, 10, 10, 10, 10, 9999999])

# expected_flags = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
# expected_carryout = [100, 30, 0]
# expected_result_keys = [1, 2, 3, 4, 9999999]
# expected_result_values = [150, 10, 60, 60, 9999999]


keys = CuArray(vcat(vcat([repeat([i], i) for i in 1:40]...), [999999 for _ in 1:77]))
values = CuArray(vcat([1 for i in 1:820], [999999 for _ in 1:77]))


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
            segment_flags[i] = true

            # Mark that this thread contains a segment end
            thread_level_segment_end_flags[tid] = true
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
    else 
        throw("padding issue")
    end

    if (length(keys) != length(values))
        throw("keys not same length as values")
    end

    nthreads = min(total_threads, 512)
    nblocks = cld(total_threads, nthreads)

    # segment_flags = CUDA.zeros(UInt16, length(keys))
    # thread_level_segment_end_flags = CUDA.zeros(UInt16, total_threads)

    segment_flags = CUDA.fill(false, length(keys))
    thread_level_segment_end_flags = CUDA.fill(false, total_threads)

    # Generate segment end flags and flags for which threads have segment ends in them
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        generate_flags_kernel!(keys, segment_flags, thread_level_segment_end_flags, VALUES_PER_THREAD)
    )
    
    println("segment_flags: $(segment_flags)")
    println("thread_level_segment_end_flags: $(thread_level_segment_end_flags)")

    thread_level_carryout = CUDA.zeros(Int32, total_threads)
    delta_shared = CUDA.zeros(UInt32, total_threads)

    # Reduce segments contained in threads and Generate carry-out values
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        reduce_kernel_1!(values, segment_flags, thread_level_carryout, VALUES_PER_THREAD)
    )

    println("thread_level_carryout: $(thread_level_carryout)")

    num_warps = div(nthreads, 32)

    println(num_warps)
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        find_tid_delta_helper(thread_level_segment_end_flags, delta_shared, num_warps)
    )

    println("delta_shared: $(delta_shared)")

    # CUDA.@sync @cuda(
    #     threads = nthreads,
    #     blocks = nblocks,
    #     seg_scan_delta_helper()
    # )

    return 
end

function find_tid_delta_helper(thread_level_segment_end_flags, delta_shared, num_warps)
    tid = Int32(threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1)
    flag = (thread_level_segment_end_flags[tid + 1])

    delta_shared[tid + 1] = find_tid_delta(tid, flag, delta_shared, num_warps)

    return
end

# numWarps = MAX_THREADS_PER_BLOCK / 32
function find_tid_delta(tid::Int32, flag::Bool, delta_shared, num_warps, warp_words_temp)
    warp = Int32(div(tid, 32))
    lane = Int32(31 & tid)
    warpMask = UInt32(0xffffffff >> (31 - lane))
    ctaMask = UInt32(0x7fffffff >> (31 - lane))

    warpBits = vote_ballot_sync(0xffffffff, flag)
    # @cuprintln(typeof(warpBits))
    # @cuprintln("warpBits: $warpBits, warpMask: $warpMask")
    delta_shared[warp + 1] = warpBits
    flag2 = 0 != delta_shared[tid + 1]

    # pseudocode, worry about index out of bounds later
    warp_words_temp[warp + 1] = vote_ballot_sync(0xffffffff, flag)
    CUDA.sync_threads()

    if (tid < num_warps) 
        ctaBits = vote_ballot_sync(0xffffffff, flag2)
        warpSegment = Int32(31 - leading_zeros(ctaMask & ctaBits));
        start = (-1 != warpSegment) ?
            (31 - leading_zeros(delta_shared[warpSegment + 1]) + 32 * warpSegment) : 0
        delta_shared[num_warps + tid + 1] = start
    end
    CUDA.sync_threads()

    # start = 31 - leading_zeros(warpMask & warpBits)
    # # @cuprintln("start before: $start")
    # if (-1 != start)
    #     start += ~31 & tid
    # else
    #     start = delta_shared[num_warps + warp + 1]
    # end
    # CUDA.sync_threads()

    # @cuprintln("tid: $tid, start: $start")
    # return tid - start
    return Int32(0)
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
