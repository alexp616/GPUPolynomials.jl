using CUDA

include("utils.jl")

global VALUES_PER_THREAD = 4
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
keys = CuArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9999999])
values = CuArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9999999])
# values = CuArray([10, 20, 30, 40, 50, 10, 10, 20, 30, 10, 20, 30, 10, 10, 10, 10, 9999999])

# expected_flags = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
# expected_carryout = [100, 30, 0]
# expected_result_keys = [1, 2, 3, 4, 9999999]
# expected_result_values = [150, 10, 60, 60, 9999999]


# keys = CuArray(vcat(vcat([repeat([i], i) for i in 1:40]...), [999999 for _ in 1:77]))
# values = CuArray(vcat([1 for i in 1:820], [999999 for _ in 1:77]))


# Count is length of original keys array

# For parameter definitions, look in mgpuhost.cuh
# keys_global must be of length (multiple of VALUES_PER_THREAD * MAX_THREADS_PER_BLOCK) + 1 otherwise index out bounds
function ReduceByKey(keys_global::CuArray{Int},
                     values_global::CuArray{Int}, 
                     keys_dest_global::CuArray{Int}, 
                     values_dest_global::CuArray{Int})

    @assert (length(keys_global) == length(values_global)) "Keys and Values have different array lengths"
    count = length(keys_global)

    ReduceByKeyPreProcess(count, keys_global, keys_dest_global, )

end

function ReduceByKeyPreProcess(count::Int,
                               keys_global::CuArray{Int},
                               keys_dest_global::CuArray{Int},
                               count_host::Any, # for now
                               count_global::CuArray{Int})
    # TODO
    threadCodes_global = CUDA.zeros(Int32, total_threads)

    @assert ((count - 1) % VALUES_PER_THREAD == 0) "keys not padded correctly"
    total_threads = count / VALUES_PER_THREAD

    nthreads = min(length(keys) / 4, 512)
    nblocks = cld(total_threads, nthreads)
    NVperblock = nthreads * VALUES_PER_THREAD

    num_flags_per_thread = CUDA.zeros(Int32, total_threads)
    end_flags_encoded = CUDA.zeros(Int32, total_threads)

    # KernelReduceByKeyPreprocess start
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        GenerateFlagsKernel(keys_global, num_flags_per_thread, end_flags_encoded)
    )

    total_segments = sum(num_flags_per_thread)
    @assert (total_segments > 0) "No segments"

    num_warps = div(nthreads, 32)
    delta_shared = CUDA.zeros(Int32, total_threads)
    threadCodes_global = CUDA.zeros(Int32, total_threads)

    storage = CUDA.zeros(Int32, total_threads)
    CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        FindSegScanDeltaAndThreadCodes(num_flags_per_thread, delta_shared, num_warps, threadCodes_global, storage, nthreads)
    )
    # KernelReduceByKeyPreprocess end 

    
    
end

function Scan(tid::Int, x::Int, storage::CuArray{Int32}, nthreads, blockstartidx, )
    temp = x
    storage[tid] = temp
    first = 0;
    CUDA.sync_threads()

    offset = 1
    while (offset < nthreads)
        if (tid >= blockstartidx + offset)
            temp += storage[first + tid - offset + 1]
        end
        first = nthreads - first
        storage[first + tid] = temp
        offset *= 2
        CUDA.sync_threads()
    end

    return temp
end

function FindSegScanDeltaAndThreadCodesKernel(num_flags_per_thread::CuArray{Int32}, delta_shared, num_warps, threadCodes_gloal, storage, nthreads)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    flag = num_flags_per_thread[tid + 1] != 0

    scan = Scan(tid, num_flags_per_thread[tid + 1], storage, nthreads, (blockIdx().x - 1) * nthreads)

    delta_shared[tid + 1] = FindSegScanDelta(tid, flag, delta_shared, num_warps)
    
    CUDA.sync_threads()
    threadCodes = Int32(endFlags | (delta_shared << 13) | (scan << 20))
    threadCodes_global[tid + 1] = threadCodes
    return
end

function FindSegScanDelta(tid::Int, flag::Boolean, delta_shared::CuArray{Int32}, num_warps::Int)
    warp = Int32(div(tid, 32))
    lane = Int32(31 & tid)
    warpMask = UInt32(0xffffffff >> (31 - lane))
    ctaMask = UInt32(0x7fffffff >> (31 - lane))

    warpBits = vote_ballot_sync(0xffffffff, flag)
    # @cuprintln("warpBits: $warpBits, warpMask: $warpMask")
    delta_shared[warp + 1] = warpBits
    CUDA.sync_threads()

    if (tid < num_warps) 
        ctaBits = UInt32(vote_ballot_sync(0xffffffff, 0 != delta_shared[tid + 1]))
        warpSegment = Int32(31 - leading_zeros(ctaMask & ctaBits));
        start = (-1 != warpSegment) ?
            (31 - leading_zeros(delta_shared[warpSegment + 1]) + 32 * warpSegment) : 0
        delta_shared[num_warps + tid + 1] = start
    end
    CUDA.sync_threads()

    start = 31 - leading_zeros(warpMask & warpBits)
    # @cuprintln("start before: $start")
    if (-1 != start)
        start += ~31 & tid
    else
        start = delta_shared[num_warps + warp + 1]
    end

    # @cuprintln("tid: $tid, start: $start")
    return Int32(tid - start)
end

# Generates segment end flags, counts number of flags per thread, and encodes where they are in end_flags_encoded
function GenerateFlagsKernel(keys_global::CuArray{Int}, 
                            num_flags_per_thread::CuArray{Int},
                            end_flags_encoded::CuArray{Int32},)

    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1

    end_flags = Int32(0)
    startindex = tid * VALUES_PER_THREAD + 1

    for i in startindex:startindex + VALUES_PER_THREAD - 1
        if (keys_global[i] != keys_global[i + 1])
            end_flags |= 1 << (i - startindex)
        end
    end

    num_flags_per_thread[tid + 1] = count_ones(end_flags)
    end_flags_encoded[tid + 1] = end_flags

    return nothing
end



















# function GenerateFlagsKernel(keys_global::CuArray{Int}, 
#                             count::Int,
#                             NV::Int,
#                             num_flags_per_thread::CuArray{Int},
#                             end_flags_encoded::CuArray{Int32},
#                             )

#     tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
#     block = blockIdx().x - 1
#     # idk what gid means
#     gid = NV * block
#     count2 = min(NV + 1, count - gid)

#     end_flags = Int32(0)
#     startindex = tid * VALUES_PER_THREAD + 1
#     if (count2 > NV)
#         for i in startindex:startindex + VALUES_PER_THREAD - 1
#             if (keys_global[i] != keys_global[i + 1])
#                 end_flags |= 1 << (i - 1)
#             end
#         end
#     else
#         for i in startindex:startindex + VALUES_PER_THREAD - 1
#             index = VALUES_PER_THREAD * tid + 1 + i
#             if (index == count2 || (index < count2 && (keys_global[i] != keys_global[i + 1])))
#                 end_flags |= 1 << (i - 1)
#             end
#         end
#     end

#     num_flags_per_thread[tid + 1] = count_ones(end_flags)
#     end_flags_encoded[tid + 1] = end_flags

#     return nothing
# end