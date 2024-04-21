"""
    num_ways_to_partition(n, k)

Count ways to partition n identical balls into
k identical boxes

(no closed form solution exists)
"""
function num_ways_to_partition(n, k)
    nums = zeros(Int, n + 1)
    nums[1] = 1

    for i in 1:k
        for j in i:n
            nums[j + 1] += nums[j - i + 1]
        end
    end

    return nums[n + 1]
end

println(num_ways_to_partition(11, 4))

"""
    generate_partitions(n, k)

Generate an array of all possible partitions of n 
identical balls into k boxes
"""
function generate_partitions(n, k)
    result = fill(Array{Int32}(undef, k), num_ways_to_partition(n, k))
    idx = 1
    for b in 1:k
        arr = vcat([n - b + 1], ones(Int32, b - 1))
        result[idx] = arr
        idx += 1

        flag = true
        while flag
            flag = kick_block_down(arr)

            if flag 
                result[idx] = arr
                idx += 1
            end
        end
    end
    return result
end

"""
    kick_block_down(arr)

Incremental step for generate_partitions.

If one partition is [4, 1], we "kick" one of the elements of the 
1st index, 4, down to the 2nd index to make [3, 2]. The condition for this is if
the value at some index is less than the first index by more than 1. This method
returns true if it was successfully able to kick a block down, and false if it
wasn't.

Another example is [4, 3, 3, 2] -> [3, 3, 3, 3]
"""
function kick_block_down(arr)
    for i in eachindex(arr)
        if arr[i] < arr[1] - 1
            arr[i] += 1
            arr[1] -= 1
            return true
        end
    end

    return false
end

n = 8
k = 3

println("Number of ways to partition $n items into $k boxes: $(num_ways_to_partition(8, 3))")
println(generate_partitions(8, 3))
