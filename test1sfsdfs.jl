using Test
using BenchmarkTools

function generate_partitions(n, k)
    partitions = zeros(Int32, binomial(n + k - 1, k - 1), k)  # Initialize 2D array to store partitions
    idx = 1
    stack = [(n, k, [])]

    while !isempty(stack)
        n, k, current_partition = pop!(stack)
        if n == 0 && k == 0
            partitions[idx, :] = current_partition
            idx += 1
        elseif n >= 0 && k > 0
            for i in 0:n
                new_partition = copy(current_partition)
                push!(new_partition, i)
                push!(stack, (n - i, k - 1, new_partition))
            end
        end
    end

    return partitions
end

function generate_partitions2(n, k)
    partitions = zeros(Int32, binomial(n + k - 1, k - 1), k)
    current_partition = zeros(Int32, k)
    current_partition[1] = n
    idx = 1
    while true
        partitions[idx, :] .= current_partition
        idx += 1
        v = current_partition[k]
        if v == n
            break
        end
        current_partition[k] = 0
        j = k - 1
        while 0 == current_partition[j]
            j -= 1
        end
        current_partition[j] -= 1
        current_partition[j + 1] = 1 + v
    end

    return partitions
end


@btime generate_partitions(9, 20)
@btime generate_partitions2(9, 20)