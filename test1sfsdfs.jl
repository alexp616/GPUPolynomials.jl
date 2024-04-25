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

println(generate_partitions(5, 3))