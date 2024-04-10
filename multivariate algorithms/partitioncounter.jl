# """
# Counts ways to partition n identical balls into
# k identical boxes using some dynamic programming
# (no closed form solution exists)
# """

function partition(n, k)
    nums = zeros(Int, n + 1)
    nums[1] = 1

    for i in 1:k
        for j in i:n
            nums[j + 1] += nums[j - i + 1]
        end
    end

    return nums[n + 1]
end

n = 8
k = 4
ways = partition(n, k)
print("Number of ways: $ways")