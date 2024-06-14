function next_pow_2(n::Int)
    return 1 << ceil(Int, log2(n))
end

n = next_pow_2(42515280)