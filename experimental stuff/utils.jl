using CUDA

function faster_mod(x, m)
    return x - div(x, m) * m
end

function reduce_mod_m(arr, m)
    CUDA.@sync arr .= faster_mod.(arr, m)
end

function next_pow_2(n::Int)
    return 1 << ceil(Int, log2(n))
end

function is_pow_2(n::Int)
    return n & (n - 1) == 0
end

function get_last_element(arr)
    return Array(arr)[end]
end