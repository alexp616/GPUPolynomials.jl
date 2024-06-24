using CUDA
using BenchmarkTools
function faster_mod(x, m)
    return x - div(x, m) * m
end

a = CuArray(Int128.([i for i in 1:10000]))

println(typeof(a))

@btime CUDA.@sync a .= faster_mod.(a, 3)
@btime CUDA.@sync a .= a .% 3
