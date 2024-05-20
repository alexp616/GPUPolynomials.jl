using CUDA
using BenchmarkTools
array = CuArray([1, 0, 0, 0, 1, 0, 1, 1])

sus = accumulate(+, array)
println(sus)
println(array)