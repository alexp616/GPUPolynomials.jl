using CUDA
using BenchmarkTools

cuarr = CUDA.fill(2, 1000000)
@btime cpuarr = Array(cuarr) # 2.515 ms (laptop not plugged in)

cpuarr2 = [i for i in 1:1000000]
@btime cuarr2 = CuArray(cpuarr2) # 1.664 ms (laptop not plugged in)