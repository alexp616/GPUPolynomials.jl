using Test
using BenchmarkTools

include("CPUUnivariateAlgorithms.jl")
include("GPUUnivariateAlgorithms.jl")


"""
Some benchmarks for different algorithms

Generally, the fancy algorithms like it when the degrees of the two polynomials
are close, which will happen a lot with squaring polynomials

The fanciest GPU algorithm only starts to beat the fancy CPU algorithm at squaring
at around degree 2^13

These results aren't final - both the fancy CPU and GPU algorithm have fine-tuning
that can be done, but it probably won't drastically change the performance.

The GPU algorithm also has to deal with converting its inputs to a different datatype,
which depending on the user doesn't have to happen.
"""


# # 33554432 x 33554432

# println("33554432 x 33554432:")
# p1 = [1 for i in 1:2^25]
# p2 = [1 for i in 1:2^25]
# # @test CPUMultiply(p1, p2) == Array(GPUMultiply(p1, p2))
# # print("CPUMultiply: ") # 
# # @btime CPUMultiply(p1, p2)  
# print("GPUMultiply: ") # 
# @btime GPUMultiply(p1, p2)
# println()


# 32768 x 32768

println("531441 x 531441:")
p1 = [1 for i in 1:531441]
p2 = [1 for i in 1:531441]
# @test CPUSlowMultiply(p1, p2) == CPUMultiply(p1, p2) == Array(GPUSlowMultiply(p1, p2)) == Array(GPUMultiply(p1, p2))
# print("CPUSlowMultiply: ") # 208 ms
# @btime CPUSlowMultiply(p1, p2)
# print("CPUMultiply: ") # 13 ms
# @btime CPUMultiply(p1, p2)  
# print("GPUSlowMultiply: ") # 53 ms
# @btime GPUSlowMultiply(p1, p2)
print("GPUMultiply: ") # 6 ms
@btime GPUMultiply(p1, p2)
println()


# 50 x 50

# println("50 x 50:")
# p1 = [1 for i in 1:51]
# p2 = [1 for i in 1:51]
# @test CPUSlowMultiply(p1, p2) == CPUMultiply(p1, p2) == Array(GPUSlowMultiply(p1, p2)) == Array(GPUMultiply(p1, p2))
# print("CPUSlowMultiply: ") # 0.6 μs
# @btime CPUSlowMultiply(p1, p2)
# print("CPUMultiply: ") # 6 μs
# @btime CPUMultiply(p1, p2)  
# print("GPUSlowMultiply: ") # 197 μs
# @btime GPUSlowMultiply(p1, p2)
# print("GPUMultiply: ") # 1911 μs
# @btime GPUMultiply(p1, p2)
# println()


# # 8191 x 8191

# println("8191 x 8191:")
# p1 = [1 for i in 1:8192]
# p2 = [1 for i in 1:8192]
# @test CPUSlowMultiply(p1, p2) == CPUMultiply(p1, p2) == Array(GPUSlowMultiply(p1, p2)) == Array(GPUMultiply(p1, p2))
# print("CPUSlowMultiply: ") # 11973 μs
# @btime CPUSlowMultiply(p1, p2)
# print("CPUMultiply: ") # 1814 μs
# @btime CPUMultiply(p1, p2)  
# print("GPUSlowMultiply: ") # 3129 μs
# @btime GPUSlowMultiply(p1, p2)
# print("GPUMultiply: ") # 3176 μs
# @btime GPUMultiply(p1, p2)
# println()


# # 4 x 9999

# println("4 x 9999:")
# p1 = [1 for i in 1:5]
# p2 = [1 for i in 1:10000]

# @test CPUSlowMultiply(p1, p2) == CPUMultiply(p1, p2) == Array(GPUSlowMultiply(p1, p2)) == Array(GPUMultiply(p1, p2))
# print("CPUSlowMultiply: ") # 15 μs
# @btime CPUSlowMultiply(p1, p2)
# print("CPUMultiply: ") # 1728 μs
# @btime CPUMultiply(p1, p2)  
# print("GPUSlowMultiply: ") # 181 μs
# @btime GPUSlowMultiply(p1, p2)
# print("GPUMultiply: ") # 3155 μs
# @btime GPUMultiply(p1, p2)
# println()


# # 1023 x 1023

# println("1023 x 1023:")
# p1 = [1 for i in 1:1024]
# p2 = [1 for i in 1:1024]

# @test CPUSlowMultiply(p1, p2) == CPUMultiply(p1, p2) == Array(GPUSlowMultiply(p1, p2)) == Array(GPUMultiply(p1, p2))
# print("CPUSlowMultiply: ") # 171 μs
# @btime CPUSlowMultiply(p1, p2)
# print("CPUMultiply: ") # 113 μs
# @btime CPUMultiply(p1, p2)  
# print("GPUSlowMultiply: ") # 308 μs
# @btime GPUSlowMultiply(p1, p2)
# print("GPUMultiply: ") # 2332 μs
# @btime GPUMultiply(p1, p2)
# println()

