using CUDA
using BenchmarkTools
using Dates

# firstarr = CuArray(rand(1:100, 1000))
# @time sort!(firstarr)


@time sort(rand(1:100, 100000000))

# arr2 = CuArray(rand(1:100, 100000000))
# CUDA.@time sort!(arr2)


# CUDA.unsafe_free!(arr2)

# arr = CuArray(rand(1:100, 100000000))
# CUDA.@profile sort!(arr)

# CUDA.unsafe_free!(arr)

# arr3 = CuArray(rand(1:100, 100000000))
# start_time = time_ns()
# sort!(arr3)
# end_time = time_ns()
# elapsed_time_ns = end_time - start_time
# elapsed_time_s = elapsed_time_ns / 1e9
# println("Elapsed time: ", elapsed_time_s, " seconds")


# keys = CuArray([2, 3, 1, 5, 4])
# values = CuArray([10, 20, 30, 40, 50])
# sortperm!(values, keys; initialized = true)
# sort!(keys)
# println(keys)
# println(values)