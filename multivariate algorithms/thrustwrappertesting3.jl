using CUDA




keys = CuArray([1, 1, 2, 2, 3])
values = CuArray([10, 20, 30, 40, 50])
keys_output = CuArray(Int, length(keys))
values_output = CuArray(Int, length(values))

