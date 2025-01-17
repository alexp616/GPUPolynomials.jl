using CUDA

function run()
    arr = CuArray(rand(1:10, (8, 3)))

    ptr = pointer()

    display(arr)
end

run()