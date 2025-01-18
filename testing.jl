using CUDA

function run()
    arr = CuArray(rand(1:10, (8, 3)))

    ptr = pointer(arr)
    ptr += sizeof(eltype(arr)) * 8
    unsafeVec = CUDA.unsafe_wrap(CuVector{Int}, ptr, 8)

    display(arr)
    display(unsafeVec)
end

run()