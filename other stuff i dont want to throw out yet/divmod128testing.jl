using CUDA
using BenchmarkTools

@inline function faster_mod(x::T, m::T)::T where T<:Integer
    r = T(x - div(x, m) * m)
    return r < 0 ? r + m : r
end

function div(n::Int128, m::Int128) 
    if n == 0
        return Int128(0)
    end

    sign = 1
    if (n < 0) != (m < 0)
        sign = -1
    end

    n = abs(n)
    m = abs(m)

    quotient = Int128(0)
    remainder = Int128(0)

    for i in 0:127
        remainder = (remainder << 1) | ((n >> (127 - i)) & 1)
        if remainder >= m
            remainder -= m
            quotient |= (Int128(1) << (127 - i))
        end
    end

    return quotient * sign
end


function reduce_mod_m_kernel(arr::CuDeviceVector{Int128, 1}, m::Int128)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    arr[idx] = faster_mod(arr[idx], m)

    # using this one gives:
    # ERROR: LLVM error: Undefined external symbol "__modti3"
    # arr[idx] = arr[idx] % m
    return
end

function div_m_kernel(arr::CuDeviceVector{Int128, 1}, m::Int128)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    arr[idx] = div(arr[idx], m)

    return
end

function test_broken_mod()
    arr = CuArray(Int128.([10, 20, 30, 40]))
    m = Int128(7)

    # ERROR: LLVM error: Undefined external symbol "__divti3"
    @cuda(
        threads = length(arr),
        blocks = 1,
        reduce_mod_m_kernel(arr, m)
    )

    return arr
end

function test_broken_div()
    arr = CuArray(Int128.([10, 20, 30, 40]))
    m = Int128(7)

    # ERROR: LLVM error: Undefined external symbol "__divti3"
    @cuda(
        threads = length(arr),
        blocks = 1,
        div_m_kernel(arr, m)
    )

    return arr
end