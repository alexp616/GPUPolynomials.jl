include("../src/algorithms/cpu/cpu_ntt_pow.jl")
include("../src/algorithms/gpu/gpu_ntt_pow.jl")

module NTTPowTests

using Test
using CUDA

using ..CPUNTTPow
using ..GPUNTTPow

function run_tests()
    test_cpu_ntt_pow()
    test_gpu_ntt_pow()
end

function test_cpu_ntt_pow()
    a = ones(Int, 4)
    
    pregen = CPUNTTPow.pregen_cpu_pow([65537], CPUNTTPow.get_fft_size(a, 2))
    result = CPUNTTPow.cpu_ntt_pow(a, 2; pregen = pregen)

    @test result == [1, 2, 3, 4, 3, 2, 1]

    a = fill(UInt(2), 4)

    pregen = CPUNTTPow.pregen_cpu_pow(UInt.([65537]), CPUNTTPow.get_fft_size(a, 2))
    result = CPUNTTPow.cpu_ntt_pow(a, 2; pregen = pregen)

    @test result == [4, 8, 12, 16, 12, 8, 4]

    a = fill(UInt(4), 100)

    pregen1 = CPUNTTPow.pregen_cpu_pow(UInt.([4294957057]), CPUNTTPow.get_fft_size(a, 3))
    pregen2 = CPUNTTPow.pregen_cpu_pow(UInt.([7681, 10753, 11777]), CPUNTTPow.get_fft_size(a, 3))
    result1 = CPUNTTPow.cpu_ntt_pow(a, 2; pregen = pregen1)
    result2 = CPUNTTPow.cpu_ntt_pow(a, 2; pregen = pregen2)

    println("result1 type: $(eltype(result1)), result2 type: $(eltype(result2))")
    @test result1 == result2
end

function test_gpu_ntt_pow()
    a = CuArray(ones(Int, 4))
    
    pregen = GPUNTTPow.pregen_gpu_pow([65537], GPUNTTPow.get_fft_size(a, 2))
    result = GPUNTTPow.gpu_ntt_pow(a, 2; pregen = pregen)

    @test Array(result) == [1, 2, 3, 4, 3, 2, 1]

    a = CuArray(fill(UInt(2), 4))

    pregen = GPUNTTPow.pregen_gpu_pow(UInt.([65537]), GPUNTTPow.get_fft_size(a, 2))
    result = GPUNTTPow.gpu_ntt_pow(a, 2; pregen = pregen)

    @test Array(result) == [4, 8, 12, 16, 12, 8, 4]

    a = CuArray(fill(UInt(4), 100))

    pregen1 = GPUNTTPow.pregen_gpu_pow(UInt.([4294957057]), GPUNTTPow.get_fft_size(a, 3))
    pregen2 = GPUNTTPow.pregen_gpu_pow(UInt.([7681, 10753, 11777]), GPUNTTPow.get_fft_size(a, 3))
    result1 = GPUNTTPow.gpu_ntt_pow(a, 2; pregen = pregen1)
    result2 = GPUNTTPow.gpu_ntt_pow(a, 2; pregen = pregen2)

    println("result1 type: $(eltype(result1)), result2 type: $(eltype(result2))")
    @test result1 == result2
end

end

using .NTTPowTests

NTTPowTests.run_tests()