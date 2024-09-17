include("../src/algorithms/cpu/cpu_ntt_pow.jl")
# include("../src/algorithms/gpu/gpu_ntt_pow.jl")

module NTTPowTests

using ..CPUNTTPow

using Test

function run_tests()
    test_cpu_ntt_pow()
end

function test_cpu_ntt_pow()
    a = ones(Int, 4)
    
    pregen = pregen_cpu_pow([65537], get_fft_size(a, 2))
    result = cpu_ntt_pow(a, 2; pregen = pregen)

    @test result == [1, 2, 3, 4, 3, 2, 1]

    a = fill(UInt(2), 4)

    pregen = pregen_cpu_pow(UInt.([65537]), get_fft_size(a, 2))
    result = cpu_ntt_pow(a, 2; pregen = pregen)

    @test result == [4, 8, 12, 16, 12, 8, 4]

    a = fill(UInt(4), 100)

    pregen1 = pregen_cpu_pow(UInt.([4294957057]), get_fft_size(a, 3))
    pregen2 = pregen_cpu_pow(UInt.([7681, 10753, 11777]), get_fft_size(a, 3))
    result1 = cpu_ntt_pow(a, 2; pregen = pregen1)
    result2 = cpu_ntt_pow(a, 2; pregen = pregen2)

    println("result1 type: $(eltype(result1)), result2 type: $(eltype(result2))")
    @test result1 == result2
end

end

using .NTTPowTests

NTTPowTests.run_tests()