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

    a = ones(UInt, 4)

    pregen = pregen_cpu_pow(UInt.([65537]), get_fft_size(a, 2))
    result = cpu_ntt_pow(a, 2; pregen = pregen)

    @test result == [1, 2, 3, 4, 3, 2, 1]
end

end

using .NTTPowTests

NTTPowTests.run_tests()