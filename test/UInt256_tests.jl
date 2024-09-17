include("../src/utils/UInt256.jl")

module UInt256Tests

using ..UInt256Module

using Test
using CUDA

function run_tests()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_mod()
end

function test_add()
    # trials = 1000000
    addtime = @timed begin
        for _ in 1:trials
            a = rand(1:BigInt(2)^255 - 1)
            b = rand(1:BigInt(2)^255 - 1)
            @assert BigInt(UInt256(a) + UInt256(b)) == a + b
        end
    end
    # println("Time for $trials additions: $(addtime.time)")
    # test passed
    @test true
end

function test_sub()
    # trials = 1000000
    subtime = @timed begin
        for _ in 1:trials
            a = rand(1:BigInt(2)^255 - 1)
            b = rand(1:a)
            @assert BigInt(UInt256(a) - UInt256(b)) == a - b
        end
    end
    # println("Time for $trials subtractions: $(subtime.time)")
    # test passed
    @test true
end

function test_mul()
    # trials = 1000000
    multime = @timed begin 
        for _ in 1:trials
            a = rand(1:BigInt(2)^128 - 1)
            b = rand(1:BigInt(2)^128 - 1)
            @assert BigInt(UInt256(a) * UInt256(b)) == a * b "mul failed: $a * $b"
        end
    end
    # println("Time for $trials multiplications: $(multime.time)")
    # test passed
    @test true
end

function test_div()
    # trials = 1000000
    divtime = @timed begin
        for _ in 1:trials
            a = rand(1:BigInt(2)^256 - 1)
            b = rand(1:BigInt(2)^256 - 1)
            @assert BigInt(UInt256(a) ÷ UInt256(b)) == a ÷ b
        end
    end
    # println("Time for $trials divisions: $(divtime.time)")
    # test passed
    @test true
end

function test_mod()
    # trials = 1000000
    modtime = @timed begin
        for _ in 1:trials
            a = rand(1:BigInt(2)^256 - 1)
            b = rand(1:BigInt(2)^256 - 1)
            @assert BigInt(mod(UInt256(a), UInt256(b))) == mod(a, b)
        end
    end
    # println("Time for $trials mods: $(modtime.time)")
    # test passed
    @test true
end

end

using .UInt256Tests
UInt256Tests.run_tests()