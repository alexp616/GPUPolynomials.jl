module MontgomeryReductionTests

include("../src/algorithms/ntt_utils.jl")

using Test
using Primes

function run_tests()
    test_add()
    test_sub()
    test_mul()
    test_pow()
end

function random_prime_in_range(low::Integer, high::Integer)
    @assert low < high "The lower bound must be less than the upper bound."
    prime = nothing
    while prime === nothing
        candidate = rand(low:2:high)
        if isprime(candidate)
            prime = candidate
        end
    end
    return prime
end

function test_add()
    # println("Starting montgomery addition tests...")
    for _ in 1:1000
        prime = rand(3:2:2^54)
        mr = MontgomeryReducer(UInt128(prime))
        for dfdf in 1:1000
            a = unsigned(rand(1:prime - 1))
            b = unsigned(rand(1:prime - 1))

            a_mont = convert_in(mr, a)
            b_mont = convert_in(mr, b)

            result = convert_out(mr, add(mr, a_mont, b_mont))
            @assert result == mod(a + b, prime) "add errored! $a + $b mod $prime != $(mod(a + b, prime))"
        end
    end

    # Test passed
    @test 1 == 1
    # println("Montgomery addition tests finished")
end

function test_sub()
    # println("Starting montgomery subtraction tests...")
    for _ in 1:1000
        prime = rand(3:2:2^54)
        mr = MontgomeryReducer(UInt128(prime))
        for dfdf in 1:1000
            a = unsigned(rand(1:prime - 1))
            b = unsigned(rand(1:prime - 1))

            a_mont = convert_in(mr, a)
            b_mont = convert_in(mr, b)

            result = convert_out(mr, sub(mr, a_mont, b_mont))
            a = signed(a)
            b = signed(b)
            @assert result == mod(a - b, prime) "sub errored! $a - $b mod $prime != $(mod(a - b, prime))"
        end
    end

    # Test passed
    @test 1 == 1
    # println("Montgomery subtraction tests finished")
end

function test_mul()
    # println("Starting montgomery multiplication tests...")
    for _ in 1:1000
        prime = rand(3:2:2^54)
        mr = MontgomeryReducer(UInt128(prime))
        for dfdf in 1:1000
            a = unsigned(rand(1:prime - 1))
            b = unsigned(rand(1:prime - 1))

            a_mont = convert_in(mr, a)
            b_mont = convert_in(mr, b)

            result = convert_out(mr, mul(mr, a_mont, b_mont))
            @assert result == mod(BigInt(a) * b, prime) "mul errored! $a * $b mod $prime != $(mod(BigInt(a) * b, prime))"
        end
    end

    # Test passed
    @test 1 == 1
    # println("Montgomery multiplication tests finished")
end

function test_pow()
    # println("Starting montgomery powering tests...")
    for _ in 1:1000
        prime = rand(3:2:2^54)
        mr = MontgomeryReducer(UInt128(prime))
        for dfdf in 1:1000
            a = unsigned(rand(1:prime - 1))
            b = unsigned(rand(1:prime - 1))

            a_mont = convert_in(mr, a)

            result = convert_out(mr, pow(mr, a_mont, b))
            @assert result == powermod(a, b, prime) "pow errored! $a ^ $b mod $prime != $(powermod(a, b, prime))"
        end
    end

    # Test passed
    @test 1 == 1
    # println("Montgomery powering tests finished")
end

end

using .MontgomeryReductionTests

MontgomeryReductionTests.run_tests()