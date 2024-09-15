include("../src/polynomials.jl")

module PolynomialsTests

using ..Polynomials

using Test
using CUDA

function run_tests()
    test_homogeneous_polynomial()
    test_sparse_polynomial()
    test_dense_polynomial()
    test_kron()
end

function test_homogeneous_polynomial()
    coeffs = [1, 2, 3, 4]
    degrees = [
        2 0 0 1
        1 1 0 0
        0 2 2 0
        0 0 1 2
    ]
    hp = HomogeneousPolynomial(coeffs, degrees)
    @test hp.homogeneousDegree == 3

    # Test remove_zeros
    hp.coeffs = [1, 0, 3, 0]
    remove_zeros(hp)
    @test length(hp.coeffs) == 2
    @test hp == HomogeneousPolynomial([1, 3], [
        2 0
        1 0
        0 2
        0 1
    ])

    @test pretty_string(hp) == "3*z^2*w + x^2*y"
end

function test_sparse_polynomial()
    sp = SparsePolynomial([1, 3], [2, 0])
    sp2 = SparsePolynomial([3, 1], [0, 2])
    sp3 = SparsePolynomial([1, 0, 3], [2, 1, 0])

    @test sp == sp2
    @test sp2 == sp3
    @test sp.coeffs == [1, 3]
end

function test_dense_polynomial()
    sp = SparsePolynomial([3, 0, 1], [5, 1, 0])
    dp = DensePolynomial(sp)
    @test dp.coeffs == [1, 0, 0, 0, 0, 3]
end

function generate_compositions(n, k, type::DataType = Int64)
    compositions = zeros(type, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(type, k)
    current_composition[1] = n
    idx = 1
    while true
        compositions[idx, :] .= current_composition
        idx += 1
        v = current_composition[k]
        if v == n
            break
        end
        current_composition[k] = 0
        j = k - 1
        while 0 == current_composition[j]
            j -= 1
        end
        current_composition[j] -= 1
        current_composition[j + 1] = 1 + v
    end

    return collect(transpose(compositions))
end

function test_kron()
    expArr = generate_compositions(16, 4, UInt32)
    cu_expArr = CuArray(expArr)

    result = cpu_encode_degrees(expArr, 17, false)
    homogResult = cpu_encode_degrees(expArr, 17, true)

    cu_result = CuArray(result)
    cu_homogResult = CuArray(homogResult)

    @test result == Array(gpu_encode_degrees(cu_expArr, 17, false))
    @test homogResult == Array(gpu_encode_degrees(cu_expArr, 17, true))


    # cpu_decode_degrees(result, 17, 4, false)
    # cpu_decode_degrees(homogResult, 17, 4, true, 16)
    @test cpu_decode_degrees(result, 17, 4, false) == Array(gpu_decode_degrees(cu_result, 17, 4, false, 16))
    @test cpu_decode_degrees(homogResult, 17, 4, true, 16) == Array(gpu_decode_degrees(cu_homogResult, 17, 4, true, 16))

    @test cpu_decode_degrees(result, 17, 4, false) == expArr
    @test cpu_decode_degrees(homogResult, 17, 4, true, 16) == expArr
end

end

using .PolynomialsTests
PolynomialsTests.run_tests()