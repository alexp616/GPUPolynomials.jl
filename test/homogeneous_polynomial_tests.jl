include("../src/GPUPolynomials.jl")

module HomogeneousPolynomialTests

include("../delta1/randompolynomials.jl")

using ..GPUPolynomials
using Oscar
using Test

function run_tests()
    test_FqMPolyRingElem()
    test_FqMPolyRingElem2()
    test_gpu_pow()
    test_free_memory()
end

function test_FqMPolyRingElem()
    R, vars = polynomial_ring(GF(11), 4)

    f = random_homog_poly_mod(11, vars, 4)
    hp = HomogeneousPolynomial(f)
    @test hp.homogDegree == 4
    
    g = convert(typeof(f), hp)

    @test f == g
end

function test_FqMPolyRingElem2()
    R, vars = polynomial_ring(GF(11), 4)
    (x, y, z, w) = vars

    f = x^4 + y^4 + z^4 + w^4
    g = x^4 + 2*y^4 + z^4 + w^4

    hp = HomogeneousPolynomial(f)
    vec = get_coeffs(hp)
    vec[2] = 2
    
    @test f == g
end

function test_gpu_pow()
    n = 4
    p = 5
    pow = 2
    R, vars = polynomial_ring(GF(p), n)
    (x, y, z, w) = vars
    # f = random_homog_poly_mod(p, vars, n)
    f = x^4 + y^4 + z^4 + w^4
    hp = HomogeneousPolynomial(f)
    gpupregen = pregen_gpu_pow(hp, pow, UInt64)
    hp_result = gpu_pow(hp, pow, gpupregen)

    g = convert(typeof(f), hp_result)
    display(g)
    @test g == f ^ pow
end

function test_free_memory()
    n = 4
    p = 5
    R, vars = polynomial_ring(GF(p), n)
    (x, y, z, w) = vars
    f = x^4 + 2*y^4 + 3*z^4 + 4*w^4

    exps = UInt.([
        4 0 0 0
        0 4 0 0
        0 0 4 0
        0 0 0 4
    ])
    coefs = UInt.([1, 2, 3, 4])
    g = fpMPolyRingElem(R.data, coefs, exps)
    # THIS DOESNT ERROR
    # g = new_MPolyRingElem(get_coeffs(f), get_exps(f), 16, R)
    @test f.data == g
end

end

using .HomogeneousPolynomialTests

HomogeneousPolynomialTests.run_tests()