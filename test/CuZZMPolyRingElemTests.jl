function pow_test()
    R, (x, y, z, w) = polynomial_ring(ZZ, 4)

    f = x^3 + 2*x^2*y + 3*z^3 + 4*w^3

    cu_f = cu(f)

    cu_c = cu_f ^ 2
    c = ZZMPolyRingElem(cu_c)

    expected = f ^ 2

    @test c == expected
end

function correct_test()
    expRange = 5:15
    deg = 16

    n = 4
    p = 5

    R, vars = polynomial_ring(ZZ, n)

    f = GPUPolynomials.random_homog_poly_mod(p, vars, deg)
    f_hp = cu(f)
    for exp in expRange
        oscar_result = f ^ exp

        plan = GPUPolynomials.MPowPlan(f_hp, exp)
        f_hp.opPlan = plan

        cu_f_exp = f_hp ^ exp

        gpu_result = ZZMPolyRingElem(cu_f_exp)

        @test oscar_result == gpu_result
    end
end

function run_tests()
    pow_test()
    correct_test()
end

run_tests()