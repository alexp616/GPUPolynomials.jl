function pow_test()
    R, (x, y, z, w) = polynomial_ring(GF(5), 4)

    f = x^3 + 2*x^2*y + 3*z^3 + 4*w^3
    f = f.data

    cu_f = cu(f)

    cu_c = cu_f ^ 2
    c = fpMPolyRingElem(cu_c)

    expected = f ^ 2

    @test c == expected

end

function correct_test()
    expRange = 2:4
    deg = 4

    n = 4
    p = 5

    R, vars = polynomial_ring(GF(p), n)

    f = GPUPolynomials.random_homog_poly_mod(p, vars, deg).data
    f_hp = cu(f)
    for exp in expRange
        oscar_result = f ^ exp

        plan = GPUPolynomials.MPowPlan(f_hp, exp)
        f_hp.opPlan = plan

        cu_f_exp = f_hp ^ exp

        gpu_result = fpMPolyRingElem(cu_f_exp)

        @test string(oscar_result) == string(gpu_result)
    end
end

function run_tests()
    pow_test()
    correct_test()
end

run_tests()