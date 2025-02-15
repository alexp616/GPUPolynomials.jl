function add_test()
    R, x = polynomial_ring(ZZ)

    f = x + x^5
    g = x + x^10

    cu_f = cu(f)
    cu_g = cu(g)

    cu_c = cu_f + cu_g + 1
    c = ZZPolyRingElem(cu_c)
    expected = f + g + 1
    
    @test c == expected
end

function sub_test()
    R, x = polynomial_ring(ZZ)

    f = x + x^5
    g = x + x^10

    cu_f = cu(f)
    cu_g = cu(g)

    cu_c = cu_f - cu_g
    c = ZZPolyRingElem(cu_c)
    expected = f - g
    
    @test c == expected
end

function mul_test()
    R, x = polynomial_ring(ZZ)

    f = 1 + 2*x + 3*x^2
    g = 10*x + x^50

    cu_f = cu(f)
    cu_g = cu(g)

    cu_c = cu_f * cu_g
    c = ZZPolyRingElem(cu_c)
    expected = f * g

    @test c == expected
end

function run_tests()
    add_test()
    sub_test()
    mul_test()
end

run_tests()