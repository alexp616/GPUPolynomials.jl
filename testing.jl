using GPUPolynomials
using Oscar

function run()
    R, (x, y, z) = polynomial_ring(ZZ, 3)

    f = x + 2*y + 3*z

    cu_f = cu(f)
    display(cu_f)

    back = ZZMPolyRingElem(cu_f)

    display(back)
end

run()