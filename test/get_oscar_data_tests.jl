include("../src/get_oscar_data.jl")

module GetOscarDataTests

using ..GetOscarData

using Oscar
using Test

function run_tests()
    test_FqMPolyRingElem()
end

function test_FqMPolyRingElem()
    R, (x, y, z, w) = polynomial_ring(GF(11), 4)

    f = x^7 + 2*y^8 + 3*z^9*w^2 + x^100 + y^100
    g = x^100000 + 2*y^1000000
    h = x^(2^62) + 2*y^(2^61)

    fc = [1, 1, 1, 2, 3]
    fd = [
        0   0   0   0   2
        0   0   0   0   9
        0   0   100 8   0
        100 7   0   0   0
    ]

    gc = [1, 2]
    gd = [
        0       0
        0       0
        0       1000000
        100000  0
    ]

    hc = [1, 2]
    hd = [
        0       0
        0       0
        0       2^61
        2^62    0
    ]

    @test f == GetOscarData.convert_data_to_oscar(fc, fd, R)
    @test g == GetOscarData.convert_data_to_oscar(gc, gd, R)
    @test h == GetOscarData.convert_data_to_oscar(hc, hd, R)
end

end

using .GetOscarDataTests

GetOscarDataTests.run_tests()