include("../src/Polynomials.jl")
include("../src/gpu_ntt_pow.jl")
include("../src/ntt_utils.jl")

using ..Polynomials
using ..GPUPow
using Oscar
using CUDA
using BitIntegers

# Max degree for restricted 11:
# x^3*y -> x^330*y^110 = 330 * 331^2 + 110 * 331 + 1 = 36191541
# nextpow(2, 36191541) = 67108864
# inputlen = x^30*y^10 = 30 * 331^2 + 10 * 331 + 1

function get_result()
    R, vars = polynomial_ring(ZZ, 4)
    (x1, x2, x3, x4) = vars
    # mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]
    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    f = sum(mons)

    g = f ^ 10
    g = map_coefficients(c -> 10, g)

    poly = HomogeneousPolynomial(g)
    
    # display(poly.coeffs)
    # display(poly.degrees)

    println("Pregenerating...")
    pregentime = @timed begin
        pregen1 = pregen_gpu_pow([2281701377], 67108864, Int, Int)
        pregen2 = pregen_gpu_pow([3221225473], 67108864, Int, Int)
        pregen3 = pregen_gpu_pow([3489660929], 67108864, Int, Int)
        pregen4 = pregen_gpu_pow([3892314113], 67108864, Int, Int)
        pregen5 = pregen_gpu_pow([7918845953], 67108864, Int, Int)
        pregen6 = pregen_gpu_pow([8858370049], 67108864, Int, Int)
    end
    println("Pregenerating took $(pregentime.time) s")
    vec = CuArray(kronecker_substitution(poly, 331, 3290141))
    gputime = @timed begin
        result1 = Array(gpu_pow(vec, 11; pregen = pregen1))
        println("Got through 1")
        result2 = Array(gpu_pow(vec, 11; pregen = pregen2))
        println("Got through 2")
        result3 = Array(gpu_pow(vec, 11; pregen = pregen3))
        println("Got through 3")
        result4 = Array(gpu_pow(vec, 11; pregen = pregen4))
        println("Got through 4")
        result5 = Array(gpu_pow(vec, 11; pregen = pregen5))
        println("Got through 5")
        result6 = Array(gpu_pow(vec, 11; pregen = pregen6))
        println("Got through 6")
    end
    a = Int512.(vcat(result1', result2', result3', result4', result5', result6'))
    b = pregen_crt([2281701377, 3221225473, 3489660929, 3892314113, 7918845953, 8858370049], Int512)

    return (a, b)
end

function crt(a, b)
    result = zeros(Int256, size(vcat, 2))

    crttime = @timed GPUPow.do_crt(a, b; dest = result)
    println("crt took $(crttime.time) s")
    max_value, max_index = findmax(result)
    println("max_value: ", max_value)
    println("max_index: ", max_index)



    # max_value, max_index = findmax(result.coeffs)
    # println(max_value)
    # println(max_index)
    # @assert gpuresult == oscar_result
    # display(result2.coeffs)
    # display(result2.degrees)

    # println("maximum coefficient: $(maximum(result1.coeffs))")
end
