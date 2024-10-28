include("../src/GPUPolynomials.jl")
include("../test/randompolynomials.jl")
using CUDA

using .GPUPolynomials

function oscar_benchmarks(homogdeg, expRange = 5:15)
    deg = homogdeg
    expRange = 5:15

    n = 4
    p = 5

    trials = 10
    R, vars = polynomial_ring(ZZ, n)

    println("OSCAR BENCHMARKS: \n")

    for exp in expRange
        f = random_homog_poly_mod(7, vars, deg)
        g = f ^ exp

        totaltime = 0
        for i in 1:trials
            f = random_homog_poly_mod(7, vars, deg)
            b = @timed begin
                g = f ^ exp
            end
            totaltime += b.time
        end
        averagetime = totaltime / trials

        println("\tRaising $n-variate, $deg-homogeneous polynomial to the $exp:")
        println("\t\t$averagetime s")
    end
end

function gpufft_benchmarks(homogdeg, expRange = 5:15)
    deg = homogdeg
    expRange = 5:15

    n = 4
    p = 5

    trials = 10
    R, vars = polynomial_ring(GF(p), n)

    println("GPUFFT BENCHMARKS: \n")

    for exp in expRange
        f = random_homog_poly_mod(7, vars, deg)
        f_hp = HomogeneousPolynomial(f)
        pregen = pregen_gpu_pow(f_hp, exp)
        g = gpu_pow(f_hp, exp, pregen)

        totaltime = 0
        for i in 1:trials
            f = random_homog_poly_mod(7, vars, deg)
            f_hp = HomogeneousPolynomial(f)
            b = CUDA.@timed begin
                g = gpu_pow(f_hp, exp, pregen)
            end
            totaltime += b.time
        end
        averagetime = totaltime / trials

        println("\tRaising $n-variate, $deg-homogeneous polynomial to the $exp:")
        println("\t\t$averagetime s")
    end
end