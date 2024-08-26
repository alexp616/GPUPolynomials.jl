include("src/GPUPolynomials.jl")

using .Polynomials
using .Delta1
using .GPUPow

using Oscar
using Test
using Combinatorics
using BenchmarkTools
using Random
using CUDA

function test_gpu_pow_no_pregen()
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)

    f = random_homog_poly_mod(7, vars, 4)
    oscar_time = @timed begin 
        oscar_result = f ^ 6
    end

    # println("OSCAR took $(oscar_time.time) s")

    f_gpu = HomogeneousPolynomial(f)
    fft_time = CUDA.@timed begin
        result_gpu = gpu_pow(f_gpu, 6)
    end
    # println("FFT took $(fft_time.time) s")
    gpu_back = convert_to_oscar(result_gpu, R)

    @test oscar_result == gpu_back
end

function test_gpu_pow()
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)

    f = random_homog_poly_mod(7, vars, 4)
    oscar_time = @timed begin 
        oscar_result = f ^ 6
    end
    
    # println("OSCAR took $(oscar_time.time) s")

    f_gpu = HomogeneousPolynomial(f)
    pregen_time = @timed begin
        pregen = GPUPow.pregen_gpu_pow(f_gpu, 6)
    end
    # println("Pregenerating for gpu_pow took $(pregen_time.time) s")

    fft_time = CUDA.@timed begin
        result_gpu = gpu_pow(f_gpu, 6; pregen = pregen)
    end
    # println("FFT took $(fft_time.time) s")
    gpu_back = convert_to_oscar(result_gpu, R)

    @test oscar_result == gpu_back
end

function time_K3_7()
    println("Running K3_7 test...")
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)
    x1, x2, x3, x4 = vars

    # Restricted monomials: Contains no x1^4, x2^4, x3^4
    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    pregentime = @timed begin
        pregen = pregen_delta1(4, 7)
    end

    println("pregen.inputLen1: ", pregen.inputLen1)
    println("Pregenerating took $(pregentime.time) s\n")
    
    f = random_homog_poly_mod_restricted(7, vars, mons)
    println("f: $f\n")
    step1time = @timed begin 
        fpminus1 = f ^ 6
    end
    println("Raising f to the 6th power took $(step1time.time) s.\nf ^ 6 has $(length(fpminus1)) terms\n")

    fpminus1_gpu = HomogeneousPolynomial(fpminus1)

    sort_to_kronecker_order(fpminus1_gpu, pregen.key1)

    # for name in fieldnames(typeof(pregen))
    #     println("$(name): $(getfield(pregen, name))")
    # end
    Δ_1fpminus1_gpu = delta1(fpminus1_gpu, 7, pregen)
    println("result has $(length(Δ_1fpminus1_gpu.coeffs)) terms")
    @test 1 == 1
end

@testset "gpu_pow" begin
    # test_gpu_pow_no_pregen()
    # test_gpu_pow()
    time_K3_7()
end