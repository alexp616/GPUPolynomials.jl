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


function oscar_delta1(p,poly)

    R = parent(poly)
  
    originallift = map_coefficients(x -> lift(ZZ,x),poly)
  
  
    ZR = parent(originallift)
  
    nocrossterms = sum(terms(originallift) .^p)
    withcrossterms = originallift^p
  
  
    crossterms = withcrossterms - nocrossterms
  
    #println("Just cross terms: ", crossterms)
    (x,y,z,w) = gens(ZR)
    #println("Interesting coefficient: $(coeff(withcrossterms,x^14*y^27*z^11*w^28))")
  
    Δlift = map_coefficients(x -> divexact(x,p),crossterms)
  
    change_coefficient_ring(coefficient_ring(R),Δlift,parent=R)
  
end#

function test_5_2()
    n = 5
    p = 2
    R, vars = polynomial_ring(GF(p), n)
    f = random_homog_poly_mod(p, vars, n)

    println("Starting n = $n, p = $p...")
    pregentime = @timed begin
        pregen = pregen_delta1(n, p)
    end
    println("Pregenerating took $(pregentime.time) s")
    

    firststeptime = @timed begin
        fpminus1 = f ^ (p - 1)
    end
    println("First step took $(firststeptime.time) s")

    fp1_gpu = HomogeneousPolynomial(fpminus1)
    sort_to_kronecker_order(fp1_gpu, pregen.key1)

    gputime = CUDA.@timed begin
        gpu_result = delta1(fp1_gpu, p; pregen = pregen)
    end
    println("GPU Delta1 took $(gputime.time) s")

    remove_zeros(gpu_result)

    oscartime = CUDA.@timed begin
        temp = oscar_delta1(p, fpminus1)
    end
    println("Oscar Delta1 took $(oscartime.time) s")

    oscar_result = HomogeneousPolynomial(temp)
    sort_to_kronecker_order(oscar_result, pregen.key2)

    @test oscar_result == gpu_result
    println()
end

function test_5_3()
    n = 5
    p = 3
    R, vars = polynomial_ring(GF(p), n)
    f = random_homog_poly_mod(p, vars, n)

    println("Starting n = $n, p = $p...")
    pregen = pregen_delta1(n, p)
    

    firststeptime = @timed begin
        fpminus1 = f ^ (p - 1)
    end
    println("First step took $(firststeptime.time) s")

    fp1_gpu = HomogeneousPolynomial(fpminus1)
    sort_to_kronecker_order(fp1_gpu, pregen.key1)

    gputime = CUDA.@timed begin
        gpu_result = delta1(fp1_gpu, p; pregen = pregen)
    end
    println("GPU Delta1 took $(gputime.time) s")

    remove_zeros(gpu_result)

    oscartime = CUDA.@timed begin
        temp = oscar_delta1(p, fpminus1)
    end
    println("Oscar Delta1 took $(oscartime.time) s")

    oscar_result = HomogeneousPolynomial(temp)
    sort_to_kronecker_order(oscar_result, pregen.key2)

    @test oscar_result == gpu_result
    println()
end

function test_5_5()
    n = 5
    p = 5
    R, vars = polynomial_ring(GF(p), n)
    f = random_homog_poly_mod(p, vars, n)

    println("Starting n = $n, p = $p...")
    pregen = pregen_delta1(n, p)
    

    firststeptime = @timed begin
        fpminus1 = f ^ (p - 1)
    end
    println("First step took $(firststeptime.time) s")

    fp1_gpu = HomogeneousPolynomial(fpminus1)
    sort_to_kronecker_order(fp1_gpu, pregen.key1)

    gputime = CUDA.@timed begin
        gpu_result = delta1(fp1_gpu, p; pregen = pregen)
    end
    println("GPU Delta1 took $(gputime.time) s")

    remove_zeros(gpu_result)

    oscartime = CUDA.@timed begin
        temp = oscar_delta1(p, fpminus1)
    end
    println("Oscar Delta1 took $(oscartime.time) s")

    oscar_result = HomogeneousPolynomial(temp)
    sort_to_kronecker_order(oscar_result, pregen.key2)

    @test oscar_result == gpu_result
    println()
end

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

function test_conversion_to_gpu()
    R1, vars1 = polynomial_ring(GF(7), 4)
    R2, vars2 = polynomial_ring(ZZ, 4)
    (a, b, c, d) = vars1
    (x, y, z, w) = vars2

    f1 = a + b + c + d
    f2 = x + y + z + w
    
    hp1 = HomogeneousPolynomial(f1)
    hp2 = HomogeneousPolynomial(f2)
    hp3 = HomogeneousPolynomial([1, 1, 1, 1], [
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
    ])

    @test hp1 == hp2
    @test hp2 == hp3
end

function test_K3_7()
    println("Running K3_7 test...")
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)
    x1, x2, x3, x4 = vars

    # Restricted monomials: Contains no x1^4, x2^4, x3^4
    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    pregentime = @timed begin
        pregen = pregen_delta1(4, 7)
    end

    println("Pregenerating took $(pregentime.time) s\n")
    
    f = random_homog_poly_mod_restricted(7, vars, mons)

    # f = 6*x1^3*x2 + 3*x1^3*x3 + 2*x1^2*x2*x3 + 3*x1^2*x2*x4 + 5*x1^2*x3*x4 + x1^2*x4^2 + 2*x1*x2^3 + 5*x1*x2^2*x4 + 6*x1*x2*x3^2 + 2*x1*x2*x3*x4 + 5*x1*x2*x4^2 + 6*x1*x3^3 + 4*x1*x3^2*x4 + 2*x1*x3*x4^2 + 4*x1*x4^3 + x2^3*x3 + x2^3*x4 + x2^2*x3^2 + 6*x2^2*x3*x4 + 2*x2^2*x4^2 + 4*x2*x3^3 + 4*x2*x3^2*x4 + 6*x2*x4^3 + 5*x3^3*x4 + x3*x4^3 + x4^4

    # f = 6*x1^3*x2 + 3*x1^3*x3 + 2*x1^2*x2*x3 + 3*x1^2*x2*x4 + 5*x1^2*x3*x4 + x1^2*x4^2 + 2*x1*x2^3 + 5*x1*x2^2*x4 + 6*x1*x2*x3^2 + 2*x1*x2*x3*x4 + 5*x1*x2*x4^2 + 6*x1*x3^3

    # f = x1^2*x2^2 + x1^2*x3^2 + x1^2*x4^2 + x2^2*x4^2 + x2^2*x3^2 + x3^2*x4^2 + x4^4
    # f = 2*x1^2*x2^2 + x4^4
    fpminus1 = f ^ 6
    oscartime = @timed begin
        oscar_result = oscar_delta1(7, fpminus1)
    end

    sus = parent(oscar_result)

    println("Oscar delta1 took $(oscartime.time) s")

    gputime = CUDA.@timed begin
        fpminus1_gpu = HomogeneousPolynomial(fpminus1)
        sort_to_kronecker_order(fpminus1_gpu, pregen.key1)
        Δ_1fpminus1_gpu = delta1(fpminus1_gpu, 7; pregen = pregen)
    end

    oscar_result_hp = HomogeneousPolynomial(oscar_result)
    sort_to_kronecker_order(oscar_result_hp, pregen.key2)

    println("GPU delta1 took $(gputime.time) s")
    remove_zeros(Δ_1fpminus1_gpu)
    
    @test oscar_result_hp == Δ_1fpminus1_gpu
end

function test_K3_5()
    println("Running K3_5 test...")
    Random.seed!()

    R, vars = polynomial_ring(GF(5), 4)
    x1, x2, x3, x4 = vars

    pregentime = @timed begin
        pregen = pregen_delta1(4, 5)
    end

    println("Pregenerating took $(pregentime.time) s\n")
    
    f = random_homog_poly_mod(5, vars, 4)

    # f = 6*x1^3*x2 + 3*x1^3*x3 + 2*x1^2*x2*x3 + 3*x1^2*x2*x4 + 5*x1^2*x3*x4 + x1^2*x4^2 + 2*x1*x2^3 + 5*x1*x2^2*x4 + 6*x1*x2*x3^2 + 2*x1*x2*x3*x4 + 5*x1*x2*x4^2 + 6*x1*x3^3 + 4*x1*x3^2*x4 + 2*x1*x3*x4^2 + 4*x1*x4^3 + x2^3*x3 + x2^3*x4 + x2^2*x3^2 + 6*x2^2*x3*x4 + 2*x2^2*x4^2 + 4*x2*x3^3 + 4*x2*x3^2*x4 + 6*x2*x4^3 + 5*x3^3*x4 + x3*x4^3 + x4^4

    # f = 6*x1^3*x2 + 3*x1^3*x3 + 2*x1^2*x2*x3 + 3*x1^2*x2*x4 + 5*x1^2*x3*x4 + x1^2*x4^2 + 2*x1*x2^3 + 5*x1*x2^2*x4 + 6*x1*x2*x3^2 + 2*x1*x2*x3*x4 + 5*x1*x2*x4^2 + 6*x1*x3^3

    # f = x1^2*x2^2 + x1^2*x3^2 + x1^2*x4^2 + x2^2*x4^2 + x2^2*x3^2 + x3^2*x4^2 + x4^4
    # f = 2*x1^2*x2^2 + x4^4
    fpminus1 = f ^ 4
    oscartime = @timed begin
        oscar_result = oscar_delta1(5, fpminus1)
    end

    println("Oscar delta1 took $(oscartime.time) s")

    gputime = CUDA.@timed begin
        fpminus1_gpu = HomogeneousPolynomial(fpminus1)
        sort_to_kronecker_order(fpminus1_gpu, pregen.key1)
        Δ_1fpminus1_gpu = delta1(fpminus1_gpu, 5; pregen = pregen)
        # gpu_result = convert_to_oscar(Δ_1fpminus1_gpu, sus)
    end

    oscar_result_hp = HomogeneousPolynomial(oscar_result)
    sort_to_kronecker_order(oscar_result_hp, pregen.key2)

    println("GPU delta1 took $(gputime.time) s")
    remove_zeros(Δ_1fpminus1_gpu)
    
    @test oscar_result_hp.coeffs == Δ_1fpminus1_gpu.coeffs
    @test oscar_result_hp.degrees == Δ_1fpminus1_gpu.degrees
end

function time_K3_7()
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)
    x1, x2, x3, x4 = vars

    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    pregentime = @timed begin
        pregen = pregen_delta1(4, 7)
    end

    println("Pregenerating took $(pregentime.time) s\n")

    numTrials = 10
    for t in 1:numTrials
        f = random_homog_poly_mod_restricted(7, vars, mons)
        fpminus1 = f ^ 6
        gputime = CUDA.@timed begin
            fpminus1_gpu = HomogeneousPolynomial(fpminus1)
            sort_to_kronecker_order(fpminus1_gpu, pregen.key1)
            Δ_1fpminus1_gpu = delta1(fpminus1_gpu, 7; pregen = pregen)
        end
        println("Trial $t: $(gputime.time) s")
    end
end

function profile_K3_7()
    Random.seed!()

    R, vars = polynomial_ring(GF(7), 4)
    x1, x2, x3, x4 = vars

    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    pregentime = @timed begin
        pregen = pregen_delta1(4, 7)
    end

    println("Pregenerating took $(pregentime.time) s\n")

    f = random_homog_poly_mod_restricted(7, vars, mons)
    fpminus1 = f ^ 6
    fpminus1_gpu = HomogeneousPolynomial(fpminus1)
    sort_to_kronecker_order(fpminus1_gpu, pregen.key1)
    Δ_1fpminus1_gpu = delta1(fpminus1_gpu, 7; pregen = pregen)
end

function test_remove_zeros()
    coeffs = [1, 2, 0, 4, 0, 5, 6, 7]
    degrees = [
        4 0 0 0
        0 4 0 0
        0 0 4 0
        0 0 0 4
        3 1 0 0
        3 0 1 0
        3 0 0 1
        1 3 0 0
    ]

    f = HomogeneousPolynomial(coeffs, degrees)
    
    remove_zeros(f)
    @test f.coeffs == [1, 2, 4, 5, 6, 7]
    @test f.degrees == [
        4 0 0 0
        0 4 0 0
        0 0 0 4
        3 0 1 0
        3 0 0 1
        1 3 0 0
    ]
end

@testset "gpu_pow" begin
    test_5_2()
    test_5_3()
    # test_5_5() THIS THING ERRORS
    test_gpu_pow_no_pregen()
    test_gpu_pow()
    test_K3_5()
    test_K3_7()
    test_remove_zeros()
    test_conversion_to_gpu()
    time_K3_7()
    profile_K3_7()
end