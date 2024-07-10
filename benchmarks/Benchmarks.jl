module Benchmarks

using Oscar

# include("../experimental stuff/TrivialMultiply.jl")
include("../src/Delta1.jl")
include("../benchmarks/random_polynomial_generator.jl")

# for the Meta.parse thing to work, these have to be global varibles
# admittedly, this is a bit of a hack.
R, (x,y,z,w) = polynomial_ring(GF(5),["x","y","z","w"])


function read_benchmarks()
    lines = readlines("benchmarks/randompolynomials.txt")

    polynomials = eval.(Meta.parse.(lines))

    polynomials
end

lift_to_ZZ(p) = map_coefficients(x -> lift(ZZ,x),p)

function convert_to_gpu_representation(p)
  coeffs = coefficients(p)

  # julia by default doesn't realize that "ZZ" is not
  # an array, so insert it as a one-element tuple "(ZZ,)"
  # so that julia will know not to broadcast along it.
  coeffs_as_int64arr = Int.(lift.((ZZ,),coeffs))

  exp_vecs = leading_exponent_vector.(terms(p))

  # shamelessly taken from 
  # https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/2 
  exponent_mat = reduce(vcat,transpose.(exp_vecs))

  (coeffs_as_int64arr,exponent_mat)
end


function benchmarks_oscar()
    polys = read_benchmarks()

    function oscar_delta1(poly, prime)
        intermed_term = poly^4
        lifted = lift_to_ZZ(intermed_term)
        return lifted ^ 5
    end
    # prime the jitter
    polys[1]^4
    lift1 = lift_to_ZZ(polys[1])
    lift1^5

    answers = []
    
    println("BENCHMARKING OSCAR...")
    # i = 1
    open("benchmarks/oscar_benchmarks.txt", "w") do file
        redirect_stdout(file) do
            println("OSCAR BENCHMARKS")
            for p in polys
                # println("Benchmark $i:")
                # i = i + 1
                @time answer = oscar_delta1(p, 5)

                push!(answers,answer)
            end
        end
    end
    println("BENCHMARKING OSCAR FINISHED")

    # display(answers)
    answers
end


function benchmarks_gpu(samples)
    polys = read_benchmarks()
    gpu_data = convert_to_gpu_representation.(polys)

    pregen = pregen_delta1(4, 5)

    h = HomogeneousPolynomial([1, 2, 3, 4], [0 0 0 4; 0 0 4 0; 0 4 0 0; 4 0 0 0])
    # prime the jitter
    # warming up the gpu even more
    for i in 1:10
        delta1(h, 5, pregen)
    end

    answers = []
  
    println("BENCHMARKING FFT...")
    i = 1
    numTermsArray = zeros(1:samples)
    open("benchmarks/fft_benchmarks.txt", "w") do file
        redirect_stdout(file) do
            println("FFT BENCHMARKS")
            for data in gpu_data 
                # println("Benchmark $i:")
                # i = i + 1
                
                p = HomogeneousPolynomial(data[1],data[2])
                numTermsArray[i] = length(p.coeffs)
                CUDA.@time answer = delta1(p, 5, pregen)
                push!(answers,answer)
                i += 1
            end
        end
    end
    println("BENCHMARKING FFT FINISHED")

    open("benchmarks/num_terms.txt", "w") do file
        redirect_stdout(file) do
            println("NUM TERMS")
            for i in eachindex(numTermsArray) 
                # println("Benchmark $i:")
                # i = i + 1
                println(Int(numTermsArray[i]))
            end
        end
    end
    answers
end


function run_all_benchmarks(numSamples)
    RandomPolynomialGenerator.run(numSamples)
    println("--------------GPU-------------")
    # Because I'm a horrible programmer, benchmarks_gpu also writes the 
    # corresponding number of terms for each benchmark
    benchmarks_gpu(numSamples)
    println("-------------Oscar------------")
    benchmarks_oscar()

    nothing # don't spam the terminal
end

end
