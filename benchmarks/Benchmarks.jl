module Benchmarks

using Oscar
using CUDA
using DataFrames
using CSV

# include("../experimental stuff/TrivialMultiply.jl")
include("../src/Delta1.jl")
include("random_polynomial_generator.jl")
using .Delta1
using .RandomPolynomialGenerator

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


function benchmarks_oscar(df)
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

    samples = nrow(df)
    df.Oscar = zeros(Float64, samples)
    println("BENCHMARKING OSCAR...")
    i = 1
    for p in polys
        # println("Benchmark $i:")
        
        a = @timed oscar_delta1(p, 5)
        df[i, :Oscar] = a.time

        if i % (samples / 5) == 0
            println("\tOSCAR $(i)/$(samples) DONE")
        end
        i = i + 1
    end

    println("BENCHMARKING OSCAR FINISHED")

end


function benchmarks_gpu(samples, df)
    polys = read_benchmarks()
    gpu_data = convert_to_gpu_representation.(polys)

    pregen = pregen_delta1(4, 5)

    h = HomogeneousPolynomial([1, 2, 3, 4], [0 0 0 4; 0 0 4 0; 0 4 0 0; 4 0 0 0])
    # prime the jitter
    # warming up the gpu even more
    for i in 1:10
        delta1(h, 5, pregen)
    end
  
    println("BENCHMARKING FFT...")
    df.FFT = zeros(Float64, samples)
    i = 1
    for data in gpu_data 
        p = HomogeneousPolynomial(data[1],data[2])
        df[i, :numTerms] = length(p.coeffs)
        a = CUDA.@timed delta1(p, 5, pregen)
        df[i, :FFT] = a.time

        if i % (samples / 5) == 0
            println("\tFFT $(i)/$(samples) DONE")
        end
        i += 1
    end
    println("BENCHMARKING FFT FINISHED")
end


function run_all_benchmarks(numSamples)
    println("Generating random polynomials...")
    RandomPolynomialGenerator.run(numSamples)
    df = DataFrame(numTerms = fill(0, numSamples))
    # Because I'm a horrible programmer, benchmarks_gpu also computes the corresponding number of terms for each benchmark
    benchmarks_gpu(numSamples, df)
    benchmarks_oscar(df)

    CSV.write("benchmarks/benchmarks.csv", df)
    nothing # don't spam the terminal
end

end
