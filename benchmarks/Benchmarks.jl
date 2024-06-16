module Benchmarks

using Oscar

include("../experimental stuff/TrivialMultiply.jl")

# for the Meta.parse thing to work, these have to be global varibles
# admittedly, this is a bit of a hack.
R, (x,y,z,w) = polynomial_ring(GF(5),["x","y","z","w"])

function read_benchmarks()
  lines = readlines("benchmarks.txt")

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

  # prime the jitter
  println("PRIMING THE JITTER:")
  @time polys[1]^4
  lift1 = lift_to_ZZ(polys[1])
  @time lift1^5

  answers = []
  
  println("TESTING:")
  i = 1
  for p in polys
    println("Benchmark $i:")
    i = i + 1

    @time intermed_term = p^4

    lifted = lift_to_ZZ(intermed_term)

    @time answer = lifted^5
    push!(answers,answer)
  end

  answers
end

function benchmarks_gpu()
  polys = read_benchmarks()
  gpu_data = convert_to_gpu_representation.(polys)

  # prime the jitter
  println("PRIMING THE JITTER:")
  h = PolynomialModule.HostPolynomial(gpu_data[1][1],gpu_data[1][2])
  CUDA.@time raise_to_power(h,4,5)

  CUDA.@time raise_to_power(h,5)

  answers = []
  
  println("TESTING:")
  i = 1
  for data in gpu_data 
    println("Benchmark $i:")
    i = i + 1
    
    p = PolynomialModule.HostPolynomial(data[1],data[2],81) # 81 = 4*4*5 + 1

    CUDA.@time intermed_term = raise_to_power(p,4,5)

    CUDA.@time answer = raise_to_power(intermed_term,5)
    push!(answers,answer)
  end

  answers
end

function run_all_benchmarks()

  println("--------------GPU-------------")
  benchmarks_gpu()
  println("-------------Oscar------------")
  benchmarks_oscar()

  nothing # don't spam the terminal
end

end
