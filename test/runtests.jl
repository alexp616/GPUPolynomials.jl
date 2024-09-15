using Test

@testset "GPUPolynomials.jl" begin
    # include("get_oscar_data_tests.jl") Memory issues with convert_data_to_oscar, need to learn more
    include("ntt_pow_tests.jl")
    include("polynomials_tests.jl")
end
