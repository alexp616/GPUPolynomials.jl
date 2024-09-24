using Test

@testset "GPUPolynomials.jl" begin
    # include("UInt256_tests.jl")
    # include("get_oscar_data_tests.jl") Memory issues with convert_data_to_oscar, need to learn more
    include("montgomery_reduction_tests.jl")
    include("ntt_pow_tests.jl")
    # include("polynomials_tests.jl")
end
