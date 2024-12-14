include("src/GPUPolynomials.jl")

function run()
    # n = UInt(512)
    # p = UInt(65537)

    display(Int(GPUPolynomials.modsqrt(UInt(64375), UInt(65537))))
end

run()