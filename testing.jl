using CUDA
using Oscar
# using GPUPolynomials
# include("src/GPUPolynomials.jl")

function run()
    R, x = polynomial_ring(ZZ, :x)

    f = x + x^10000

    display(fieldnames(typeof(f)))
    display(f.length)
end

run()