module RandomPolynomialGenerator

include("../src/GPUPolynomials.jl")
using .Delta1

function generate_random_polynomials(numVars, modulus, numPolynomials, outputFile = "benchmarks/randompolynomials.txt")
    maxTerms = binomial(numVars + numVars - 1, numVars)
    open(outputFile, "w") do file
        redirect_stdout(file) do
            for _ in 1:numPolynomials
                numTerms = rand(1:maxTerms)
                println(pretty_string(random_homogeneous_polynomial(numVars, numTerms, modulus)))
            end
        end
    end
end

function run(numSamples)
    generate_random_polynomials(4, 5, numSamples)
end

end