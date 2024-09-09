using Oscar
include("../src/RandomPolynomials.jl")

function get_max_coeff(numVars, prime)
    R, vars = polynomial_ring(ZZ, numVars)

    timetaken = @timed begin
        monsvec = collect(allmonomialcombos(vars, (prime - 1) * numVars))
        mons = map(x -> prod(x), monsvec)

        fpminus1 = sum(mons)
        fpminus1 *= prime - 1

        result = fpminus1 ^ prime

        # display(length(result))
        coeffsvec = BigInt.(collect(coefficients(result)))
        max = maximum(coeffsvec)
        println("maximum coefficient: $max")
    end

    println("time taken: $(timetaken.time) s")
end