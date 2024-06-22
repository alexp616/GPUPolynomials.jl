using Test
include("../utils.jl")

"""
This file includes 3 main algorithms and their implementation on the CPU.
    - CPUSlowMultiply
    - CPUDFT
    - CPUMultiply

The GPU-parallelized versions are in GPUAlgorithms.jl
"""



"""
    CPUSlowMultiply(p1, p2)

Multiply two polynomials represented by vectors p1 and p2 and return
the resulting polynomial as an vector.

Runs in O(mn) asymptotic time, where m and n are the degrees of p1 and p2.
"""
function CPUSlowMultiply(p1, p2)
    temp = fill(0, length(p1) + length(p2)-1)

    for i in eachindex(p1) 
        for j in eachindex(p2)
            @inbounds temp[i + j - 1] += p1[i] * p2[j]
        end
    end

    return temp
end


"""
    CPUMultiply(p1, p2)

Multiply two polynomials represented by vectors p1 and p2 and return
the resulting polynomial as an vector. Assumes coefficients are integers.

Runs in O(nlogn) asymptotic time, where n is the degree of the resulting
polynomial.
"""
function CPUMultiply(p1, p2)
    n = Int(2^ceil(log2(length(p1) + length(p2) - 1)))
    log2n = UInt32(log2(n));
    finalLength = length(p1) + length(p2) - 1

    copyp1 = copy(p1)
    copyp2 = copy(p2)

    append!(copyp1, zeros(Int, n - length(p1)))
    append!(copyp2, zeros(Int, n - length(p2)))

    y1 = CPUDFT(copyp1, n, log2n)
    y2 = CPUDFT(copyp2, n, log2n)

    ans = CPUIDFT(y1 .* y2, n, log2n)
    return Int.(round.(real.(ans[1:finalLength])))
end


"""
    CPUDFT(p, n, log2n, inverted = 1)

Return the DFT of vector p as a vector. Output can be complex.

Does not work when log2(length(p)) ∉ z
"""
function CPUDFT(p, n, log2n, inverted = 1)
    result = fill(ComplexF32(0), n)

    for i in 0:n-1
        rev = bitReverse(i, log2n)
        result[i+1] = p[rev+1]
    end

    for i in 1:log2n
        m = 1 << i
        m2 = m >> 1
        theta = complex(1,0)
        theta_m = cis(inverted * pi/m2)
        println("theta_m: $theta_m")
        for j in 0:m2-1
            for k in j:m:n-1
                t = theta * result[k + m2 + 1]
                u = result[k + 1]

                result[k + 1] = u + t
                result[k + m2 + 1] = u - t
            end
            theta *= theta_m
        end
    end

    return result
end

CPUDFT([1, 2, 3, 4, 0, 0, 0, 0], 8, 3)

"""
    CPUIDFT(y, log2n, n)

Return the inverse DFT of vector p as a vector.

Does not work when length(p) != 2^k for k∈Z
"""
function CPUIDFT(y, n, log2n)
    return CPUDFT(y, n, log2n, -1) ./ n
end


# TEST CASE
polynomial1 = rand(-100:100, rand(10:50))
polynomial2 = rand(-100:100, rand(10:50))

@test CPUMultiply(polynomial1, polynomial2) == CPUSlowMultiply(polynomial1, polynomial2)



# Recursive implementation of the iterative algorithm above, in case I ever need it
# 
# 
# function recursiveDFT(a, inverted = 1)
#     theta = [cis(2 * (i - 1) * inverted * pi / length(a)) for i in 1:div(length(a), 2)]
#     return recursiveDFThelper(a, theta, 0, inverted)
# end

# function recursiveDFThelper(a, theta, depth = 0, inverted = 1)
#     n = length(a)
#     if n == 1 return a
#     end

#     # Slicing up polynomial for p(x) = p0(x^2) + xp1(x^2)
#     a0 = a[1:2:n]
#     a1 = a[2:2:n]

#     # Recursive step, this is what makes the algorithm nlog(n)
#     y0 = recursiveDFThelper(a0, theta, depth + 1, inverted)
#     y1 = recursiveDFThelper(a1, theta, depth + 1, inverted)

#     # Initializing final array
#     result = fill(ComplexF32(0), n)

#     for i in 1:div(n, 2)
#         # p(x) = p0(x^2) + xp1(x^2)
#         @inbounds result[i] = y0[i] + theta[(2^depth) * (i - 1) + 1] * y1[i]
#         @inbounds result[i + div(n, 2)] = y0[i] - theta[(2^depth) * (i - 1) + 1] * y1[i]
#     end

#     return result
# end


# function recursiveIDFT(y)
#     n = length(y)
#     result = recursiveDFT(y, -1)
#     return [result[i] / n for i in eachindex(result)]
# end

# function recursiveMultiply(p1, p2)
#     n = Int.(2^ceil(log2(length(p1) + length(p2) - 1)))
#     finalLength = length(p1) + length(p2) - 1

#     copyp1 = copy(p1)
#     copyp2 = copy(p2)

#     append!(copyp1, zeros(Int, n - length(p1)))
#     append!(copyp2, zeros(Int, n - length(p2)))

#     y1 = recursiveDFT(copyp1)
#     y2 = recursiveDFT(copyp2)

#     ans = recursiveIDFT([y1[i] * y2[i] for i in 1:n])
#     return [round(Int, real(ans[i])) for i in 1:finalLength]
# end