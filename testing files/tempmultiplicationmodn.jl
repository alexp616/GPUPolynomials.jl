function inverseModP(p, n)
    for i in 1:p-1
        if n * i % p == 1
            return i
        end
    end
end

function nthPrincipalRootOfUnity(p, n)
    # numbers satisfying a^n = 1
    candidates = []
    finishers = []

    for i in 2:p-1
        if i^n % p == 1
            push!(candidates, i)
        end
    end

    println("Candidates: $candidates")

    for i in eachindex(candidates)
        flag = true
        for k in 1:n-1
            sum = 0
            for j in 0:n-1
                sum += candidates[i]^(k * j)
            end
            # println("sum for $(candidates[i]) with k=$k: $sum")
            if sum % p != 0
                flag = false
                break
            end
        end
        if flag
            push!(finishers, candidates[i])
        end
    end
    println(finishers)
    return 
end


# function nttmodp(vec, prime)
#     n = length(vec)
#     result = fill(Int(0), n)
#     primroot = nthPrincipalRootOfUnity(prime, n)
#     println("$n th principal root of unity mod $prime :$primroot")
#     for i in 0:n-1
#         sum = 0;
#         temp = primroot^i % prime
#         for j in 0:n-1
#             sum += vec[j+1] * temp^j
#         end
#         result[i+1] = sum
#     end

#     return result .% prime
# end

# function inttmodp(vec, prime)
#     n = length(vec)
#     result = fill(Int(0), n)
#     primroot = nthPrincipalRootOfUnity(prime, n)
#     inverse = inverseModP(prime, primroot)
#     println(inverse)
#     for i in 0:n-1
#         sum = 0;
#         temp = inverse^i % prime
#         for j in 0:n-1
#             sum += vec[j+1] * temp^j
#         end
#         result[i+1] = sum
#         println()
#     end

#     inverse_of_length = inverseModP(prime, n)
#     return inverse_of_length .* result .% prime
# end

# function modnMultiply(p1, p2, prime)
#     n = Int.(2^ceil(log2(length(p1) + length(p2) - 1)))
#     finalLength = length(p1) + length(p2) - 1

#     copyp1 = copy(p1)
#     copyp2 = copy(p2)

#     append!(copyp1, zeros(Int, n - length(p1)))
#     append!(copyp2, zeros(Int, n - length(p2)))

#     println("copyp1: $copyp1")
#     println("copyp2: $copyp2")
#     y1 = nttmodp(copyp1, prime)
#     y2 = nttmodp(copyp2, prime)

#     println("y1: $y1")
#     println("y2: $y2")
    
#     y3 = y1 .* y2
#     println("y3: $y3")
#     return 
# end

# p1 = [1,2,3,4]
# p2 = [1,1,1,1]


# modnMultiply(p1, p2, 5)

# for i in 2:24
#     println("i:$i i^8 % 25 = $(i^8 % 25)")
# end

nthPrincipalRootOfUnity(5, 3)