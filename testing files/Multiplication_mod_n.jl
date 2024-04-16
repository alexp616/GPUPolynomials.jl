function bitReverse(x, log2n)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end

# Finds multiplicative inverse of n mod p
# Works perfectly fine when p is small, probably faster
# than Berkelamp
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
    results = []

    for i in 2:p-1
        println("i^n: $(i^n), i: $i")
        if i^n % p == 1
            push!(candidates, i)
        end
    end

    # println("candidates: $candidates")
    while length(candidates) > 0
        flag = true
        for k in 1:n-1
            if candidates[1] ^ k % p == 1
                flag = false
                break
            end
        end
        if flag 
            push!(results, candidates[1])
        end
        popfirst!(candidates)
    end

    return results

    # No idea if this can theoretically ever happen.
    # No idea if the nth principal root(s)? of unity is unique.
    # I'll figure it out another day
    error("$p doesn't have a $n th principal root of unity")
end

# Added principal root of unity method, replaced:
# theta -> alpha = 1 (multiplicative identity in Z_prime)
# theta_m -> alpha_m = npru
function ntt(vec, n, log2n, prime, inverted = 1)
    result = fill(Int64(0), n)

    # nth principal root of unity
    npru = nthPrincipalRootOfUnity(prime, n)

    for i in 0:n-1
        rev = bitReverse(i, log2n)
        result[i+1] = vec[rev+1]
    end

    for i in 1:log2n
        m = 1 << i
        m2 = m >> 1
        alpha = 1
        alpha_m = inverted * npru
        for j in 0:m2-1
            for k in j:m:n-1
                t = alpha * result[k + m2 + 1]
                u = result[k + 1]

                result[k + 1] = u + t
                result[k + m2 + 1] = u - t
            end
            alpha *= alpha_m % prime
        end
    end

    return result .% prime
end

function intt(vec, n, log2n, prime)
    return (inverseModP(prime, n) .* ntt(vec, n, log2n, prime, -1)) .% prime
end


function modnMultiply(p1, p2, prime)
    n = Int.(2^ceil(log2(length(p1) + length(p2) - 1)))
    log2n = UInt32(log2(n));
    finalLength = length(p1) + length(p2) - 1

    copyp1 = copy(p1)
    copyp2 = copy(p2)

    append!(copyp1, zeros(Int, n - length(p1)))
    append!(copyp2, zeros(Int, n - length(p2)))

    y1 = ntt(copyp1, n, log2n, prime)
    y2 = ntt(copyp2, n, log2n, prime)

    println(y1)
    println(y2)

    ans = intt(y1 .* y2, n, log2n, prime)
    return ans
end



function slowMultiply(p1, p2)
    temp = fill(0, length(p1) + length(p2)-1)

    for i in eachindex(p1) 
        for j in eachindex(p2)
            @inbounds temp[i + j - 1] += p1[i] * p2[j]
        end
    end

    return temp
end

# NTT and INTT work perfectly well. At least, assuming n is a power of 2, and p is a relatively small prime.

# Need to think about why multiplication doesn't work. Maybe something is just wrong in my code and I'm going to 
# insane trying to find an answer to a problem that doesn't exist


p1 = [1,2,3,4]
p2 = [1,1,1,1]

# println(modnMultiply(p1, p2, 5))

# println(ntt(p1, 8, 3, 5))

# modnMultiply(p1, p2, 5)


# println(slowMultiply(p1, p2) .% 5)
# println(modnMultiply(p1, p2, 5))

println(nthPrincipalRootOfUnity(125, 100))