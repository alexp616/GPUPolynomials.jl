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

    for i in 1:p-1
        if i^n % p == 1
            push!(candidates, i)
        end
    end

    while length(candidates) > 0
        for k in 1:n-1
            sum = 0
            for j in 0:n-1
                sum += candidates[1]^(k * j)
            end
            if sum % p == 0
                # TODO polish data typing
                return candidates[1]
            end
        end
        popfirst!(candidates)
    end

    # No idea if this can theoretically ever happen.
    # No idea if the nth principal root(s)? of unity is unique.
    # I'll figure it out another day
    error("p doesn't have an nth principal root of unity")
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

# Unsure, but I don't think polynomial multiplication mod n can be implemented using
# this version of NTT. The part I was confused about with the algorithm for multiplying
# polynomials using DFT was how D^-1(D(x)∘D(y))=conv(x,y). D(x) and D(y) are both length
# n, which is usually a higher degree than the actual product of x and y. 2 things I 
# don't get:
# How does D^-1(D(x)∘D(y))=conv(x,y) ? ∘ is component-wise multiplication, how is it doing
# the same thing as multiplying every term by every other term?
# 
# How does D^-1 know what the degree of the ending polynomial is? Polynomial interpolation
# is usually given n points, an n-1 degree polynomial can be found that fits it. But D^-1
# puts 0's for every degree after the actual degree of conv(x,y)

# Looks like I have some more studying to do


p1 = [1,2,3,4,0,0,0,0]
p2 = [1,2,3,4]

println(ntt(p1, 8, 3, 5))

# modnMultiply(p1, p2, 5)


# println(slowMultiply(p1, p2) .% 5)
# println(modnMultiply(p1, p2, 5))