import Base.+
import Base.*


mutable struct Polynomial
    coeffs::Vector{Int}
    degrees::Vector{Int}
    numTerms::Int

    function Polynomial(coeffs::Vector{Int}, degrees::Vector{Int})
        @assert length(coeffs) == length(degrees)
        new(coeffs, degrees, length(coeffs))
    end
end

function +(p1::Polynomial, p2::Polynomial)::Polynomial
    # Pre-allocate sufficiently large array to avoid out of boundsing and append!()
    resultCoeffs = zeros(Int, p1.numTerms + p2.numTerms)
    resultDegrees = zeros(Int, p1.numTerms + p2.numTerms)

    # The algorithm
    k = 0
    i = 1
    j = 1

    while i <= p1.numTerms && j <= p2.numTerms
        k += 1
        if p1.degrees[i] < p2.degrees[j]
            resultCoeffs[k] = p2.coeffs[j]
            resultDegrees[k] = p2.degrees[j]
            j += 1
        elseif p1.degrees[i] == p2.degrees[j]
            resultCoeffs[k] = p1.coeffs[i] + p2.coeffs[j]
            resultDegrees[k] = p1.degrees[i]
            if resultCoeffs[k] == 0
                k -= 1
            end
            i += 1
            j += 1
        else
            resultCoeffs[k] = p1.coeffs[i]
            resultDegrees[k] = p1.degrees[i]
            i += 1
        end

    end
    while i <= p1.numTerms
        k += 1
        resultCoeffs[k] = p1.coeffs[i]
        resultDegrees[k] = p1.degrees[i]
        i += 1
    end
    while j <= p2.numTerms
        k += 1
        resultCoeffs[k] = p2.coeffs[j]
        resultDegrees[k] = p2.degrees[j]
        j += 1
    end

    return Polynomial(resultCoeffs[1:k], resultDegrees[1:k])
end

function test_cpu_add()
    f = Polynomial([1, 2, -3, 4, 1], [9, 7, 5, 2, 0])
    g = Polynomial([3, 2, 3, 2, 1], [5, 3, 2, 1, 0])

    f + g
end

function *(p1::Polynomial, p2::Polynomial)::Polynomial
    if p1.numTerms == 0 || p2.numTerms == 0
        return Polynomial(Int[], Int[])
    end

    resultCoeffs = zeros(Int, p1.numTerms * p2.numTerms)
    resultDegrees = zeros(Int, p1.numTerms * p2.numTerms)

    k = 1
    resultDegrees[1] = p1.degrees[1] + p2.degrees[1]

    F = ones(Int, p1.numTerms)
    I = 1
    while I <= p1.numTerms
        s = find_an_s(I, p1.numTerms, F, p1.degrees, p2.degrees)
        if resultDegrees[k] != p1.degrees[s] + p2.degrees[F[s]]
            if resultCoeffs[k] != 0
                k += 1
                resultCoeffs[k] = 0
            end
            resultDegrees[k] = p1.degrees[s] + p2.degrees[F[s]]
        end
        resultCoeffs[k] += p1.coeffs[s] * p2.coeffs[F[s]]
        F[s] += 1
        if F[s] > p2.numTerms
            I += 1
        end
    end

    return Polynomial(resultCoeffs[1:k], resultDegrees[1:k])
end

function find_an_s(I, n, F, degrees1, degrees2)
    currMaxS = I
    currMaxDegree = degrees1[I] + degrees2[F[I]]
    for s in I + 1:n
        if degrees1[s] + degrees2[F[s]] > currMaxDegree
            currMaxS = s
        end
    end

    return currMaxS
end

function test_cpu_multiply()
    f = Polynomial([1, 4, 1], [5, 2, 0])
    g = Polynomial([2, 3, 2], [3, 2, 1])

    f * g
end

function polynomial_power(f, k)
    d = length(f) - 1  # Degree of the polynomial
    g = zeros(eltype(f), (k - 1) * d + 1)
    a = zeros(eltype(f), k * d + 1)
    
    g[(k - 1) * d + 1] = f[d + 1]^(k - 1)
    a[k * d + 1] = f[d + 1]^k
    
    for i in (k * d) - 1:-1:0
        s = 0
        c = 0
        
        for j in max(0, d + i - d * k):min(d - 1, i)
            s += g[i - j + 1] * f[j + 1]
            if i >= d
                c += (i - k * j) * g[i - j + 1] * f[j + 1]
            end
        end
        
        c /= (d * k - i) * f[d + 1]
        s += c * f[d + 1]
        if i >= d
            g[i - d + 1] = c
        end
        a[i + 1] = s
    end
    
    return a, g
end

@time polynomial_power([1, 2, 3, 4], 5)
@time polynomial_power(ones(1, 104976), 5)

