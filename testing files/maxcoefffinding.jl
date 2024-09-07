include("../src/Polynomials.jl")
include("../src/gpu_ntt_pow.jl")

using ..Polynomials
using ..GPUPow
using Oscar
using CUDA
function mod_inverse(n::Integer, p::Integer)
    n = mod(n, p)

    t, new_t = 0, 1
    r, new_r = p, n

    while new_r != 0
        quotient = r ÷ new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    end

    return t < 0 ? t + p : t
end
# Max degree for restricted 11:
# x^3*y -> x^330*y^110 = 330 * 331^2 + 110 * 331 + 1 = 36191541
# nextpow(2, 36191541) = 67108864
# inputlen = x^30*y^10 = 30 * 331^2 + 10 * 331 + 1

function crt_vectors(vectors::Vector{Vector{T}}, moduli::Vector{T}) where T<:Integer
    n = length(vectors)      # Number of vectors
    len = length(vectors[1])  # Length of each vector
    
    # Ensure all vectors have the same length
    for vec in vectors
        if length(vec) != len
            throw(ArgumentError("All vectors must have the same length"))
        end
    end
    
    # Compute the product of all moduli
    M = prod(moduli)
    
    # Initialize the result vector
    result = zeros(Int, len)
    
    # Apply CRT for each element in the vectors
    for i in 1:len
        for j in 1:n
            # Compute M_j = M / moduli[j]
            M_j = div(M, moduli[j])
            # Use the modular inverse of M_j mod moduli[j]
            inv_Mj = mod_inverse(M_j, moduli[j])
            # Apply CRT formula: result[i] = sum(vectors[j][i] * M_j * inv_Mj) mod M
            result[i] += vectors[j][i] * M_j * inv_Mj
        end
        result[i] = mod(result[i], M)
    end
    
    return result
end

function run()
    R, vars = polynomial_ring(ZZ, 4)
    (x1, x2, x3, x4) = vars
    # mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]
    mons = [x1^3*x2, x1^3*x3, x1^3*x4, x1^2*x2^2, x1^2*x2*x3, x1^2*x2*x4, x1^2*x3^2, x1^2*x3*x4, x1^2*x4^2, x1*x2^3, x1*x2^2*x3, x1*x2^2*x4, x1*x2*x3^2, x1*x2*x3*x4, x1*x2*x4^2, x1*x3^3, x1*x3^2*x4, x1*x3*x4^2, x1*x4^3, x2^3*x3, x2^3*x4, x2^2*x3^2, x2^2*x3*x4, x2^2*x4^2, x2*x3^3, x2*x3^2*x4, x2*x3*x4^2, x2*x4^3, x3^3*x4, x3^2*x4^2, x3*x4^3, x4^4]

    f = sum(mons)

    g = f ^ 10
    g = map_coefficients(c -> 10, g)

    poly = HomogeneousPolynomial(g)
    
    # display(poly.coeffs)
    # display(poly.degrees)

    println("Pregenerating...")
    pregentime = @timed begin
        pregen1 = pregen_gpu_pow([2281701377], 67108864, Int, Int)
        pregen2 = pregen_gpu_pow([3221225473], 67108864, Int, Int)
        pregen3 = pregen_gpu_pow([3489660929], 67108864, Int, Int)
        pregen4 = pregen_gpu_pow([3892314113], 67108864, Int, Int)
        pregen5 = pregen_gpu_pow([7918845953], 67108864, Int, Int)
        pregen6 = pregen_gpu_pow([8858370049], 67108864, Int, Int)
    end
    println("Pregenerating took $(pregentime.time) s")
    vec = CuArray(kronecker_substitution(poly, 331, 3290141))
    gputime = @timed begin
        result1 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen1)))
        println("Got through 1")
        result2 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen2)))
        println("Got through 2")
        result3 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen3)))
        println("Got through 3")
        result4 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen4)))
        println("Got through 4")
        result5 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen5)))
        println("Got through 5")
        result6 = BigInt.(Array(gpu_pow(vec, 11; pregen = pregen6)))
        println("Got through 6")
    end
    a = [result1, result2, result3, result4, result5, result6]
    b = BigInt.([2281701377, 3221225473, 3489660929, 3892314113, 7918845953, 8858370049])

    println("gpupow took $(gputime.time) s")
    result = crt_vectors(a, b)

    for i in eachindex(result)
        if result1.coeffs[i] % 11 != 0
            throw(skibidi)
        end
    end

    max_value, max_index = findmax(result)
    println("max_value: ", max_value)
    println("max_index: ", max_index)

    
    # gpuresult = decode_kronecker_substitution(result1, 331, 4, 440)

    # for i in eachindex(result1.coeffs)
    #     if result1.coeffs[i] % 11 != 0
    #         throw(skibidi)
    #     end
    # end


    # max_value, max_index = findmax(result.coeffs)
    # println(max_value)
    # println(max_index)
    # @assert gpuresult == oscar_result
    # display(result2.coeffs)
    # display(result2.degrees)

    # println("maximum coefficient: $(maximum(result1.coeffs))")
end

run()

