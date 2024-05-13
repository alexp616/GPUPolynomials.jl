include("GPUhypertriangleV2.jl")

using SparseArrays

k = 3
p = 3

sum = 0
for i in 0:div(k, p)
    sum += factorial(k) / ((factorial(i)) ^ (p) * factorial(k - p*i))
end

println(sum)

compositions = generate_compositions(k, k)
ones = fill(1, size(compositions, 1))
polynomial = hcat(ones, compositions)
arr = Array(raise_to_n(polynomial, p))


function encode_composition(arr, maxValue)
    encoded = 0
    for i in eachindex(arr)
        encoded += arr[i] * maxValue ^ i
    end

    return encoded
end

function encodepolynomial(polynomial, max_degree)
    encodedpolynomial = zeros(Int, size(polynomial, 1), 2)

    for i in axes(encodedpolynomial, 1)
        encodedpolynomial[i, 1] = polynomial[i, 1]
        encodedpolynomial[i, 2] = encode_composition(polynomial[i, 2:size(polynomial, 2)], max_degree)
    end
    return encodedpolynomial
end

encodedpolynomial = encodepolynomial(arr, k*p)

# println(encodedpolynomial)

values = encodedpolynomial[:, 1] 
keys = encodedpolynomial[:, 2]

S = sparsevec(keys, values)

println(maximum(S))