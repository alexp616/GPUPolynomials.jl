mutable struct SparsePolynomial{T<:Integer}
    coeffs::Vector{T}
    degrees::Vector{Int}
end

function generate_compositions(n, k, type::DataType = Int64)
    compositions = zeros(type, binomial(n + k - 1, k - 1), k)
    current_composition = zeros(type, k)
    current_composition[1] = n
    idx = 1
    while true
        compositions[idx, :] .= current_composition
        idx += 1
        v = current_composition[k]
        if v == n
            break
        end
        current_composition[k] = 0
        j = k - 1
        while 0 == current_composition[j]
            j -= 1
        end
        current_composition[j] -= 1
        current_composition[j + 1] = 1 + v
    end

    return compositions
end

function kron(seq, key)
    result = 0
    for i in eachindex(seq)
        result += seq[i] * key ^ (i - 1)
    end

    return result
end

function find_max_coeff(numVars, prime)
    mons = generate_compositions(numVars * (prime - 1), numVars)

    coeffs = fill(BigInt(prime - 1), size(mons, 1))
    degrees = mapslices(v -> kron(v, numVars * (prime - 1) * prime + 1), mons; dims = 2)
    degrees = vec(degrees)

    sp = SparsePolynomial(coeffs, degrees)

    result = trivial_power(sp, prime)

    return maximum(result.coeffs)
end

function trivial_multiply(p1::SparsePolynomial{T}, p2::SparsePolynomial{T}) where T<:Integer
    result_dict = Dict{Int, T}()

    for i in 1:length(p1.coeffs)
        for j in 1:length(p2.coeffs)
            new_coeff = p1.coeffs[i] * p2.coeffs[j]
            new_degree = p1.degrees[i] + p2.degrees[j]
            
            if haskey(result_dict, new_degree)
                result_dict[new_degree] += new_coeff
            else
                result_dict[new_degree] = new_coeff
            end
        end
    end

    sorted_degrees = sort(collect(keys(result_dict)))
    sorted_coeffs = [result_dict[d] for d in sorted_degrees]

    return SparsePolynomial(sorted_coeffs, sorted_degrees)
end

function trivial_power(p::SparsePolynomial{T}, n::Int) where T<:Integer
    if n == 0
        return SparsePolynomial([1], [0])
    elseif n == 1
        return p
    end

    result = p

    for _ in 2:n
        result = trivial_multiply(result, p)
    end

    return result
end

open("bounds.txt", "a") do io
    primes = [5, 7, 11, 13]
    for p in primes
        timetaken = @timed begin
            result = find_max_coeff(4, p)
        end

        println(io, "Max coeff of numVars = 4, prime = $p is $(result)")
        println(io, "This computation took $(timetaken.time) s \n")

        flush(io)
    end
end