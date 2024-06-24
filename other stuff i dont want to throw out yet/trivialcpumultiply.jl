function to_bits(n)
    bits = [0 for _ in 1:floor(Int, log2(n)) + 1]
    for i in eachindex(bits)
        bits[i] = n & 1
        n >>= 1
    end
    return bits
end

function add_to_dict(dict, k, v)
    if haskey(dict, k)
        dict[k] += v
    else
        dict[k] = v
    end
end

function cpumultiply(p1, p2)
    p3 = Dict()
    for (key1, value1) in p1
        for (key2, value2) in p2
            add_to_dict(p3, key1 .+ key2, value1 * value2)
        end
    end

    return p3
end

function raise_to_power(p, n::Int)
    # Only takes positive integer n>=1
    bitarr = to_bits(n)

    result = Dict{}((0, 0, 0, 0) => 1)
    temp = p

    for i in 1:length(bitarr) - 1
        if bitarr[i] == 1
            result = cpumultiply(temp, result)
        end
        temp = cpumultiply(temp, temp)
        println("this thing is actually running")
    end

    result = cpumultiply(temp, result)

    return result
end

degrees = map(row -> Tuple(row), eachrow([4 0 0 0; 3 1 0 0; 3 0 1 0; 3 0 0 1; 2 2 0 0; 2 1 1 0; 2 1 0 1; 2 0 2 0; 2 0 1 1; 2 0 0 2; 1 3 0 0; 1 2 1 0; 1 2 0 1; 1 1 2 0; 1 1 1 1; 1 1 0 2; 1 0 3 0; 1 0 2 1; 1 0 1 2; 1 0 0 3; 0 4 0 0; 0 3 1 0; 0 3 0 1; 0 2 2 0; 0 2 1 1; 0 2 0 2; 0 1 3 0; 0 1 2 1; 0 1 1 2; 0 1 0 3; 0 0 4 0; 0 0 3 1; 0 0 2 2; 0 0 1 3; 0 0 0 4]))
coeffs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

testpoly = Dict{Tuple, Int64}(zip(degrees, coeffs))

@time poly2 = raise_to_power(testpoly, 4)
poly2
@time poly3 = raise_to_power(poly2, 5)