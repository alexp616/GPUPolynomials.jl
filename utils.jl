"""
Just one method used globally right now, will add more if needed later.
Also doubles as storage for code I wrote earlier that isn't completely useless
"""


"""
    bitReverse(x, log2n)

Compute the reversed bitwise representation of integer x and return
the result.

Does not work when x >= n
"""
function bitReverse(x, log2n)
    temp = 0
    for i in 0:log2n-1
        temp <<= 1
        temp |= (x & 1)
        x >>= 1
    end
    return temp
end




# TODO figure out how to optimize this
function polynomialSquare(p)
    return iterativeMultiply(p, p)
end

function toBits(n)
    bits = [0 for i in 1:ceil(log2(n))]
    for i in eachindex(bits)
        bits[i] = n & 1
        n >>= 1
    end
    return bits
end

function polynomialPow(p, n)
    # Only takes positive integer n>=1
    bitarr = toBits(n)
    result = [1]
    temp = p
    for i in 1:length(bitarr)-1
        if i == 1
            result = iterativeMultiply(result, temp)
        temp = polynomialSquare(temp)
        end
    end
    if bitarr[end] == 1
        result = iterativeMultiply(result, temp)
    end
    return result
end