using Test
include("../utils.jl")

# CUDA.jl's only data structure is the CuArray, so everything is implemented
# with arrays

# 1 + x
polynomial1 = [
    1 1
    2 0
    3 0
]

# 1 + y
polynomial2 = [
    1 0 2
    1 0 4
]

# Modified to move bit reversal algorithm outside
function CPUDFT(p, n, log2n, inverted = 1)
    for i in 1:log2n
        m = 1 << i
        m2 = m >> 1
        theta = complex(1,0)
        theta_m = cis(inverted * pi/m2)
        for j in 0:m2-1
            for k in j:m:n-1
                t = theta * p[k + m2 + 1]
                u = p[k + 1]

                p[k + 1] = u + t
                p[k + m2 + 1] = u - t
            end
            theta *= theta_m
        end
    end

    return p
end

function CPUIDFT(y, n, log2n)
    return CPUDFT(y, n, log2n, -1) ./ n
end

function print_matrix(matrix)
    for i in axes(matrix, 1)
        for j in axes(matrix, 2)
            print(matrix[i, j], "\t")
        end
        println()
    end
end

"""
    getHighestDegrees(polynomial)

Return the highest x-degree and highest y-degree in the polynomial
represented by the matrix

Assumes user isn't putting in an empty array
"""
function getHighestDegrees(polynomial)
    greatestx = 0
    greatesty = 0
    rowzeros = zeros(Int, size(polynomial, 2))
    colzeros = zeros(Int, size(polynomial, 1))

    for i in size(polynomial, 2):-1:1
        if polynomial[:, i] != colzeros
            greatestx = i
            break
        end
    end

    for j in size(polynomial, 1):-1:1
        if polynomial[j, :] != rowzeros
            greatesty = j
            break
        end
    end

    return (greatestx - 1, greatesty - 1)
end


function padMatrix(matrix, r, c)
    m = ComplexF32.(matrix)
    function addZeroRows(matrix, n)
        num_cols = size(matrix, 2)
        zerosr = zeros(ComplexF32, n, num_cols)
        return vcat(matrix, zerosr)
    end

    function addZeroCols(matrix, m)
        num_rows = size(matrix, 1)
        zerosc = zeros(ComplexF32, num_rows, m)
        return hcat(matrix, zerosc)
    end

    println("type of m: $(typeof(m))")
    return addZeroCols(addZeroRows(m, r), c)
end

function bitReverseCopy(input::AbstractArray, temp::AbstractArray, dims::Tuple, log2dims::Tuple)
    flag = true
    for dim in eachindex(dims)
        for i in 0:dims[dim] - 1
            rev = bitReverse(i, log2dims[i])
            if flag
                temp[]
            end
        end
    end
    for i in 0:rows-1
        rev = bitReverse(i, log2rows)
        output[i + 1, :] = input[rev + 1]
    end
    for j in 0:cols-1
        rev = bitReverse(j, log2cols)
        output[:, j + 1]
    end
end


function DFT2d(input::AbstractArray, dims::Tuple, log2dims::Tuple, inverted = 1)
    result = zeros(ComplexF32, dims)
    bitReverseCopy(input, result)

    if inverted == 1
        for i in axes(result, 2)
            result[:, i] .= CPUDFT(result[:, i], rows, log2rows)
        end

        for j in axes(result, 1)
            result[j, :] .= CPUDFT(result[j, :], cols, log2cols)
        end
    end

    if inverted == -1
        for i in axes(result, 2)
            result[:, i] .= CPUIDFT(result[:, i], rows, log2rows)
        end

        for j in axes(result, 1)
            result[j, :] .= CPUIDFT(result[j, :], cols, log2cols)
        end
    end

    return result
end

function IDFT2d(input::AbstractArray, rows, log2rows, cols, log2cols)
    return DFT2d(input, rows, log2rows, cols, log2cols, -1)
end

function multiply2d(polynomial1, polynomial2)
    # Figure out dimensions of resulting product, pad to 2^n power

    # resultSize stores (cols, rows)
    resultSize = getHighestDegrees(polynomial1) .+ getHighestDegrees(polynomial2) .+ 1

    rows = Int.(2^ceil(log2(resultSize[2])))
    cols = Int.(2^ceil(log2(resultSize[1])))

    # Scale polynomial1 and polynomial2 dimensions to 2^n power
    copyp1 = padMatrix(polynomial1, rows - size(polynomial1, 1), cols - size(polynomial1, 2))
    copyp2 = padMatrix(polynomial2, rows - size(polynomial2, 1), cols - size(polynomial2, 2))

    log2rows = UInt32(log2(rows));
    log2cols = UInt32(log2(cols));

    # DFT polynomial1 and polynomial2
    y1 = DFT2d(copyp1, rows, log2rows, cols, log2cols)
    y2 = DFT2d(copyp2, rows, log2rows, cols, log2cols)

    # Perform element-wise multiplication -> result
    result = IDFT2d(y1 .* y2, rows, log2rows, cols, log2cols)
    # IDFT result, broadcast dividing by dimensions
    return Int.(round.(real.(result[1:resultSize[2], 1:resultSize[1]])))
end

print_matrix(multiply2d(polynomial1, polynomial2))
